import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET_DIR  = r"riot_dump/BR1/challenger/ranked_solo_420/dataset_ts_csv"
MINUTO_CORTE = 10
MIN_COVERAGE = 0.5        # mínimo de minutos reais vs T
TEST_SIZE    = 0.10
VAL_SIZE     = 0.15
RANDOM_STATE = 42
BATCH_SIZE   = 64
EPOCHS       = 200
PATIENCE     = 20          # early stopping por val_loss
LR           = 1e-3
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_COLS = {"matchId", "game_datetime_utc", "gameCreation_ms",
             "y_blue_win", "patch", "gameDuration_s"}

# ─────────────────────────────────────────────
# 1. Carregamento — sequência 3D por partida
# ─────────────────────────────────────────────
def load_sequence(csv_path: str, minute_cutoff: int) -> dict | None:
    """
    Retorna:
      seq   → np.ndarray (T, n_features)  apenas minutos 1..T
      label → int
      match_id → str
    """
    df = pd.read_csv(csv_path)

    if df.empty or "t_min" not in df.columns or "y_blue_win" not in df.columns:
        return None

    df = df.sort_values("t_min").reset_index(drop=True)

    # ── Corte temporal: sem dados além de T ──────────────────────────────────
    df = df[df["t_min"] <= minute_cutoff]
    if df.empty:
        return None

    # ── Filtro de cobertura mínima ───────────────────────────────────────────
    min_real = int(minute_cutoff * MIN_COVERAGE)
    if df["t_min"].max() < min_real:
        return None

    label    = int(df["y_blue_win"].iloc[0])
    match_id = str(df["matchId"].iloc[0]) if "matchId" in df.columns else csv_path

    feature_cols = [c for c in df.columns if c not in DROP_COLS | {"t_min"}]

    # ── Reindexar 1..T com forward-fill (sem inventar dados futuros) ─────────
    minute_idx = pd.RangeIndex(1, minute_cutoff + 1, name="t_min")
    seq_df = (
        df.set_index("t_min")[feature_cols]
          .reindex(minute_idx)
          .ffill()
          .bfill()
          .fillna(0)
    )

    return {
        "match_id"    : match_id,
        "label"       : label,
        "seq_df"      : seq_df,          # (T, n_features) como DataFrame
        "feature_cols": feature_cols,
    }


def build_dataset(dataset_dir: str, minute_cutoff: int):
    """
    Retorna:
      records   → list of dicts {match_id, label, seq_df, feature_cols}
      cat_cols  → list de colunas categóricas
      num_cols  → list de colunas numéricas
    """
    records = []

    for csv_path in glob.glob(os.path.join(dataset_dir, "*.csv")):
        rec = load_sequence(csv_path, minute_cutoff)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("Nenhum CSV válido encontrado.")

    # Deduplica por match_id
    seen = set()
    records = [r for r in records if not (r["match_id"] in seen or seen.add(r["match_id"]))]

    feature_cols = records[0]["feature_cols"]
    cat_cols = [c for c in feature_cols
                if "_championName" in c or "_individualPosition" in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return records, cat_cols, num_cols


# ─────────────────────────────────────────────
# 2. Encoders — fit APENAS no treino
# ─────────────────────────────────────────────
class FeatureEncoder:
    """
    Encoda categóricas com LabelEncoder e normaliza numéricas com StandardScaler.
    Fit feito exclusivamente no treino — aplicado em val/teste.
    """
    def __init__(self, cat_cols, num_cols):
        self.cat_cols  = cat_cols
        self.num_cols  = num_cols
        self.les       = {c: LabelEncoder() for c in cat_cols}
        self.scaler    = StandardScaler()
        self._fitted   = False

    def fit(self, records: list):
        # Coleta todos os valores de treino
        cat_data = {c: [] for c in self.cat_cols}
        num_rows = []

        for rec in records:
            df = rec["seq_df"]
            for c in self.cat_cols:
                if c in df.columns:
                    cat_data[c].extend(df[c].fillna("").astype(str).tolist())
            if self.num_cols:
                num_rows.append(df[self.num_cols].values)

        for c in self.cat_cols:
            self.les[c].fit(cat_data[c] + ["__unknown__"])

        if num_rows:
            self.scaler.fit(np.vstack(num_rows))

        self._fitted = True

    def transform(self, rec: dict) -> np.ndarray:
        """Retorna (T, n_encoded_features)."""
        assert self._fitted, "Chame .fit() antes de .transform()"
        df = rec["seq_df"].copy()
        parts = []

        # Numéricas normalizadas
        if self.num_cols:
            num_vals = df[self.num_cols].values.astype(float)
            parts.append(self.scaler.transform(num_vals))

        # Categóricas → inteiro (embedding será feito no modelo)
        for c in self.cat_cols:
            if c in df.columns:
                vals = df[c].fillna("").astype(str).tolist()
                encoded = []
                for v in vals:
                    if v in self.les[c].classes_:
                        encoded.append(self.les[c].transform([v])[0])
                    else:
                        encoded.append(self.les[c].transform(["__unknown__"])[0])
                parts.append(np.array(encoded, dtype=float).reshape(-1, 1))

        return np.hstack(parts).astype(np.float32)   # (T, n_features_encoded)


# ─────────────────────────────────────────────
# 3. Dataset PyTorch
# ─────────────────────────────────────────────
class LoLSequenceDataset(Dataset):
    def __init__(self, records: list, encoder: FeatureEncoder):
        self.sequences = [encoder.transform(r) for r in records]
        self.labels    = [r["label"] for r in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),  # (T, F)
            torch.tensor(self.labels[idx],    dtype=torch.float32),
        )


# ─────────────────────────────────────────────
# 4. Modelo GRU
# ─────────────────────────────────────────────
class LoLGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = False,      # ← causal: não vê minutos futuros
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # sem Sigmoid aqui — BCEWithLogitsLoss é mais estável numericamente
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_size)
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        last   = h_n[-1]              # última camada: (batch, hidden)
        return self.head(last).squeeze(1)


# ─────────────────────────────────────────────
# 5. Split — por matchId, anti-leakage
# ─────────────────────────────────────────────
def stratified_split(records: list, test_size: float, label: str):
    y = np.array([r["label"] for r in records])
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_STATE
    )
    train_idx, holdout_idx = next(splitter.split(range(len(records)), y))

    main    = [records[i] for i in train_idx]
    holdout = [records[i] for i in holdout_idx]

    # Verificação explícita de matchId
    main_ids    = {r["match_id"] for r in main}
    holdout_ids = {r["match_id"] for r in holdout}
    inter = main_ids & holdout_ids
    if inter:
        raise RuntimeError(f"Leakage ({label}): {list(inter)[:3]}")

    print(f"  [{label}] principal={len(main)} | holdout={len(holdout)}")
    return main, holdout


# ─────────────────────────────────────────────
# 6. Treino
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        preds.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
        targets.extend(y.cpu().int().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    return avg_loss, acc, preds, targets


# ─────────────────────────────────────────────
# 7. Pipeline principal
# ─────────────────────────────────────────────
def run_gru():
    print(f"Device: {DEVICE}")
    print(f"Carregando sequências até t={MINUTO_CORTE}...\n")

    records, cat_cols, num_cols = build_dataset(DATASET_DIR, MINUTO_CORTE)
    print(f"Partidas válidas    : {len(records)}")
    print(f"Features numéricas  : {len(num_cols)}")
    print(f"Features categóricas: {len(cat_cols)}")
    print(f"Dimensão por timestep (total): {len(num_cols) + len(cat_cols)}")

    # ── Splits ───────────────────────────────────────────────────────────────
    print("\nSplits:")
    trainval, test_recs = stratified_split(records, TEST_SIZE, "treino+val / teste")
    train_recs, val_recs = stratified_split(trainval, VAL_SIZE,  "treino / val")

    # ── Encoder — fit SOMENTE no treino ──────────────────────────────────────
    encoder = FeatureEncoder(cat_cols, num_cols)
    encoder.fit(train_recs)

    # ── Datasets e Loaders ───────────────────────────────────────────────────
    train_ds = LoLSequenceDataset(train_recs, encoder)
    val_ds   = LoLSequenceDataset(val_recs,   encoder)
    test_ds  = LoLSequenceDataset(test_recs,  encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Modelo ───────────────────────────────────────────────────────────────
    input_size = len(num_cols) + len(cat_cols)
    model      = LoLGRU(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion  = nn.BCEWithLogitsLoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    print(f"\nModelo: {sum(p.numel() for p in model.parameters())} parâmetros")
    print(f"Treino: {len(train_recs)} | Val: {len(val_recs)} | Teste: {len(test_recs)}\n")

    # ── Loop de treino com early stopping na val_loss ─────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping na epoch {epoch} (patience={PATIENCE})")
                break

    # ── Avaliação final com melhor checkpoint ────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE)

    _, _, y_pred, y_true = eval_epoch(model, test_loader, criterion)

    print("\n==== RESULTADOS FINAIS (teste virgem) ====")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}\n")
    print("Matriz de confusão:\n", confusion_matrix(y_true, y_pred))
    print("\nRelatório:\n", classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    run_gru()
