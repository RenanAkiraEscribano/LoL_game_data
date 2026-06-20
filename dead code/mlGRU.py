import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET_DIR  = r"riot_dump/BR1/challenger/ranked_solo_420/dataset_ts_csv"
MINUTO_CORTE = 10
MIN_COVERAGE = 0.5
TEST_SIZE    = 0.10
VAL_SIZE     = 0.15
RANDOM_STATE = 42
BATCH_SIZE   = 128         # era 64  — treino mais rápido para 11k samples
EPOCHS       = 150         # era 200 — suficiente para convergir com 11k samples
LR           = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT_GRU  = 0.1         # era 0.2 — dataset grande o suficiente para relaxar
DROPOUT_HEAD = 0.3         # era 0.4 — idem
WEIGHT_DECAY = 1e-4        # antes hardcoded no optimizer
GRAD_CLIP    = 1.0         # antes hardcoded no clip_grad_norm_
CHECKPOINT   = "best_model.pt"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_COLS = {"matchId", "game_datetime_utc", "gameCreation_ms",
             "y_blue_win", "patch", "gameDuration_s"}


# ─────────────────────────────────────────────
# 1. Carregamento — sequência 3D por partida
# ─────────────────────────────────────────────
def load_sequence(csv_path: str, minute_cutoff: int) -> dict | None:
    df = pd.read_csv(csv_path)

    if df.empty or "t_min" not in df.columns or "y_blue_win" not in df.columns:
        return None

    df = df.sort_values("t_min").reset_index(drop=True)
    df = df[df["t_min"] <= minute_cutoff]
    if df.empty:
        return None

    min_real = int(minute_cutoff * MIN_COVERAGE)
    if df["t_min"].max() < min_real:
        return None

    label    = int(df["y_blue_win"].iloc[0])
    match_id = str(df["matchId"].iloc[0]) if "matchId" in df.columns else csv_path

    feature_cols = [c for c in df.columns if c not in DROP_COLS | {"t_min"}]

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
        "seq_df"      : seq_df,
        "feature_cols": feature_cols,
    }


def build_dataset(dataset_dir: str, minute_cutoff: int):
    records = []
    for csv_path in glob.glob(os.path.join(dataset_dir, "*.csv")):
        rec = load_sequence(csv_path, minute_cutoff)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("Nenhum CSV válido encontrado.")

    seen = set()
    records = [r for r in records
               if not (r["match_id"] in seen or seen.add(r["match_id"]))]

    feature_cols = records[0]["feature_cols"]
    cat_cols = [c for c in feature_cols
                if "_championName" in c or "_individualPosition" in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return records, cat_cols, num_cols


# ─────────────────────────────────────────────
# 2. Encoder — fit APENAS no treino
# ─────────────────────────────────────────────
def dynamic_emb_dim(vocab_size: int) -> int:
    """
    Dimensão de embedding proporcional ao vocabulário.
    Regra prática: min(50, (vocab + 1) // 2)
      - Campeões (~170 vocab) → 50
      - Posições (~6 vocab)   →  3
    """
    return min(50, (vocab_size + 1) // 2)


class FeatureEncoder:
    """
    - Numéricas   → StandardScaler  → float tensor
    - Categóricas → LabelEncoder    → int tensor (índices p/ Embedding)
    """
    def __init__(self, cat_cols: list, num_cols: list):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.les      = {c: LabelEncoder() for c in cat_cols}
        self.scaler   = StandardScaler()
        self._fitted  = False

    @property
    def cat_vocab_sizes(self) -> list[int]:
        assert self._fitted
        return [len(self.les[c].classes_) for c in self.cat_cols]

    @property
    def cat_emb_dims(self) -> list[int]:
        """Dimensão dinâmica de embedding por coluna categórica."""
        return [dynamic_emb_dim(v) for v in self.cat_vocab_sizes]

    def fit(self, records: list):
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

    def transform(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna:
          x_num → (T, n_num)   float32  — numéricas normalizadas
          x_cat → (T, n_cat)   int64    — índices inteiros p/ Embedding
        """
        assert self._fitted
        df = rec["seq_df"].copy()

        # Numéricas
        if self.num_cols:
            x_num = self.scaler.transform(
                df[self.num_cols].values.astype(float)
            ).astype(np.float32)
        else:
            x_num = np.zeros((len(df), 0), dtype=np.float32)

        # Categóricas → inteiros
        cat_encoded = []
        for c in self.cat_cols:
            vals = df[c].fillna("").astype(str).tolist() if c in df.columns \
                   else ["__unknown__"] * len(df)
            encoded = [
                self.les[c].transform([v])[0]
                if v in self.les[c].classes_
                else self.les[c].transform(["__unknown__"])[0]
                for v in vals
            ]
            cat_encoded.append(np.array(encoded, dtype=np.int64).reshape(-1, 1))

        x_cat = np.hstack(cat_encoded) if cat_encoded \
                else np.zeros((len(df), 0), dtype=np.int64)

        return x_num, x_cat


# ─────────────────────────────────────────────
# 3. Dataset PyTorch
# ─────────────────────────────────────────────
class LoLSequenceDataset(Dataset):
    def __init__(self, records: list, encoder: FeatureEncoder):
        self.x_nums = []
        self.x_cats = []
        self.labels = []

        for rec in records:
            x_num, x_cat = encoder.transform(rec)
            self.x_nums.append(torch.tensor(x_num, dtype=torch.float32))
            self.x_cats.append(torch.tensor(x_cat, dtype=torch.long))
            self.labels.append(rec["label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.x_nums[idx],
            self.x_cats[idx],
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ─────────────────────────────────────────────
# 4. Modelo GRU com Embeddings dinâmicos + Atenção
# ─────────────────────────────────────────────
class LoLGRU(nn.Module):
    """
    v3 — melhorias sobre v2:
      - EMB_DIM dinâmico por coluna (proporcional ao vocabulário)
      - DROPOUT_GRU e DROPOUT_HEAD separados
      - HIDDEN_SIZE=128
    """
    def __init__(
        self,
        num_size        : int,
        cat_vocab_sizes : list[int],
        cat_emb_dims    : list[int],
        hidden_size     : int   = HIDDEN_SIZE,
        num_layers      : int   = NUM_LAYERS,
        dropout_gru     : float = DROPOUT_GRU,
        dropout_head    : float = DROPOUT_HEAD,
    ):
        super().__init__()

        # Embedding dinâmico por coluna categórica
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, emb_dim)
            for vocab, emb_dim in zip(cat_vocab_sizes, cat_emb_dims)
        ])

        gru_input = num_size + sum(cat_emb_dims)

        # GRU causal (bidirectional=False)
        self.gru = nn.GRU(
            input_size   = gru_input,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout_gru if num_layers > 1 else 0.0,
            bidirectional= False,
        )

        # Atenção temporal
        self.attn = nn.Linear(hidden_size, 1)

        # Cabeça de classificação
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(32, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb_parts = [emb(x_cat[..., i]) for i, emb in enumerate(self.embeddings)]

        x = torch.cat([x_num] + emb_parts, dim=-1) if emb_parts else x_num

        out, _ = self.gru(x)                         # (batch, T, hidden)
        scores  = self.attn(out)                     # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)       # (batch, T, 1)
        context = (weights * out).sum(dim=1)         # (batch, hidden)

        return self.head(context).squeeze(1)         # (batch,)


# ─────────────────────────────────────────────
# 5. Split estratificado anti-leakage
# ─────────────────────────────────────────────
def stratified_split(records: list, test_size: float, label: str):
    y = np.array([r["label"] for r in records])
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_STATE
    )
    train_idx, holdout_idx = next(splitter.split(range(len(records)), y))

    main    = [records[i] for i in train_idx]
    holdout = [records[i] for i in holdout_idx]

    inter = {r["match_id"] for r in main} & {r["match_id"] for r in holdout}
    if inter:
        raise RuntimeError(f"Leakage ({label}): {list(inter)[:3]}")

    print(f"  [{label}] principal={len(main)} | holdout={len(holdout)}")
    return main, holdout


# ─────────────────────────────────────────────
# 6. Treino / Avaliação
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0.0

    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()          # OneCycleLR: passo por batch
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, probs, targets = [], [], []

    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        logits = model(x_num, x_cat)
        total_loss += criterion(logits, y).item() * len(y)

        prob = torch.sigmoid(logits).cpu()
        probs.extend(prob.tolist())
        preds.extend((prob >= 0.5).int().tolist())
        targets.extend(y.cpu().int().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(targets, preds)
    return avg_loss, acc, preds, probs, targets


# ─────────────────────────────────────────────
# 7. Pipeline principal
# ─────────────────────────────────────────────
def run_gru():
    print(f"Device: {DEVICE}")
    print(f"Carregando sequências até t={MINUTO_CORTE}...\n")

    records, cat_cols, num_cols = build_dataset(DATASET_DIR, MINUTO_CORTE)
    print(f"Partidas válidas     : {len(records)}")
    print(f"Features numéricas   : {len(num_cols)}")
    print(f"Features categóricas : {len(cat_cols)}")

    # ── Splits ───────────────────────────────────────────────────────────────
    print("\nSplits:")
    trainval, test_recs  = stratified_split(records,  TEST_SIZE, "treino+val / teste")
    train_recs, val_recs = stratified_split(trainval, VAL_SIZE,  "treino / val")

    # ── Encoder (fit apenas no treino) ───────────────────────────────────────
    encoder = FeatureEncoder(cat_cols, num_cols)
    encoder.fit(train_recs)

    cat_vocab_sizes = encoder.cat_vocab_sizes
    cat_emb_dims    = encoder.cat_emb_dims

    print("\nEmbeddings categóricos:")
    for col, vocab, emb in zip(cat_cols, cat_vocab_sizes, cat_emb_dims):
        print(f"  {col:40s} vocab={vocab:4d} → emb_dim={emb}")

    # ── Datasets e Loaders ───────────────────────────────────────────────────
    train_ds = LoLSequenceDataset(train_recs, encoder)
    val_ds   = LoLSequenceDataset(val_recs,   encoder)
    test_ds  = LoLSequenceDataset(test_recs,  encoder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── pos_weight para desbalanceamento ─────────────────────────────────────
    n_neg = sum(1 for r in train_recs if r["label"] == 0)
    n_pos = sum(1 for r in train_recs if r["label"] == 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"\nDistribuição treino — neg={n_neg} | pos={n_pos} "
          f"| pos_weight={pos_weight.item():.3f}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = LoLGRU(
        num_size        = len(num_cols),
        cat_vocab_sizes = cat_vocab_sizes,
        cat_emb_dims    = cat_emb_dims,
        hidden_size     = HIDDEN_SIZE,
        num_layers      = NUM_LAYERS,
        dropout_gru     = DROPOUT_GRU,
        dropout_head    = DROPOUT_HEAD,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # OneCycleLR — sem early stopping, roda as EPOCHS completas
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = LR,
        steps_per_epoch = len(train_loader),
        epochs          = EPOCHS,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo : {n_params:,}")
    print(f"Treino={len(train_recs)} | Val={len(val_recs)} | Teste={len(test_recs)}\n")

    # ── Loop de treino — sem early stopping ──────────────────────────────────
    # O OneCycleLR decai o LR até eta_min ao fim das EPOCHS,
    # congelando o modelo gradualmente. O melhor checkpoint é
    # salvo a cada melhora de val_loss e usado na avaliação final.
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        val_loss, val_acc, _, val_probs, val_targets = eval_epoch(
            model, val_loader, criterion
        )
        val_auc = roc_auc_score(val_targets, val_probs)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} "
              f"| val_auc={val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : best_val_loss,
                "val_auc"    : val_auc,
                "config": {
                    "num_size"        : len(num_cols),
                    "cat_vocab_sizes" : cat_vocab_sizes,
                    "cat_emb_dims"    : cat_emb_dims,
                    "hidden_size"     : HIDDEN_SIZE,
                    "num_layers"      : NUM_LAYERS,
                    "dropout_gru"     : DROPOUT_GRU,
                    "dropout_head"    : DROPOUT_HEAD,
                },
            }, CHECKPOINT)
            print(f"  ✓ Checkpoint salvo (epoch {epoch})")

    # ── Avaliação final com melhor checkpoint ────────────────────────────────
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    print(f"\nCheckpoint carregado: epoch {ckpt['epoch']} "
          f"| val_loss={ckpt['val_loss']:.4f} | val_auc={ckpt['val_auc']:.4f}")

    _, _, y_pred, y_prob, y_true = eval_epoch(model, test_loader, criterion)

    print("\n==== RESULTADOS FINAIS (teste virgem) ====")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}\n")
    print("Matriz de confusão:\n", confusion_matrix(y_true, y_pred))
    print("\nRelatório:\n", classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    run_gru()
