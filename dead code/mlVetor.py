import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool

DATASET_DIR = r"riot_dump/BR1/challenger/ranked_solo_420/dataset_ts_csv"
MINUTO_CORTE = 10
TEST_SIZE    = 0.10
VAL_SIZE     = 0.15
RANDOM_STATE = 42

# Colunas que NUNCA entram no vetor (leakage ou identificadores)
DROP_COLS = {"matchId", "game_datetime_utc", "gameCreation_ms",
             "y_blue_win", "patch", "gameDuration_s", "t_min"}

# -------------------------
# Vetor achatado t=1..T
# -------------------------
def build_flat_vector(csv_path: str, minute_cutoff: int) -> dict | None:
    """
    Lê um CSV de partida e retorna um dict com:
      - matchId, y_blue_win
      - para cada feature f e cada minuto t=1..T: f__t{t}

    Minutos ausentes são preenchidos com forward-fill (último valor conhecido).
    Minutos além de T são descartados — sem leakage futuro.
    """
    df = pd.read_csv(csv_path)

    if df.empty or "t_min" not in df.columns or "y_blue_win" not in df.columns:
        return None

    df = df.sort_values("t_min").reset_index(drop=True)

    # Descarta tudo após T  ← sem leakage de dados futuros
    df = df[df["t_min"] <= minute_cutoff]
    if df.empty:
        return None

    # Colunas de feature (exclui leakage e meta-colunas)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    # Reindexar para ter uma linha por minuto inteiro de 1..T
    # Forward-fill: se o minuto não existe, usa o último snapshot disponível
    minute_index = pd.RangeIndex(1, minute_cutoff + 1, name="t_min")
    df = (
        df.set_index("t_min")[feature_cols]
          .reindex(minute_index)
          .ffill()           # propaga último valor conhecido para frente
          .bfill()           # preenche início se t=1 estiver faltando
          .fillna(0)         # segurança: zeros se a partida for muito curta
    )

    # Achata: feature f no minuto t → coluna "f__t{t}"
    flat = {}
    for t in range(1, minute_cutoff + 1):
        for col in feature_cols:
            flat[f"{col}__t{t}"] = df.loc[t, col] if t in df.index else 0.0

    # Meta-informações (não entram no X)
    original = pd.read_csv(csv_path)
    flat["matchId"]    = str(original["matchId"].iloc[0])
    flat["y_blue_win"] = int(original["y_blue_win"].iloc[0])

    return flat


def build_flat_table(dataset_dir: str, minute_cutoff: int) -> pd.DataFrame:
    rows = []

    for csv_path in glob.glob(os.path.join(dataset_dir, "*.csv")):
        raw = pd.read_csv(csv_path)

        # Descarta rematch muito curtas
        if "t_min" not in raw.columns or raw["t_min"].max() <= 3:
            continue

        vec = build_flat_vector(csv_path, minute_cutoff)
        if vec is not None:
            rows.append(vec)

    if not rows:
        raise RuntimeError("Nenhum CSV válido encontrado.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["matchId"]).reset_index(drop=True)
    return df


# -------------------------
# Preparação X/y
# -------------------------
def prepare_Xy(df: pd.DataFrame):
    y = df["y_blue_win"].astype(int)

    # Remove identificadores e target
    meta_cols = [c for c in ["matchId", "y_blue_win"] if c in df.columns]
    X = df.drop(columns=meta_cols, errors="ignore").copy()

    # Categóricas: colunas que contêm nome de campeão ou posição achatadas
    cat_cols = [c for c in X.columns
                if "_championName__t" in c or "_individualPosition__t" in c]

    for c in cat_cols:
        X[c] = X[c].fillna("").astype(str)

    num_cols = [c for c in X.columns if c not in cat_cols]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y, cat_cols


# -------------------------
# Split com anti-leakage
# -------------------------
def stratified_split(df: pd.DataFrame, y: pd.Series,
                     test_size: float, label: str) -> tuple:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_STATE
    )
    train_idx, holdout_idx = next(splitter.split(range(len(df)), y))

    df_main    = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    # Verificação explícita: matchId disjunto
    inter = set(df_main["matchId"]) & set(df_holdout["matchId"])
    if inter:
        raise RuntimeError(
            f"Leakage ({label}): matchId em treino e holdout! ex.: {list(inter)[:3]}"
        )

    print(f"  [{label}] principal={len(df_main)} | holdout={len(df_holdout)}")
    return df_main, df_holdout


# -------------------------
# Treino + Avaliação
# -------------------------
def run_catboost_flat():
    print(f"Construindo vetores achatados até t={MINUTO_CORTE}...")
    df = build_flat_table(DATASET_DIR, MINUTO_CORTE)

    n_features_por_minuto = len([c for c in df.columns
                                  if c.endswith("__t1")])
    print(f"\nPartidas carregadas : {len(df)}")
    print(f"Minutos no vetor    : {MINUTO_CORTE}")
    print(f"Features por minuto : {n_features_por_minuto}")
    print(f"Dimensão total de X : {n_features_por_minuto * MINUTO_CORTE}")
    print(f"Balanceamento:\n{df['y_blue_win'].value_counts(normalize=True).round(4)}\n")

    y_full = df["y_blue_win"].astype(int)

    # ── Splits ──────────────────────────────────────────────────────────────
    print("Splits:")
    df_trainval, df_test    = stratified_split(df,          y_full,                          TEST_SIZE, "treino+val / teste")
    df_train,    df_val     = stratified_split(df_trainval, df_trainval["y_blue_win"].astype(int), VAL_SIZE,  "treino / val")

    # ── X/y ─────────────────────────────────────────────────────────────────
    X_train, y_train, cat_cols = prepare_Xy(df_train)
    X_val,   y_val,   _        = prepare_Xy(df_val)
    X_test,  y_test,  _        = prepare_Xy(df_test)

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)   # early stopping
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)   # avaliação final

    # ── Treino ───────────────────────────────────────────────────────────────
    print("\nTreinando CatBoost...")
    model = CatBoostClassifier(random_seed=RANDOM_STATE)
    model.fit(
        train_pool,
        eval_set=val_pool,       # ✅ val, nunca o teste
        use_best_model=True,
        verbose=100,
    )

    # ── Avaliação final ───────────────────────────────────────────────────────
    y_pred = model.predict(test_pool).astype(int).reshape(-1)

    print("\n==== RESULTADOS (teste virgem) ====")
    print(f"  Treino : {len(df_train):>5} partidas")
    print(f"  Val    : {len(df_val):>5} partidas  ← early stopping")
    print(f"  Teste  : {len(df_test):>5} partidas  ← avaliação final\n")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}\n")
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório:\n", classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    run_catboost_flat()
