import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool

DATASET_DIR = r"riot_dump/BR1/challenger/ranked_solo_420/dataset_ts_csv"
MINUTO_CORTE = 10
TEST_SIZE = 0.10
VAL_SIZE = 0.15       # 15% do treino vira validação
RANDOM_STATE = 42

# -------------------------
# Snapshot por partida
# -------------------------
def load_snapshot_per_match(csv_path: str, minute_cutoff: int) -> pd.Series | None:
    df = pd.read_csv(csv_path)

    if df.empty or "t_min" not in df.columns:
        return None

    df = df.sort_values("t_min")
    eligible = df[df["t_min"] <= minute_cutoff]
    row = eligible.iloc[-1] if len(eligible) > 0 else df.iloc[0]
    return row


def build_match_level_table(dataset_dir: str, minute_cutoff: int) -> pd.DataFrame:
    rows = []

    for csv_path in glob.glob(os.path.join(dataset_dir, "*.csv")):
        df = pd.read_csv(csv_path)
        if df["t_min"].max() <= 3:
            continue

        row = load_snapshot_per_match(csv_path, minute_cutoff)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("Nenhum CSV válido encontrado.")

    df = pd.DataFrame(rows)

    if "matchId" not in df.columns:
        raise RuntimeError("matchId não encontrado.")

    df = df.drop_duplicates(subset=["matchId"]).reset_index(drop=True)
    return df


# -------------------------
# Preparação: X/y + cats
# -------------------------
def prepare_Xy(df_match: pd.DataFrame):
    y = df_match["y_blue_win"].astype(int)

    drop_cols = [
        c for c in ["matchId", "game_datetime_utc", "gameCreation_ms", "y_blue_win"]
        if c in df_match.columns
    ]
    if "patch"          in df_match.columns: drop_cols.append("patch")
    if "gameDuration_s" in df_match.columns: drop_cols.append("gameDuration_s")
    if "t_min"          in df_match.columns: drop_cols.append("t_min")

    X = df_match.drop(columns=drop_cols, errors="ignore").copy()

    cat_cols = [c for c in X.columns if c.endswith("_championName") or c.endswith("_individualPosition")]
    for c in cat_cols:
        X[c] = X[c].fillna("").astype(str)

    num_cols = [c for c in X.columns if c not in cat_cols]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y, cat_cols


# -------------------------
# Split com verificação anti-leakage
# -------------------------
def stratified_split(df: pd.DataFrame, y: pd.Series, test_size: float, label: str) -> tuple:
    """Retorna (df_main, df_holdout) com garantia de matchId disjunto."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx = list(range(len(df)))
    train_idx, holdout_idx = next(splitter.split(idx, y))

    df_main    = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    # Verificação anti-leakage
    main_ids    = set(df_main["matchId"].astype(str))
    holdout_ids = set(df_holdout["matchId"].astype(str))
    inter = main_ids.intersection(holdout_ids)
    if inter:
        raise RuntimeError(f"Leakage ({label}): matchId repetido (ex.: {list(inter)[:3]})")

    print(f"  [{label}] main={len(df_main)} | holdout={len(df_holdout)}")
    return df_main, df_holdout


# -------------------------
# Treino + Avaliação
# -------------------------
def run_catboost_test():
    print("Carregando dados...")
    df_match = build_match_level_table(DATASET_DIR, MINUTO_CORTE)
    y_full   = df_match["y_blue_win"].astype(int)

    print(f"\nTotal de partidas: {len(df_match)}")
    print(f"Balanceamento geral:\n{y_full.value_counts(normalize=True).round(4)}\n")

    # ── 1. Separa teste (nunca tocado até avaliação final) ──────────────────
    print("Splits:")
    df_trainval, df_test = stratified_split(df_match, y_full, TEST_SIZE, "treino+val / teste")

    # ── 2. Separa validação do bloco de treino ──────────────────────────────
    y_trainval = df_trainval["y_blue_win"].astype(int)
    df_train, df_val = stratified_split(df_trainval, y_trainval, VAL_SIZE, "treino / val")

    # ── 3. Prepara X/y para cada split ─────────────────────────────────────
    X_train, y_train, cat_cols = prepare_Xy(df_train)
    X_val,   y_val,   _        = prepare_Xy(df_val)
    X_test,  y_test,  _        = prepare_Xy(df_test)

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)   # ← early stopping
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)   # ← avaliação final

    # ── 4. Treino ───────────────────────────────────────────────────────────
    print("\nTreinando...")
    model = CatBoostClassifier(random_seed=RANDOM_STATE)
    model.fit(
        train_pool,
        eval_set=val_pool,          
        use_best_model=True,
        verbose=100,
    )

    # ── 5. Avaliação final (teste virgem) ───────────────────────────────────
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
    run_catboost_test()
