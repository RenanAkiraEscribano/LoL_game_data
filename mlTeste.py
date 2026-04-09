import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool


from sklearn.ensemble import RandomForestClassifier

DATASET_DIR = r"riot_dump/BR1/challenger/ranked_solo_420/dataset_ts_csv"
MINUTO_CORTE = 10
TEST_SIZE = 0.10
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

    drop_cols = [c for c in ["matchId", "game_datetime_utc", "gameCreation_ms", "y_blue_win"] if c in df_match.columns]

    # Evitar vazamento/atalhos
    if "patch" in df_match.columns:
        drop_cols.append("patch")
    if "gameDuration_s" in df_match.columns:
        drop_cols.append("gameDuration_s")
    if "t_min" in df_match.columns:
        drop_cols.append("t_min")

    X = df_match.drop(columns=drop_cols, errors="ignore").copy()

    # Definir colunas categóricas por regra de nome (mais robusto que dtype)
    cat_cols = [c for c in X.columns if c.endswith("_championName") or c.endswith("_individualPosition")]

    # Garantir tipo string para cats (train/test)
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna("").astype(str)

    # Garantir numéricos coerentes
    num_cols = [c for c in X.columns if c not in cat_cols]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y, cat_cols

# -------------------------
# Split estratificado por partida
# -------------------------
def stratified_split_by_match(df_match: pd.DataFrame, y: pd.Series):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    idx = list(range(len(df_match)))
    train_idx, test_idx = next(splitter.split(idx, y))

    df_train = df_match.iloc[train_idx].reset_index(drop=True)
    df_test = df_match.iloc[test_idx].reset_index(drop=True)

    # Anti-leakage: matchId disjunto
    train_ids = set(df_train["matchId"].astype(str))
    test_ids = set(df_test["matchId"].astype(str))
    inter = train_ids.intersection(test_ids)
    if inter:
        raise RuntimeError(f"Leakage: matchId repetido em treino e teste (ex.: {list(inter)[:3]})")

    return df_train, df_test

# -------------------------
# Treino + Avaliação
# -------------------------
def run_catboost_test():
    df_match = build_match_level_table(DATASET_DIR, MINUTO_CORTE)
    y_full = df_match["y_blue_win"].astype(int)

    df_train, df_test = stratified_split_by_match(df_match, y_full)

    X_train, y_train, cat_cols = prepare_Xy(df_train)
    X_test, y_test, _ = prepare_Xy(df_test)

    # ✅ CatBoost: passar cat_features por NOME (evita índices errados)
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        random_seed=RANDOM_STATE,
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_pred = model.predict(test_pool).astype(int).reshape(-1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n==== RESULTADOS ====")
    print(f"Partidas treino: {len(df_train)} | Partidas teste: {len(df_test)}")
    print("Balanceamento (treino):\n", df_train["y_blue_win"].value_counts(normalize=True))
    print("Balanceamento (teste):\n", df_test["y_blue_win"].value_counts(normalize=True))
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score (classe 1 = blue win): {f1:.4f}\n")
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório:\n", classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    run_catboost_test()