import os
import json
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd

# ============================================================
# Config (mesma estrutura do dataExtractor.py)
# ============================================================

OUT_DIR = "riot_dump"
MATCH_QUEUE_ID = 420  # Solo/Duo ranked

BASE_DIR = os.path.join(OUT_DIR, "BR1", "challenger", f"ranked_solo_{MATCH_QUEUE_ID}")
MATCHES_DIR = os.path.join(BASE_DIR, "matches")
TIMELINES_DIR = os.path.join(BASE_DIR, "timelines")

# 1 CSV por partida:
CSV_OUT_DIR = os.path.join(BASE_DIR, "dataset_ts_csv")
os.makedirs(CSV_OUT_DIR, exist_ok=True)

# Log opcional (persistente) de partidas processadas:
PROCESSED_PATH = os.path.join(BASE_DIR, "processed_matches.jsonl")


# ============================================================
# Utils
# ============================================================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_write_jsonl_line(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_processed(path: str) -> set[str]:
    processed = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                mid = line.strip()
                if mid:
                    processed.add(mid)
    return processed

def extract_patch(game_version: str) -> str:
    if not game_version:
        return ""
    parts = str(game_version).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(game_version)

def ms_to_datetime_utc(ms: int) -> str:
    if ms is None or ms == "":
        return ""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.isoformat()


# ============================================================
# Team / participant mapping (timeline)
# ============================================================

BLUE_PIDS = set(range(1, 6))   # participantId 1..5
RED_PIDS  = set(range(6, 11))  # participantId 6..10

def pid_to_team(pid: int) -> int:
    return 100 if pid in BLUE_PIDS else 200

def team_sign(team_id: int) -> int:
    return 1 if team_id == 100 else -1


# ============================================================
# Timeline events -> counters
# ============================================================

def update_counters_from_event(ev: dict, counters: dict, first_flags: dict):
    et = ev.get("type", "")

    if et == "CHAMPION_KILL":
        killer = ev.get("killerId", 0)
        victim = ev.get("victimId", 0)
        assists = ev.get("assistingParticipantIds", []) or []

        if isinstance(killer, int) and 1 <= killer <= 10:
            t_k = pid_to_team(killer)
            counters["team"][t_k]["kills"] += 1
            counters["player"][killer]["kills"] += 1
            if first_flags["firstKill"] == 0:
                first_flags["firstKill"] = team_sign(t_k)

        if isinstance(victim, int) and 1 <= victim <= 10:
            t_v = pid_to_team(victim)
            counters["team"][t_v]["deaths"] += 1
            counters["player"][victim]["deaths"] += 1

        for a in assists:
            if isinstance(a, int) and 1 <= a <= 10:
                t_a = pid_to_team(a)
                counters["team"][t_a]["assists"] += 1
                counters["player"][a]["assists"] += 1
        return

    if et == "WARD_PLACED":
        creator = ev.get("creatorId", 0)
        if isinstance(creator, int) and 1 <= creator <= 10:
            t = pid_to_team(creator)
            counters["team"][t]["wardsPlaced"] += 1
            counters["player"][creator]["wardsPlaced"] += 1
        return

    if et == "WARD_KILL":
        killer = ev.get("killerId", 0)
        if isinstance(killer, int) and 1 <= killer <= 10:
            t = pid_to_team(killer)
            counters["team"][t]["wardsKilled"] += 1
            counters["player"][killer]["wardsKilled"] += 1
        return

    if et == "BUILDING_KILL":
        killer = ev.get("killerId", 0)
        building_type = ev.get("buildingType", "")

        if isinstance(killer, int) and 1 <= killer <= 10:
            t = pid_to_team(killer)
            if building_type == "TOWER_BUILDING":
                counters["team"][t]["towers"] += 1
                counters["player"][killer]["towers"] += 1
                if first_flags["firstTower"] == 0:
                    first_flags["firstTower"] = team_sign(t)
            elif building_type == "INHIBITOR_BUILDING":
                counters["team"][t]["inhibs"] += 1
        return

    if et == "ELITE_MONSTER_KILL":
        killer = ev.get("killerId", 0)
        monster_type = ev.get("monsterType", "")
        monster_sub = ev.get("monsterSubType", "")

        if isinstance(killer, int) and 1 <= killer <= 10:
            t = pid_to_team(killer)

            if monster_type == "DRAGON":
                if monster_sub == "ELDER_DRAGON":
                    counters["team"][t]["elders"] += 1
                    if first_flags["firstElderDragon"] == 0:
                        first_flags["firstElderDragon"] = team_sign(t)
                else:
                    counters["team"][t]["dragons"] += 1
                    if first_flags["firstDragon"] == 0:
                        first_flags["firstDragon"] = team_sign(t)

            elif monster_type == "BARON_NASHOR":
                counters["team"][t]["barons"] += 1
                if first_flags["firstBaron"] == 0:
                    first_flags["firstBaron"] = team_sign(t)

            elif monster_type == "RIFTHERALD":
                counters["team"][t]["heralds"] += 1

            elif monster_type == "HORDE" and monster_sub in {"VOIDGRUBS", "GRUBS", "VOID_GRUBS"}:
                counters["team"][t]["grubs"] += 1
                if first_flags["firstGrub"] == 0:
                    first_flags["firstGrub"] = team_sign(t)
        return


# ============================================================
# Match details -> meta + player static maps
# ============================================================

def extract_meta_from_match_details(match_details: dict) -> dict:
    info = match_details.get("info", {}) or {}

    game_creation_ms = info.get("gameCreation")
    game_duration_s = info.get("gameDuration")
    patch = extract_patch(info.get("gameVersion", ""))

    y_blue_win = None
    for t in info.get("teams", []) or []:
        if t.get("teamId") == 100:
            y_blue_win = 1 if t.get("win") else 0
            break

    return {
        "patch": patch,
        "game_datetime_utc": ms_to_datetime_utc(game_creation_ms),
        "gameCreation_ms": game_creation_ms if game_creation_ms is not None else "",
        "gameDuration_s": game_duration_s if game_duration_s is not None else "",
        "y_blue_win": y_blue_win if y_blue_win is not None else "",
    }

def extract_player_static_by_pid(match_details: dict) -> dict:
    info = match_details.get("info", {}) or {}
    parts = info.get("participants", []) or []

    by_pid = {}
    for p in parts:
        pid = p.get("participantId", None)
        if isinstance(pid, int):
            by_pid[pid] = {
                "championName": p.get("championName", ""),
                "individualPosition": p.get("individualPosition", ""),
                # "puuid": p.get("puuid", ""),  # não usar no modelo, mas útil pra auditoria
                "teamId": p.get("teamId", None),
            }
    return by_pid


# ============================================================
# Build TS dataset (sem *_diff)
# ============================================================

def build_ts_dataset_from_timeline(timeline_json: dict, meta: dict, player_static_by_pid: dict) -> pd.DataFrame:
    match_id = timeline_json["metadata"]["matchId"]
    frames = timeline_json["info"]["frames"]

    counters = {
        "team": {100: defaultdict(int), 200: defaultdict(int)},
        "player": {pid: defaultdict(int) for pid in range(1, 11)},
    }

    first_flags = {
        "firstKill": 0,
        "firstDragon": 0,
        "firstGrub": 0,
        "firstTower": 0,
        "firstBaron": 0,
        "firstElderDragon": 0,
    }

    rows = []
    for frame in frames:
        t_ms = frame.get("timestamp", 0)
        t_min = int(round(t_ms / 60000))

        for ev in frame.get("events", []) or []:
            update_counters_from_event(ev, counters, first_flags)

        pframes = frame.get("participantFrames", {}) or {}
        team_snapshot = {100: defaultdict(float), 200: defaultdict(float)}
        player_feats = {}

        for pid_str, st in pframes.items():
            pid = int(pid_str)
            team = pid_to_team(pid)

            current_gold = st.get("currentGold", 0)
            total_gold = st.get("totalGold", 0)
            xp = st.get("xp", 0)
            level = st.get("level", 0)
            cs = st.get("minionsKilled", 0)
            jcs = st.get("jungleMinionsKilled", 0)

            dmg = st.get("damageStats", {}) or {}
            dmg_done = dmg.get("totalDamageDone", 0)
            dmg_taken = dmg.get("totalDamageTaken", 0)
            dmg_to_champs = dmg.get("totalDamageDoneToChampions", 0)

            pos = st.get("position", {}) or {}
            posx = pos.get("x", None)
            posy = pos.get("y", None)

            # macro por time (valores absolutos)
            team_snapshot[team]["totalGold"] += total_gold
            team_snapshot[team]["currentGold"] += current_gold
            team_snapshot[team]["xp"] += xp
            team_snapshot[team]["cs"] += cs
            team_snapshot[team]["jcs"] += jcs
            team_snapshot[team]["dmg_done"] += dmg_done
            team_snapshot[team]["dmg_taken"] += dmg_taken
            team_snapshot[team]["dmg_to_champs"] += dmg_to_champs
            team_snapshot[team]["levels_sum"] += level

            prefix = f"p{pid}"
            static = player_static_by_pid.get(pid, {})

            # estáticas
            player_feats[f"{prefix}_championName"] = static.get("championName", "")
            player_feats[f"{prefix}_individualPosition"] = static.get("individualPosition", "")

            # numéricas por frame
            player_feats[f"{prefix}_currentGold"] = current_gold
            player_feats[f"{prefix}_totalGold"] = total_gold
            player_feats[f"{prefix}_xp"] = xp
            player_feats[f"{prefix}_level"] = level
            player_feats[f"{prefix}_cs"] = cs
            player_feats[f"{prefix}_jungleCS"] = jcs
            player_feats[f"{prefix}_totalDamageDone"] = dmg_done
            player_feats[f"{prefix}_totalDamageTaken"] = dmg_taken
            player_feats[f"{prefix}_totalDamageToChamps"] = dmg_to_champs
            player_feats[f"{prefix}_posX"] = posx
            player_feats[f"{prefix}_posY"] = posy

            # cumulativos
            player_feats[f"{prefix}_kills"] = counters["player"][pid]["kills"]
            player_feats[f"{prefix}_deaths"] = counters["player"][pid]["deaths"]
            player_feats[f"{prefix}_assists"] = counters["player"][pid]["assists"]
            player_feats[f"{prefix}_wardsPlaced"] = counters["player"][pid]["wardsPlaced"]
            player_feats[f"{prefix}_wardsDestroyed"] = counters["player"][pid]["wardsKilled"]
            player_feats[f"{prefix}_towersDestroyed"] = counters["player"][pid]["towers"]

        # macro ABSOLUTO por time (sem diff)
        blue, red = 100, 200
        team_feats = {}

        for k in ["totalGold", "currentGold", "xp", "cs", "jcs", "dmg_done", "dmg_taken", "dmg_to_champs", "levels_sum"]:
            team_feats[f"{k}_Blue"] = float(team_snapshot[blue][k])
            team_feats[f"{k}_Red"] = float(team_snapshot[red][k])

        for k in ["kills", "deaths", "assists", "wardsPlaced", "wardsKilled",
                  "towers", "inhibs", "dragons", "elders", "barons", "heralds", "grubs"]:
            team_feats[f"{k}_Blue"] = int(counters["team"][blue][k])
            team_feats[f"{k}_Red"] = int(counters["team"][red][k])

        for fk, val in first_flags.items():
            team_feats[fk] = int(val)

        row = {
            "matchId": match_id,
            "patch": meta.get("patch", ""),
            "y_blue_win": meta.get("y_blue_win", ""),
            "t_min": t_min,
            **team_feats,
            **player_feats,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["matchId", "t_min"]).reset_index(drop=True)


# ============================================================
# Batch processing: filtra JSON + dedup por matchId + 1 CSV por partida
# ============================================================

def process_all_matches() -> None:
    if not os.path.isdir(MATCHES_DIR) or not os.path.isdir(TIMELINES_DIR):
        raise RuntimeError(
            "Pastas de entrada não encontradas. Verifique se o dataExtractor.py já gerou:\n"
            f"  {MATCHES_DIR}\n"
            f"  {TIMELINES_DIR}"
        )

    processed = load_processed(PROCESSED_PATH)

    timeline_files = [f for f in os.listdir(TIMELINES_DIR) if f.endswith(".json")]
    timeline_files.sort()

    total = len(timeline_files)
    saved = 0
    skipped = 0

    for idx, fname in enumerate(timeline_files, start=1):
        timeline_path = os.path.join(TIMELINES_DIR, fname)

        # matchId (nome do arquivo)
        match_id = os.path.splitext(fname)[0]

        # evita repetir: (1) log (2) csv existente
        csv_path = os.path.join(CSV_OUT_DIR, f"{match_id}.csv")
        if match_id in processed or os.path.exists(csv_path):
            skipped += 1
            continue

        details_path = os.path.join(MATCHES_DIR, f"{match_id}.json")
        if not os.path.exists(details_path):
            # sem match details -> não conseguimos meta / jogadores
            skipped += 1
            continue

        try:
            details = load_json(details_path)

            # filtro nos JSON: só Solo/Duo ranked (420)
            if details.get("info", {}).get("queueId") != MATCH_QUEUE_ID:
                skipped += 1
                continue

            timeline = load_json(timeline_path)

            meta = extract_meta_from_match_details(details)
            player_static_by_pid = extract_player_static_by_pid(details)

            df = build_ts_dataset_from_timeline(
                timeline_json=timeline,
                meta=meta,
                player_static_by_pid=player_static_by_pid,
            )

            # 1 CSV por partida
            df.to_csv(csv_path, index=False)

            # marca como processado somente após sucesso
            processed.add(match_id)
            safe_write_jsonl_line(PROCESSED_PATH, match_id)

            saved += 1
            if saved % 25 == 0:
                print(f"[{idx}/{total}] saved={saved} skipped={skipped} last={match_id}")

        except Exception as e:
            # não marcar como processado se falhou
            print(f"[{idx}/{total}] ERROR matchId={match_id}: {e}")

    print("---- DONE ----")
    print(f"timelines encontrados: {total}")
    print(f"csv gerados: {saved}")
    print(f"pulados: {skipped}")
    print(f"saida: {CSV_OUT_DIR}")
    print(f"log: {PROCESSED_PATH}")


if __name__ == "__main__":
    process_all_matches()