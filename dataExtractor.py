import os
import time
import json
import requests
from typing import List, Set, Optional

# ============================================================
# Configurações
# ============================================================

API_KEY = "" # Defina sua chave de API da Riot aqui. Você consegue uma chave gratuita em https://developer.riotgames.com/ (crie uma conta, vá em "API Keys")
HEADERS = {"X-Riot-Token": API_KEY}

# Plataforma (serviços regionais LoL "platform routing")
PLATFORM = "br1"  # BR

# Cluster regional (Match-V5 / Account-V1 "regional routing")
REGIONAL = "americas"

# Fila/queue
LEAGUE_QUEUE = "RANKED_SOLO_5x5"  # Desafiante Solo/Duo (league-v4)
MATCH_QUEUE_ID = 420             # Ranked Solo/Duo (match-v5 filter / info.queueId)

# Volume
MATCH_COUNT_PER_PLAYER = 20      # quantos matchIds puxar por PUUID (por execução)
SLEEP_BETWEEN_CALLS = 0.12       # ajuste fino para evitar rate limit

# Saída
OUT_DIR = "riot_dump"
BASE_DIR = os.path.join(OUT_DIR, "BR1", "challenger", f"ranked_solo_{MATCH_QUEUE_ID}")
MATCHES_DIR = os.path.join(BASE_DIR, "matches")
TIMELINES_DIR = os.path.join(BASE_DIR, "timelines")
os.makedirs(MATCHES_DIR, exist_ok=True)
os.makedirs(TIMELINES_DIR, exist_ok=True)

# Dedup persistente (um matchId por linha)
SEEN_PATH = os.path.join(OUT_DIR, "seen_matches.jsonl")

# ============================================================
# HTTP helper com retry
# ============================================================

def riot_get(url: str, headers: dict, max_retries: int = 6, timeout: int = 30) -> requests.Response:
    last = None
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, timeout=timeout)
        last = r

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
            time.sleep(wait)
            continue

        if r.status_code in (500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue

        return r

    return last  # type: ignore

# ============================================================
# Dedup persistente
# ============================================================

def load_seen_matches(path: str) -> Set[str]:
    seen: Set[str] = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                mid = line.strip()
                if mid:
                    seen.add(mid)
    return seen

def append_seen_match(path: str, match_id: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(match_id + "\n")

# ============================================================
# Riot endpoints
# ============================================================

def get_challenger_puuids(queue: str = LEAGUE_QUEUE) -> List[str]:
    """
    league-v4 challengerleagues/by-queue retorna entries[] com puuid (no seu caso BR veio assim).
    """
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}"
    r = riot_get(url, HEADERS)
    r.raise_for_status()
    data = r.json()
    entries = data.get("entries", []) or []
    puuids = [e["puuid"] for e in entries if isinstance(e, dict) and "puuid" in e]
    return puuids

def get_match_ids_by_puuid(puuid: str, count: int = MATCH_COUNT_PER_PLAYER, queue: Optional[int] = MATCH_QUEUE_ID) -> List[str]:
    """
    match-v5 matchlist. Usa filtro queue quando possível; fallback sem queue (filtra depois pelo details.queueId).
    """
    base = f"https://{REGIONAL}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

    if queue is not None:
        url = f"{base}?start=0&count={count}&queue={queue}"
        r = riot_get(url, HEADERS)
        if r.status_code == 200:
            ids = r.json()
            if isinstance(ids, list) and len(ids) > 0:
                return ids

    # fallback
    url = f"{base}?start=0&count={count}"
    r = riot_get(url, HEADERS)
    r.raise_for_status()
    ids = r.json()
    return ids if isinstance(ids, list) else []

def fetch_match_details(match_id: str) -> dict:
    url = f"https://{REGIONAL}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = riot_get(url, HEADERS)
    r.raise_for_status()
    return r.json()

def fetch_match_timeline(match_id: str) -> dict:
    url = f"https://{REGIONAL}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    r = riot_get(url, HEADERS)
    r.raise_for_status()
    return r.json()

def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ============================================================
# Pipeline principal
# ============================================================

def run_challenger_br_pipeline() -> None:
    if not API_KEY or API_KEY == "SUA_API_KEY_AQUI":
        raise RuntimeError("Defina RIOT_API_KEY no ambiente ou edite API_KEY no script.")

    # sanity check (key funcionando?)
    status_url = f"https://{PLATFORM}.api.riotgames.com/lol/status/v4/platform-data"
    r = riot_get(status_url, HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"Falha no teste de chave. Status {r.status_code}: {r.text[:200]}")

    puuids = get_challenger_puuids()
    print(f"Challenger BR entries (PUUIDs): {len(puuids)}")

    seen_matches = load_seen_matches(SEEN_PATH)
    print(f"Already seen matches: {len(seen_matches)}")

    new_saved = 0
    unique_seen_before = len(seen_matches)

    for i, puuid in enumerate(puuids, start=1):
        try:
            time.sleep(SLEEP_BETWEEN_CALLS)

            match_ids = get_match_ids_by_puuid(puuid, count=MATCH_COUNT_PER_PLAYER, queue=MATCH_QUEUE_ID)

            for mid in match_ids:
                # dedup global
                if mid in seen_matches:
                    continue

                # marca como visto imediatamente (evita repetir quando jogadores se enfrentam)
                seen_matches.add(mid)
                append_seen_match(SEEN_PATH, mid)

                match_path = os.path.join(MATCHES_DIR, f"{mid}.json")
                timeline_path = os.path.join(TIMELINES_DIR, f"{mid}.json")

                # dedup por arquivo (se rodou parcialmente antes)
                if os.path.exists(match_path) and os.path.exists(timeline_path):
                    continue

                time.sleep(SLEEP_BETWEEN_CALLS)
                details = fetch_match_details(mid)

                # garante queue 420 (solo/duo). Se a lista veio sem filtro, aqui corta.
                if details.get("info", {}).get("queueId") != MATCH_QUEUE_ID:
                    continue

                time.sleep(SLEEP_BETWEEN_CALLS)
                timeline = fetch_match_timeline(mid)

                save_json(details, match_path)
                save_json(timeline, timeline_path)
                new_saved += 1

        except requests.HTTPError as e:
            # mostra prefixo do puuid só pra debug sem expor inteiro
            print(f"[{i}/{len(puuids)}] HTTPError puuid={puuid[:10]}...: {e}")
        except Exception as e:
            print(f"[{i}/{len(puuids)}] Error puuid={puuid[:10]}...: {e}")

    print(f"New matches saved this run: {new_saved}")
    print(f"Total unique matches tracked: {len(seen_matches)} (added {len(seen_matches) - unique_seen_before})")
    print(f"Files: matches={MATCHES_DIR} timelines={TIMELINES_DIR}")

# ============================================================
# Execução
# ============================================================

if __name__ == "__main__":
    run_challenger_br_pipeline()