#!/usr/bin/env python3
"""
rating_using_nodes_final.py

- Targets 500 oracle positions total (use cache if >=500; otherwise sample PGN and compute).
- Oracle = full-strength Stockfish but capped at ORACLE_TIME_LIMIT (seconds) per position.
- Maia engines run with Threads=1 to avoid instability; other engines default to ENGINE_THREADS.
- Batch writes to GRANULAR_LOG_PATH, resume support, and per-engine live stats after each batch.
- Robust parsing of real_engines.csv (expects columns engine_name,path,rating,uci_options or mode/value).
"""

import os
import json
import random
import time
import csv
from pathlib import Path
from tqdm import tqdm

import chess
import chess.pgn
import chess.engine
import pandas as pd
import numpy as np

# -------------------- CONFIG --------------------
PGN_FILE = "chessgames_august2025.pgn"
ENGINE_INFO_FILE = "real_engines.csv"            # your CSV of engines
ORACLE_CACHE_FILE = "oracle_cache.json"          # cached oracle moves (fen -> uci)
GRANULAR_LOG_PATH = "granular_analysis_log.csv"  # output per-(fen,engine) results

TARGET_ORACLE_POSITIONS = 500
MIN_PLY_FOR_SAMPLING = 10        # skip early opening
SAMPLE_PER_GAME = 5              # candidate sample per game (when building cache)

ORACLE_ENGINE_PATH = r"C:\Users\desja\Documents\my_programming\chess_analysis\engines\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe"
ORACLE_TIME_LIMIT = 600          # seconds cap for Oracle analysis (per position)

ENGINE_THREADS = 2               # default threads for non-Maia engines
LOG_BUFFER_SIZE = 200            # flush to disk every N rows
BATCH_STATS_INTERVAL = 200       # after this many rows written, recompute and print engine stats

# -------------------------------------------------

def load_oracle_cache():
    if Path(ORACLE_CACHE_FILE).exists():
        with open(ORACLE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_oracle_cache(cache):
    with open(ORACLE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def sample_positions_from_pgn(pgn_file, player_name, per_game=SAMPLE_PER_GAME):
    """
    For each game that includes player_name, collect FENs where it's player's turn
    and ply >= MIN_PLY_FOR_SAMPLING. Then sample up to per_game positions per game.
    Returns list of FENs (may contain duplicates across games; dedupe later).
    """
    sampled = []
    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            if player_name not in (white, black):
                continue

            # collect candidate FENs for this player's moves after MIN_PLY_FOR_SAMPLING
            board = game.board()
            move_list = list(game.mainline_moves())
            fen_candidates = []
            ply = 0
            for m in move_list:
                board.push(m)
                ply += 1
                # count ply -> full moves roughly, we want moves >= MIN_PLY_FOR_SAMPLING
                if ply < MIN_PLY_FOR_SAMPLING * 2:
                    continue
                # Is it player's turn to move in this position? (we want player's move to be the move chosen)
                # After pushing the move, board.turn is the next side to move; we want positions where player is to move
                # So construct position *before* player's move by replaying up to idx-1 instead:
                # Instead, simpler: iterate indices and build board for each index below.
                pass

    # The above loop is awkward; implement safer method below (reopen)
    sampled = []
    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            if player_name not in (white, black):
                continue

            moves = list(game.mainline_moves())
            # Build per-index boards from scratch so we never reuse mutated board
            prefix_board = chess.Board()
            fen_list = []
            # iterate moves and capture FEN before each move (so it's the side to move)
            for i, move in enumerate(moves):
                # At this point prefix_board is the position before move i
                # move_number (ply) = i+1
                ply = i + 1
                if ply >= MIN_PLY_FOR_SAMPLING * 2:
                    # Which side will move from prefix_board?
                    side_to_move = prefix_board.turn
                    # We want positions where the player is to move
                    if (white == player_name and side_to_move == chess.WHITE) or (black == player_name and side_to_move == chess.BLACK):
                        fen_list.append(prefix_board.fen())
                # push the move for next iteration
                try:
                    prefix_board.push(move)
                except Exception:
                    # if invalid for some reason, skip this game
                    fen_list = []
                    break

            if not fen_list:
                continue

            picks = random.sample(fen_list, min(per_game, len(fen_list)))
            sampled.extend(picks)

    return sampled

def ensure_oracle_positions(pgn_file, player_name, target=TARGET_ORACLE_POSITIONS):
    """
    Ensure we have at least `target` oracle positions in cache.
    If cache already >= target, return it.
    Otherwise sample positions and compute missing oracle moves (and save cache incrementally).
    """
    oracle_cache = load_oracle_cache()
    if len(oracle_cache) >= target:
        print(f"[CACHE] Found {len(oracle_cache)} cached oracle positions (>= {target}). Using them.")
        return oracle_cache

    print(f"[CACHE] Only {len(oracle_cache)} cached positions. Sampling more positions from PGN to reach {target}...")

    # sample candidate positions from PGN
    sampled = sample_positions_from_pgn(pgn_file, player_name, per_game=SAMPLE_PER_GAME)
    # dedupe preserving order
    seen = set(oracle_cache.keys())
    new_candidates = [fen for fen in sampled if fen not in seen]

    if not new_candidates:
        print("[CACHE] No new candidate positions found in PGN (or all already cached).")
        return oracle_cache

    # limit to needed
    needed = max(0, target - len(oracle_cache))
    new_candidates = new_candidates[:needed]
    print(f"[ORACLE] Will compute {len(new_candidates)} new oracle moves (to reach {target}).")

    # generate oracle moves (full-strength with time cap)
    with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle:
        oracle.configure({"Threads": 1, "Hash": 1024})
        # compute and save incrementally to avoid rework if interrupted
        for idx, fen in enumerate(tqdm(new_candidates, desc="Generating oracle moves")):
            board = chess.Board(fen)
            try:
                info = oracle.analyse(board, chess.engine.Limit(time=ORACLE_TIME_LIMIT))
                # info["pv"] may be present; some engines return pv in analyse
                move = None
                if "pv" in info and info["pv"]:
                    move = info["pv"][0].uci()
                elif "bestmove" in info and info["bestmove"]:
                    move = info["bestmove"].uci() if hasattr(info["bestmove"], "uci") else None
                # fallback to engine.play with a short limit (shouldn't be needed)
                if not move:
                    try:
                        res = oracle.play(board, chess.engine.Limit(time=1.0))
                        move = res.move.uci() if res and res.move else None
                    except Exception:
                        move = None

                if move:
                    oracle_cache[fen] = move
                else:
                    print(f"[WARN] Oracle did not return move for fen: {fen}")
            except Exception as e:
                print(f"[ERROR] Oracle failed on FEN (skipping): {fen} -- {e}")

            # Save cache periodically
            if (idx + 1) % 10 == 0:
                save_oracle_cache(oracle_cache)

    save_oracle_cache(oracle_cache)
    print(f"[CACHE] Cache now contains {len(oracle_cache)} positions.")
    return oracle_cache

# ---------------- engine helpers ----------------
def parse_uci_options_str(s):
    """Try JSON parse, otherwise eval fallback, else empty dict."""
    if not s or (isinstance(s, float) and np.isnan(s)):
        return {}
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        try:
            return eval(s)
        except Exception:
            return {}

def open_and_configure_engine(path, engine_name, uci_options):
    """
    Open engine and apply UCI options (but not search-specific limits).
    We'll set Threads separately.
    """
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
    except Exception as e:
        print(f"[ERROR] Could not start engine {engine_name} at {path}: {e}")
        return None

    # Apply non-search options (skip depth/nodes/time/movetime)
    opts = parse_uci_options_str(uci_options)
    for k, v in opts.items():
        if k.lower() in ("depth", "nodes", "time", "movetime"):
            continue
        try:
            engine.configure({k: v})
        except Exception as e:
            # Some engines reject options; that's fine
            print(f"[WARN] Engine {engine_name} rejected option {k}={v}: {e}")
    return engine

def build_limit_from_options(opts):
    """
    Build chess.engine.Limit from parsed uci_options or explicit columns.
    opts: dict possibly containing "depth","nodes","movetime","time"
    Returns chess.engine.Limit instance.
    """
    if not opts:
        return None
    o = {}
    if "nodes" in opts:
        try:
            o["nodes"] = int(opts["nodes"])
        except Exception:
            pass
    if "depth" in opts:
        try:
            o["depth"] = int(opts["depth"])
        except Exception:
            pass
    if "movetime" in opts:
        try:
            # movetime in ms
            o["time"] = float(opts["movetime"]) / 1000.0
        except Exception:
            pass
    if "time" in opts:
        try:
            o["time"] = float(opts["time"])
        except Exception:
            pass

    # If none of these found, return None; caller will pick a safe default
    if not o:
        return None
    return chess.engine.Limit(**o)

# ---------------- main analysis ----------------
def analyze_with_engines(positions, oracle_cache, engine_csv, buffer_size=LOG_BUFFER_SIZE):
    """
    positions: list of FENs to test (length TARGET_ORACLE_POSITIONS)
    oracle_cache: dict fen->uci move
    engine_csv: path to engine CSV with at least engine_name,path and uci_options (or mode/value)
    """
    # Read engine info
    engines_df = pd.read_csv(engine_csv)
    # Prepare resume
    existing_seen = set()
    if Path(GRANULAR_LOG_PATH).exists():
        try:
            prev = pd.read_csv(GRANULAR_LOG_PATH)
            for _, r in prev.iterrows():
                existing_seen.add((r["fen"], r["engine"]))
            print(f"[RESUME] Found {len(prev)} existing results; will skip them.")
        except Exception:
            existing_seen = set()

    # We'll buffer writes
    buffer = []
    written_count = 0

    # We'll also track per-engine counts in memory for live stats (fast)
    engine_counts = {}  # engine -> [hits, total]

    # Helper to flush buffer
    def flush_buffer():
        nonlocal buffer, written_count
        if not buffer:
            return
        header = not Path(GRANULAR_LOG_PATH).exists()
        df = pd.DataFrame(buffer)
        df.to_csv(GRANULAR_LOG_PATH, mode="a", header=header, index=False)
        written_count += len(buffer)
        buffer = []

    # iterate engines one by one (open/close per engine to avoid leaking handles)
    for _, row in engines_df.iterrows():
        eng_name = str(row.get("engine_name", "UNKNOWN"))
        eng_path = str(row.get("path", ""))
        uci_options_raw = row.get("uci_options", "{}")
        # parse uci_options column
        uci_opts = parse_uci_options_str(uci_options_raw)

        # open engine and configure
        engine = open_and_configure_engine(eng_path, eng_name, uci_options_raw)
        if engine is None:
            print(f"[SKIP] Could not start engine {eng_name}; skipping.")
            continue

        # Threads handling: Maia single-thread, else ENGINE_THREADS
        try:
            if "maia" in eng_name.lower() or "leela" in eng_name.lower() or "lc0" in eng_name.lower():
                engine.configure({"Threads": 1})
                threads_for_engine = 1
            else:
                engine.configure({"Threads": ENGINE_THREADS})
                threads_for_engine = ENGINE_THREADS
        except Exception:
            pass

        # initialize counts
        engine_counts.setdefault(eng_name, [0, 0])

        # For each position
        for fen in tqdm(positions, desc=f"Analyzing with {eng_name}", leave=False):
            if (fen, eng_name) in existing_seen:
                continue

            board = chess.Board(fen)
            # Build limit from uci options (depth/nodes/time)
            limit = build_limit_from_options(uci_opts)
            if limit is None:
                # fallback default short limit so engines don't search forever
                limit = chess.engine.Limit(depth=5, time=0.05)

            # Run engine safely: try play() first (move), fall back to analyse if needed
            move_uci = None
            try:
                res = engine.play(board, limit)
                if res and res.move:
                    move_uci = res.move.uci()
            except Exception as e_play:
                # try analyse as fallback (some engines prefer analyse)
                try:
                    info = engine.analyse(board, limit)
                    if "pv" in info and info["pv"]:
                        move_uci = info["pv"][0].uci()
                except Exception as e_an:
                    # If engine fails repeatedly, log and skip this fen for this engine
                    print(f"[ERROR] Engine {eng_name} failed on FEN {fen}: {e_an}")
                    move_uci = None

            correct = 1 if move_uci == oracle_cache.get(fen) else 0
            buffer.append({"fen": fen, "engine": eng_name, "correct": correct})

            # update in-memory counts
            engine_counts[eng_name][1] += 1
            engine_counts[eng_name][0] += correct

            # periodic flush
            if len(buffer) >= buffer_size:
                flush_buffer()

            # periodic live stats printing
            if (written_count + len(buffer)) % BATCH_STATS_INTERVAL == 0:
                # flush then print
                flush_buffer()
                try:
                    log_df = pd.read_csv(GRANULAR_LOG_PATH)
                    stats = log_df.groupby("engine")["correct"].agg(["mean", "count", "sum"]).reset_index()
                    print("\n[STATS] Current engine accuracies (from file):")
                    for _, r in stats.iterrows():
                        print(f"  {r['engine']:25s} → {100*r['mean']:5.1f}% ({int(r['sum'])}/{int(r['count'])})")
                    print("-" * 60)
                except Exception as e:
                    print(f"[WARN] could not compute stats: {e}")

        # done with engine
        try:
            engine.quit()
        except Exception:
            pass

        # flush after each engine (to keep resume robust)
        flush_buffer()

    # final flush
    flush_buffer()
    print(f"[DONE] Analysis finished; wrote results to {GRANULAR_LOG_PATH}")

# -------------------- RUN --------------------
def main():
    random.seed(42)
    print("--- Starting Rating Estimation Script ---")

    # Step 1: sample/extract candidate positions and ensure oracle cache covers TARGET_ORACLE_POSITIONS
    print("[STEP] Ensuring oracle cache / positions...")
    oracle_cache = ensure_oracle_positions(PGN_FILE, player_name="Desjardins373", target=TARGET_ORACLE_POSITIONS)

    # Choose exactly TARGET_ORACLE_POSITIONS positions (if cache larger, sample deterministically)
    all_cached = list(oracle_cache.keys())
    if len(all_cached) < TARGET_ORACLE_POSITIONS:
        raise SystemError(f"Cache contains only {len(all_cached)} positions; expected {TARGET_ORACLE_POSITIONS}")
    # deterministic selection: sort and pick first TARGET positions to keep reproducibility
    all_cached_sorted = sorted(all_cached)
    positions = all_cached_sorted[:TARGET_ORACLE_POSITIONS]
    print(f"[STEP] Using {len(positions)} positions for engine testing (from cache).")

    # Step 2: run engines
    analyze_with_engines(positions, oracle_cache, ENGINE_INFO_FILE)

if __name__ == "__main__":
    main()
