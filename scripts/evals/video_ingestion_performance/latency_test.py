#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import math
import os
import random
import string
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue, current_process
from typing import Any, Dict, List, Optional, Tuple

import requests

# Pool of realistic video search queries (actions, subjects, scenes, events)
SEARCH_QUERY_POOL: List[str] = [
    "sunset over the ocean",
    "dog catching a frisbee in a park",
    "busy city street at night with neon lights",
    "car driving on a mountain road",
    "soccer player scoring a goal",
    "chef chopping onions in a kitchen",
    "time lapse of clouds over mountains",
    "cat jumping onto a couch",
    "concert crowd waving hands",
    "rain falling on a window",
    "drone footage of a forest",
    "child riding a bicycle",
    "airplane taking off at an airport",
    "close up of typing on a keyboard",
    "waves crashing on rocky shore",
    "snowboarder doing a trick",
    "runner sprinting on a track",
    "basketball dunk in slow motion",
    "fireworks over a city skyline",
    "construction workers on a site",
    "bee pollinating a flower macro",
    "wind turbines spinning in a field",
    "train passing through countryside",
    "cooking pasta in boiling water",
    "pouring coffee into a mug",
    "horse galloping on a beach",
    "storm clouds rolling in",
    "lightning strike in the distance",
    "people hiking a forest trail",
    "kayaker paddling on a lake",
    "skateboarder grinding a rail",
    "handwriting notes with a pen",
    "robot arm assembling parts",
    "teacher writing on a chalkboard",
    "child drawing with crayons",
    "closeup of eyes blinking",
    "bird flying over a field",
    "sunrise time lapse city skyline",
    "car drifting around a corner",
    "cyclist riding uphill",
    "pianist playing classical music",
    "guitarist strumming acoustic guitar",
    "waves lapping against a pier",
    "campfire at night with sparks",
    "butterfly landing on a leaf",
    "squirrel eating a nut",
    "people dancing at a wedding",
    "baby laughing and clapping",
    "runner tying shoelaces",
    "basket being woven by hand",
    "3d printer creating a model",
    "cars stuck in traffic",
    "bus arriving at bus stop",
    "subway doors closing",
    "snow falling on pine trees",
    "penguins waddling on ice",
    "dolphins jumping out of water",
    "surfer riding a big wave",
    "volcano erupting ash plume",
    "lava flowing at night",
    "meteor shower in the sky",
    "astronaut floating in space",
    "rocket launching from pad",
    "satellite orbiting earth animation",
    "drone flying over farmland",
    "farmers harvesting wheat",
    "combine harvester in field",
    "tractor plowing soil",
    "rainbow over waterfall",
    "people kayaking in rapids",
    "climber ascending a cliff",
    "paraglider soaring above valley",
    "ballet dancer pirouette",
    "boxing match slow motion punch",
    "tennis player serving ball",
    "baseball home run swing",
    "hockey players on the ice",
    "track relay baton pass",
    "swimmer diving into pool",
    "crowd cheering at stadium",
    "parade marching down street",
    "street food vendor cooking",
    "barista making latte art",
    "sushi chef slicing fish",
    "bread dough kneading",
    "cake being decorated",
    "chocolate melting in bowl",
    "ice cream scooped into cone",
    "leaves rustling in the wind",
    "river flowing through canyon",
    "desert sand dunes in wind",
    "camel caravan in desert",
    "northern lights over snowy landscape",
    "city skyline aerial at dusk",
    "bridge with cars at night",
    "fountain in town square",
    "museum gallery with paintings",
    "library shelves and readers",
    "students in a classroom",
    "scientist mixing chemicals",
    "microscope slide examination",
    "lab technician pipetting",
    "medical doctor checking heartbeat",
    "nurse giving vaccination",
    "ambulance with flashing lights",
    "firetruck driving to emergency",
    "police car patrolling",
    "helicopter hovering over city",
    "mountain bikers on trail",
    "fisherman casting a line",
    "sailing boat on a lake",
    "cargo ship entering harbor",
    "cranes loading containers",
    "forklift moving pallets",
    "warehouse workers packing boxes",
    "office workers in a meeting",
    "presentation with projector",
    "typing on a laptop",
    "drag and drop on computer screen",
    "zoom video conference call",
    "person scrolling on smartphone",
    "smartwatch showing heart rate",
    "drone inspecting wind turbine",
    "solar panels tracking sun",
    "rainwater dripping from roof",
    "snowplow clearing street",
    "lawn mower cutting grass",
    "hedge trimmer in garden",
    "plant being watered",
    "seedlings sprouting time lapse",
    "bees at a beehive entrance",
    "ant colony activity closeup",
    "snail moving slowly on leaf",
    "frog jumping into pond",
    "fish swimming in aquarium",
    "octopus changing color",
    "crab walking sideways on beach",
    "whale breaching in ocean",
    "seal sunbathing on rocks",
    "bear catching salmon in river",
    "deer running through forest",
    "fox trotting across field",
    "wolf howling at night",
    "owl flying silently",
    "eagle soaring over mountains",
    "parrot speaking on perch",
    "peacock displaying feathers",
    "zebra herd on savannah",
    "giraffe eating tree leaves",
    "elephants crossing river",
    "lion resting under tree",
    "gorilla beating chest",
    "monkeys swinging in trees",
    "koala sleeping on branch",
    "kangaroo hopping across plain",
    "panda eating bamboo",
    "cooking omelette in pan",
    "stir-fry vegetables in wok",
    "boiling soup in pot",
    "grilling steak on barbecue",
    "flipping pancakes on griddle",
    "baker taking bread from oven",
    "pouring wine into glass",
    "mixing cocktail with shaker",
    "cutting lemon on cutting board",
    "opening a soda can",
    "unboxing a new gadget",
    "assembling furniture from box",
    "painting a wall with roller",
    "drilling a hole in wood",
    "sanding a wooden surface",
    "hammering nails into plank",
    "sawing board with hand saw",
    "sewing fabric on machine",
    "ironing shirt on board",
    "folding laundry",
    "vacuuming living room",
    "mopping kitchen floor",
    "washing dishes in sink",
    "loading clothes into washer",
    "hanging clothes on a line",
    "making the bed neatly",
    "watering houseplants",
    "arranging flowers in vase",
    "wrapping a gift box",
    "lighting candles on cake",
    "decorating a Christmas tree",
    "carving a pumpkin",
    "drawing a portrait with pencil",
    "painting watercolor landscape",
    "calligraphy writing practice",
    "playing chess at a table",
    "shuffling a deck of cards",
    "dealing poker cards",
    "rolling dice on board",
    "vr headset gameplay demo",
    "driving simulator cockpit view",
    "flight simulator landing",
    "coding on a computer screen",
    "server rack blinking lights",
    "ethernet cables being plugged",
    "robot vacuum cleaning floor",
    "smart home voice assistant",
    "security camera footage hallway",
    "doorbell camera view porch",
    "3d animation spinning logo",
    "logo reveal with particles",
    "product spin on turntable",
    "hand holding product closeup",
    "makeup tutorial applying eyeliner",
    "haircut with scissors",
    "beard trim at barbershop",
    "nail polish being applied",
    "yoga sun salutation sequence",
    "push-up workout set",
    "jump rope in gym",
    "cycling class indoor",
    "treadmill running session",
    "elliptical machine workout",
    "rowing machine exercise",
    "medicine ball throws",
]

# Active query pool (can be limited by CLI)
ACTIVE_QUERY_POOL: List[str] = []


def rand_text(min_len: int = 3, max_len: int = 12) -> str:
    # Ignore lengths; choose realistic query from active pool if set, else default pool
    pool = ACTIVE_QUERY_POOL if ACTIVE_QUERY_POOL else SEARCH_QUERY_POOL
    return random.choice(pool)


def build_search_payload(
    query_text: str,
    top_k: int,
    reconstruct: bool,
    search_params: Dict[str, Any],
    filters: Dict[str, Any],
    generate_asset_url: bool,
) -> Dict[str, Any]:
    return {
        "query": {"text": query_text},
        "top_k": top_k,
        "reconstruct": reconstruct,
        "search_params": search_params or {},
        "filters": filters or {},
        "generate_asset_url": generate_asset_url,
        "clf": {},
    }


def do_search(
    base_url: str,
    collection_id: str,
    payload: Dict[str, Any],
    verbose: bool = False,
    timeout: int = 120,
) -> Tuple[float, Dict[str, Any]]:
    url = f"{base_url}/v1/collections/{collection_id}/search"
    headers = {"Content-Type": "application/json"}
    if verbose:
        print("REQUEST:")
        print(json.dumps({"url": url, "payload": payload}, indent=2))
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    t1 = time.perf_counter()
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        if verbose:
            print(f"HTTP {resp.status_code}: {resp.text}")
        raise
    result = resp.json()
    if verbose:
        print("RESPONSE:")
        print(json.dumps(result, indent=2))
    return (t1 - t0), result


def one_off(
    base_url: str,
    collection_id: str,
    query_text: Optional[str],
    random_query: bool,
    top_k: int,
    reconstruct: bool,
    search_params: Dict[str, Any],
    filters: Dict[str, Any],
    generate_asset_url: bool,
    verbose: bool,
):
    if random_query or not query_text:
        query_text = rand_text()
    payload = build_search_payload(
        query_text=query_text,
        top_k=top_k,
        reconstruct=reconstruct,
        search_params=search_params,
        filters=filters,
        generate_asset_url=generate_asset_url,
    )
    latency, _ = do_search(base_url, collection_id, payload, verbose=verbose)
    print(f"Latency: {latency:.3f}s")


def user_worker(
    base_url: str,
    collection_id: str,
    top_k: int,
    reconstruct: bool,
    search_params: Dict[str, Any],
    filters: Dict[str, Any],
    generate_asset_url: bool,
    pattern: str,
    end_time: float,
    q: Queue,
    query_pool: List[str],
):
    random.seed(os.getpid() + int(time.time()))
    count = 0
    # Choose from provided pool
    pool = query_pool if query_pool else SEARCH_QUERY_POOL
    while time.perf_counter() < end_time:
        query = random.choice(pool)
        payload = build_search_payload(
            query_text=query,
            top_k=top_k,
            reconstruct=reconstruct,
            search_params=search_params,
            filters=filters,
            generate_asset_url=generate_asset_url,
        )
        latency = float("nan")
        try:
            latency, _ = do_search(base_url, collection_id, payload, verbose=False)
            q.put({"latency": latency, "ok": 1, "query": query})
            count += 1
        except Exception:
            q.put({"latency": math.nan, "ok": 0, "query": query})

        if pattern == "stochastic":
            # Sleep up to 3N times a nominal search time; approximate using last latency
            # If no latency yet, small random jitter
            if count > 0 and not math.isnan(latency):
                max_sleep = 3.0 * latency
            else:
                max_sleep = random.uniform(0.01, 0.2)
            time.sleep(random.uniform(0.0, max_sleep))
        else:
            # continuous: immediately issue another request
            pass


def embed_latency(base_url: str, model: str, text: str, timeout: int = 120) -> float:
    url = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {
        "input": [text],
        "model": model,
        "request_type": "query",
        "encoding_format": "float",
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    t1 = time.perf_counter()
    resp.raise_for_status()
    _ = resp.json()
    return t1 - t0


def measure_simple_endpoints(
    base_url: str, collection_id: Optional[str], iterations: int = 10, timeout: int = 10
) -> Dict[str, List[float]]:
    endpoints: List[Tuple[str, str]] = [
        ("GET", f"{base_url}/health"),
        ("GET", f"{base_url}/ready"),
        ("GET", f"{base_url}/v1/collections"),
    ]
    if collection_id:
        endpoints.append(("GET", f"{base_url}/v1/collections/{collection_id}"))

    results: Dict[str, List[float]] = {}
    for method, url in endpoints:
        latencies: List[float] = []
        for _ in range(max(1, iterations)):
            t0 = time.perf_counter()
            try:
                if method == "GET":
                    r = requests.get(url, timeout=timeout)
                else:
                    r = requests.request(method, url, timeout=timeout)
                r.raise_for_status()
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
            except Exception:
                latencies.append(math.nan)
        results[url] = [x for x in latencies if not math.isnan(x)]
    return results


def latency_test(
    base_url: str,
    collection_id: str,
    duration_sec: int,
    users: int,
    pattern: str,
    csv_out: Optional[str],
    top_k: int,
    reconstruct: bool,
    search_params: Dict[str, Any],
    filters: Dict[str, Any],
    generate_asset_url: bool,
    embed_base_url: Optional[str] = None,
    embed_model: str = "nvidia/cosmos-embed1",
    simple_iters: int = 10,
    query_pool_size: int = 0,
):
    manager_q: Queue = Queue()
    # Fetch collection info (items) and measure simple endpoints on VS
    print(
        f"Starting latency test: users={users}, duration={duration_sec}s, pattern={pattern}, top_k={top_k}"
    )
    if query_pool_size and query_pool_size > 0:
        print(f"Using query pool size limit: {query_pool_size}")
    collection_items = None
    try:
        info_url = f"{base_url}/v1/collections/{collection_id}"
        r_info = requests.get(info_url, timeout=10)
        r_info.raise_for_status()
        info_json = r_info.json()
        collection_items = info_json.get("total_documents_count", None)
    except Exception:
        collection_items = None
    if collection_items is not None:
        print(f"Collection info fetched: items={collection_items}")
    else:
        print("Collection info fetch skipped or failed")

    # Build active pool according to requested size
    if query_pool_size and query_pool_size > 0:
        pool_size = min(query_pool_size, len(SEARCH_QUERY_POOL))
        active_pool = random.sample(SEARCH_QUERY_POOL, k=pool_size)
    else:
        active_pool = list(SEARCH_QUERY_POOL)
    print(f"Active query pool size: {len(active_pool)}")

    # Measure simple endpoints on VS
    print(f"Measuring VS baseline endpoints (simple_iters={simple_iters}) ...")
    simple_vs = measure_simple_endpoints(
        base_url, collection_id, iterations=simple_iters
    )
    baseline_candidates: List[float] = []
    for url, lats in simple_vs.items():
        if ("/health" in url) or ("/ready" in url):
            baseline_candidates.extend(lats)
    if not baseline_candidates:
        for lats in simple_vs.values():
            baseline_candidates.extend(lats)
    vs_baseline_avg = (
        (sum(baseline_candidates) / len(baseline_candidates))
        if baseline_candidates
        else float("nan")
    )
    vs_baseline_min = min(baseline_candidates) if baseline_candidates else float("nan")
    vs_baseline_samples = len(baseline_candidates)
    end_time = time.perf_counter() + duration_sec
    print("Starting main search loop ...")

    procs: List[Process] = []
    for _ in range(max(1, users)):
        p = Process(
            target=user_worker,
            args=(
                base_url,
                collection_id,
                top_k,
                reconstruct,
                search_params,
                filters,
                generate_asset_url,
                pattern,
                end_time,
                manager_q,
                active_pool,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    latencies: List[float] = []
    per_request: List[Tuple[str, float]] = []
    successes = 0
    start_wall = time.perf_counter()
    last_progress = 0.0
    while any(p.is_alive() for p in procs) or not manager_q.empty():
        try:
            item = manager_q.get(timeout=0.25)
            latency = item.get("latency", math.nan)
            ok = item.get("ok", 0)
            query = item.get("query", "")
            if ok and not math.isnan(latency):
                latencies.append(latency)
                per_request.append((query, latency))
                successes += 1
        except Exception:
            pass

        if time.perf_counter() > end_time and all(p.is_alive() for p in procs):
            for p in procs:
                p.join(timeout=0.1)

        # Progress output throttled
        now = time.perf_counter()
        if now - last_progress >= 0.5:
            last_progress = now
            elapsed = now - start_wall
            remaining = max(0.0, end_time - now)
            total = max(1e-9, duration_sec)
            pct = min(1.0, elapsed / total)
            bar_width = 30
            filled = int(bar_width * pct)
            bar = "#" * filled + "." * (bar_width - filled)
            sys.stdout.write(
                f"\rProgress [{bar}] {pct*100:5.1f}% | searches={successes} | elapsed={elapsed:5.1f}s | remaining={remaining:5.1f}s"
            )
            sys.stdout.flush()

    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

    # End progress line
    sys.stdout.write("\n")
    total_time = max(1e-9, time.perf_counter() - start_wall)

    # Stats
    if latencies:
        min_lat = min(latencies)
        max_lat = max(latencies)
        avg_lat = sum(latencies) / len(latencies)
    else:
        min_lat = max_lat = avg_lat = float("nan")

    sps = successes / total_time

    # Breakdown
    vs_travel_avg = vs_baseline_avg
    vs_travel_min = vs_baseline_min
    vs_search_only_avg = (
        (avg_lat - vs_travel_avg)
        if not math.isnan(avg_lat) and not math.isnan(vs_travel_avg)
        else float("nan")
    )
    vs_search_only_min = (
        (min_lat - vs_travel_min)
        if not math.isnan(min_lat) and not math.isnan(vs_travel_min)
        else float("nan")
    )

    # Optional Cosmos Embed service
    embed_latencies: List[float] = []
    embed_baseline_avg = float("nan")
    embed_baseline_min = float("nan")
    embed_baseline_samples = 0
    if embed_base_url:
        # Prefer Cosmos Embed service /v1/health/ready as baseline per guidance
        print(
            f"Measuring Cosmos Embed service baseline /v1/health/ready (simple_iters={simple_iters}) ..."
        )
        embed_ready_lats: List[float] = []
        for _ in range(max(1, simple_iters)):
            t0 = time.perf_counter()
            try:
                rr = requests.get(f"{embed_base_url}/v1/health/ready", timeout=5)
                rr.raise_for_status()
                t1 = time.perf_counter()
                embed_ready_lats.append(t1 - t0)
            except Exception:
                pass
        embed_baseline_samples = len(embed_ready_lats)
        if embed_ready_lats:
            embed_baseline_avg = sum(embed_ready_lats) / len(embed_ready_lats)
            embed_baseline_min = min(embed_ready_lats)
        print(
            f"Cosmos Embed service baseline collected: samples={embed_baseline_samples}, min/avg={embed_baseline_min if embed_ready_lats else float('nan'):.3f}/{embed_baseline_avg if embed_ready_lats else float('nan'):.3f}"
        )

        print(
            f"Running Cosmos Embed service embedding latency sampling for {simple_iters} iterations ..."
        )
        for _ in range(max(1, simple_iters)):
            text = random.choice(active_pool)
            try:
                lt = embed_latency(embed_base_url, embed_model, text)
                embed_latencies.append(lt)
            except Exception:
                pass

    # CSV
    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            report_writer = csv.writer(f)
            report_writer.writerow(["section", "metric", "value"])
            report_writer.writerow(["vs", "users", users])
            report_writer.writerow(["vs", "pattern", pattern])
            report_writer.writerow(["vs", "duration_sec", duration_sec])
            report_writer.writerow(["vs", "total_searches", successes])
            report_writer.writerow(["vs", "min_latency_sec", f"{min_lat:.6f}"])
            report_writer.writerow(["vs", "avg_latency_sec", f"{avg_lat:.6f}"])
            report_writer.writerow(["vs", "max_latency_sec", f"{max_lat:.6f}"])
            report_writer.writerow(["vs", "searches_per_sec", f"{sps:.6f}"])
            report_writer.writerow(["vs", "baseline_min_sec", f"{vs_baseline_min:.6f}"])
            report_writer.writerow(["vs", "baseline_avg_sec", f"{vs_baseline_avg:.6f}"])
            report_writer.writerow(
                ["vs", "search_only_min_sec", f"{vs_search_only_min:.6f}"]
            )
            report_writer.writerow(
                ["vs", "search_only_avg_sec", f"{vs_search_only_avg:.6f}"]
            )
            report_writer.writerow(["vs", "baseline_samples", vs_baseline_samples])
            report_writer.writerow(["vs", "collection_id", collection_id])
            report_writer.writerow(
                [
                    "vs",
                    "collection_items",
                    collection_items if collection_items is not None else "unknown",
                ]
            )

            for url, lats in simple_vs.items():
                if lats:
                    report_writer.writerow(
                        [
                            "vs_endpoint",
                            url,
                            f"min={min(lats):.6f};avg={(sum(lats)/len(lats)):.6f};max={max(lats):.6f}",
                        ]
                    )
                else:
                    report_writer.writerow(["vs_endpoint", url, "no_success"])

            if embed_base_url:
                if embed_latencies:
                    embed_min = min(embed_latencies)
                    embed_max = max(embed_latencies)
                    embed_avg = sum(embed_latencies) / len(embed_latencies)
                else:
                    embed_min = embed_max = embed_avg = float("nan")
                report_writer.writerow(
                    ["cosmos_embed", "min_latency_sec", f"{embed_min:.6f}"]
                )
                report_writer.writerow(
                    ["cosmos_embed", "avg_latency_sec", f"{embed_avg:.6f}"]
                )
                report_writer.writerow(
                    ["cosmos_embed", "max_latency_sec", f"{embed_max:.6f}"]
                )
                report_writer.writerow(
                    ["cosmos_embed", "baseline_min_sec", f"{embed_baseline_min:.6f}"]
                )
                report_writer.writerow(
                    ["cosmos_embed", "baseline_avg_sec", f"{embed_baseline_avg:.6f}"]
                )
                report_writer.writerow(
                    ["cosmos_embed", "baseline_samples", embed_baseline_samples]
                )
                report_writer.writerow(
                    ["cosmos_embed", "samples", len(embed_latencies)]
                )
                embed_avg = (
                    (embed_avg - embed_baseline_avg)
                    if not math.isnan(embed_avg) and not math.isnan(embed_baseline_avg)
                    else float("nan")
                )
                milvus_avg = (
                    (avg_lat - vs_baseline_avg - embed_avg)
                    if (
                        not math.isnan(avg_lat)
                        and not math.isnan(vs_baseline_avg)
                        and not math.isnan(embed_avg)
                    )
                    else float("nan")
                )
                report_writer.writerow(
                    ["breakdown", "travel_avg_sec", f"{vs_baseline_avg:.6f}"]
                )
                report_writer.writerow(
                    ["breakdown", "embed_avg_sec", f"{embed_avg:.6f}"]
                )
                report_writer.writerow(
                    [
                        "breakdown",
                        "milvus_search_avg_sec",
                        f"{max(0.0, milvus_avg):.6f}",
                    ]
                )

            # Histogram of queries
            query_to_lats: Dict[str, List[float]] = defaultdict(list)
            for qtext, l in per_request:
                query_to_lats[qtext].append(l)
            report_writer.writerow(
                ["histogram", "query", "stats"]
            )  # header for histogram section
            for qtext, lst in query_to_lats.items():
                cnt = len(lst)
                qmin = min(lst)
                qavg = sum(lst) / cnt
                qmax = max(lst)
                report_writer.writerow(
                    [
                        "histogram",
                        qtext,
                        f"count={cnt};min={qmin:.6f};avg={qavg:.6f};max={qmax:.6f}",
                    ]
                )

            report_writer.writerow([])
            detail_writer = csv.writer(f)
            detail_writer.writerow(["latency_sec", "query"])  # reversed order
            for qtext, l in per_request:
                detail_writer.writerow([f"{l:.6f}", qtext])

    print("Latency Test Report")
    print(f"  Users: {users}")
    print(f"  Pattern: {pattern}")
    print(f"  collection: {collection_id}")
    print(
        f"  collection items: {collection_items if 'collection_items' in locals() and collection_items is not None else 'unknown'}"
    )
    print(f"  Duration (s): {duration_sec}")
    print(f"  Total searches: {successes}")
    print(f"  Min/Avg/Max latency (s): {min_lat:.3f}/{avg_lat:.3f}/{max_lat:.3f}")
    print(f"  Searches per second: {sps:.3f}")
    print(f"  VS baseline min/avg (s): {vs_baseline_min:.3f}/{vs_baseline_avg:.3f}")
    print(
        f"  VS search-only min/avg (s): {vs_search_only_min:.3f}/{vs_search_only_avg:.3f}"
    )
    # Histogram stdout
    query_to_lats: Dict[str, List[float]] = defaultdict(list)
    for qtext, l in per_request:
        query_to_lats[qtext].append(l)
    print("Histogram (per query):")
    for qtext, lst in query_to_lats.items():
        cnt = len(lst)
        qmin = min(lst)
        qavg = sum(lst) / cnt
        qmax = max(lst)
        print(f"  {qtext}: count={cnt}, min/avg/max={qmin:.3f}/{qavg:.3f}/{qmax:.3f}")
    if embed_base_url:
        if embed_latencies:
            embed_min = min(embed_latencies)
            embed_max = max(embed_latencies)
            embed_avg = sum(embed_latencies) / len(embed_latencies)
        else:
            embed_min = embed_max = embed_avg = float("nan")
        print(
            f"Cosmos Embed service Latency Report ({embed_baseline_samples} iterations)"
        )
        print(
            f"  Min/Avg/Max latency (s): {embed_min:.3f}/{embed_avg:.3f}/{embed_max:.3f}"
        )
        print(
            f"  Cosmos Embed service baseline min/avg (s): {embed_baseline_min:.3f}/{embed_baseline_avg:.3f}"
        )
        embed_avg = (
            (embed_avg - embed_baseline_avg)
            if not math.isnan(embed_avg) and not math.isnan(embed_baseline_avg)
            else float("nan")
        )
        milvus_avg = (
            (avg_lat - vs_baseline_avg - embed_avg)
            if (
                not math.isnan(avg_lat)
                and not math.isnan(vs_baseline_avg)
                and not math.isnan(embed_avg)
            )
            else float("nan")
        )
        print("Breakdown (approx)")
        print(f"  travel (avg s): {vs_baseline_avg:.3f}")
        print(f"  cosmos-embed (avg s): {embed_avg:.3f}")
        print(f"  milvus search (avg s): {max(0.0, milvus_avg):.3f}")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Text search client for CVDS visual search service",
    )
    p.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    p.add_argument("--collection-id", required=True)

    # One-off mode
    p.add_argument(
        "--query",
        help="Text to search for; if omitted and --random-query, random text will be used",
    )
    p.add_argument(
        "--random-query", action="store_true", help="Use a randomly generated query"
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--reconstruct", action="store_true")
    p.add_argument(
        "--search-params",
        default="{}",
        help='JSON dict for search_params, e.g. {"nprobe": 32}',
    )
    p.add_argument("--filters", default="{}", help="JSON dict for filters")
    p.add_argument(
        "--no-asset-url", action="store_true", help="Disable asset URL generation"
    )
    p.add_argument("--verbose", action="store_true", help="Print request and response")

    # Latency test mode
    p.add_argument("--latency-test", action="store_true", help="Run latency test mode")
    p.add_argument(
        "--duration", type=int, default=30, help="Duration (seconds) for latency test"
    )
    p.add_argument(
        "--users", type=int, default=1, help="Number of simulated users (processes)"
    )
    p.add_argument(
        "--pattern", choices=["continuous", "stochastic"], default="continuous"
    )
    p.add_argument(
        "--csv-out", default=None, help="Path to write per-query latencies CSV"
    )
    p.add_argument(
        "--simple-iters",
        type=int,
        default=10,
        help="Iterations for simple endpoint latency sampling",
    )
    p.add_argument(
        "--cosmos-embed-base-url",
        dest="embed_base_url",
        default=None,
        help="Optional Cosmos Embed service base URL for direct latency tests",
    )
    p.add_argument(
        "--cosmos-embed-model",
        dest="embed_model",
        default="nvidia/cosmos-embed1",
        help="Model for Cosmos Embed service embeddings endpoint",
    )
    p.add_argument(
        "--query-pool-size",
        type=int,
        default=0,
        help="Limit size of search query dictionary (0 means full pool)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    try:
        search_params = json.loads(args.search_params) if args.search_params else {}
    except Exception as e:
        print(f"Invalid --search-params JSON: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        filters = json.loads(args.filters) if args.filters else {}
    except Exception as e:
        print(f"Invalid --filters JSON: {e}", file=sys.stderr)
        sys.exit(2)

    generate_asset_url = not args.no_asset_url

    if args.latency_test:
        latency_test(
            base_url=args.base_url,
            collection_id=args.collection_id,
            duration_sec=args.duration,
            users=args.users,
            pattern=args.pattern,
            csv_out=args.csv_out,
            top_k=args.top_k,
            reconstruct=args.reconstruct,
            search_params=search_params,
            filters=filters,
            generate_asset_url=generate_asset_url,
            embed_base_url=args.embed_base_url,
            embed_model=args.embed_model,
            simple_iters=args.simple_iters,
            query_pool_size=args.query_pool_size,
        )
    else:
        one_off(
            base_url=args.base_url,
            collection_id=args.collection_id,
            query_text=args.query,
            random_query=args.random_query,
            top_k=args.top_k,
            reconstruct=args.reconstruct,
            search_params=search_params,
            filters=filters,
            generate_asset_url=generate_asset_url,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
