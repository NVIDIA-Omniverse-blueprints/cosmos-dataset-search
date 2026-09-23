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

"""
Simple HTTP static file host with recursive indexing and CSV output.

Features:
- Serves files recursively from a specified root directory
- Configurable port
- Generates/overwrites hosted_files.csv at startup with URLs for each hosted file
- URL host selectable: 0.0.0.0 or IPv4 of a specified network interface

Usage examples:
  python scripts/http_file_host.py --root /path/to/files --port 8080 --url-host-mode local
  python scripts/http_file_host.py --root /path/to/files --port 8080 \
      --url-host-mode interface --interface eth0
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
from typing import Optional
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host a directory over HTTP and index files to CSV.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to serve recursively.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--url-host-mode",
        choices=["local", "interface"],
        default="local",
        help="How to construct file URLs in CSV: 'local' uses 0.0.0.0; 'interface' uses IPv4 of the specified interface.",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Network interface name (e.g., eth0, en0, Ethernet). Required if --url-host-mode=interface.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional path for hosted_files.csv. Defaults to <root>/hosted_files.csv.",
    )
    return parser.parse_args()


def get_interface_ipv4_address(interface_name: str) -> Optional[str]:
    """Attempt to get the IPv4 address for a given interface name.

    Tries psutil if available; falls back to system tools when possible.
    Returns None if not found.
    """
    # Try psutil if available
    try:
        import psutil  # type: ignore

        addrs = psutil.net_if_addrs().get(interface_name)
        if addrs:
            for addr in addrs:
                # 2 == AF_INET without importing socket here to avoid extra imports
                if getattr(addr, "family", None) == 2 and addr.address:
                    return addr.address
    except Exception:
        pass

    # Fallback: Linux 'ip' command
    try:
        proc = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "dev", interface_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            # output like: 2: eth0    inet 10.0.0.5/24 brd 10.0.0.255 scope global eth0
            for line in proc.stdout.splitlines():
                parts = line.split()
                if "inet" in parts:
                    cidr = parts[parts.index("inet") + 1]
                    return cidr.split("/")[0]
    except Exception:
        pass

    # Fallback: Windows 'netsh' (best-effort)
    if os.name == "nt":
        try:
            proc = subprocess.run(
                [
                    "netsh",
                    "interface",
                    "ipv4",
                    "show",
                    "addresses",
                    f"name={interface_name}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if line.lower().startswith("ip address") or line.lower().startswith("ipv4 address"):
                        # formats may vary; take the last token if dotted quad
                        tokens = [t for t in line.replace(":", " ").split() if t]
                        for token in reversed(tokens):
                            if token.count(".") == 3:
                                return token
        except Exception:
            pass

    return None


def compute_url_host(url_host_mode: str, interface_name: Optional[str]) -> str:
    if url_host_mode == "local":
        return "0.0.0.0"

    # interface mode
    if not interface_name:
        print("--interface is required when --url-host-mode=interface", file=sys.stderr)
        sys.exit(2)

    ip = get_interface_ipv4_address(interface_name)
    if not ip:
        print(
            f"Could not determine IPv4 address for interface '{interface_name}'. "
            "If available, install 'psutil' for more reliable detection.",
            file=sys.stderr,
        )
        sys.exit(2)
    return ip


def write_csv(root_dir: str, base_url: str, csv_path: str) -> int:
    """Write hosted_files.csv with relative_path and URL; returns count of files indexed."""
    count = 0
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "url"])  # header
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                abs_path = os.path.join(dirpath, filename)
                # Skip the CSV itself if it's under root
                if os.path.abspath(abs_path) == os.path.abspath(csv_path):
                    continue
                rel_path = os.path.relpath(abs_path, root_dir)
                # Ensure URL-safe and forward slashes
                rel_url_path = quote(rel_path.replace(os.sep, "/"))
                file_url = f"{base_url}/{rel_url_path}"
                writer.writerow([rel_path, file_url])
                count += 1
    return count


def serve(root_dir: str, port: int) -> None:
    handler_cls = partial(SimpleHTTPRequestHandler, directory=root_dir)
    with ThreadingTCPServer(("0.0.0.0", port), handler_cls) as httpd:
        sa = httpd.socket.getsockname()
        print(f"Serving HTTP on {sa[0]} port {sa[1]} (root='{root_dir}') ...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Root directory does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    csv_path = args.csv_path or os.path.join(root, "hosted_files.csv")
    url_host = compute_url_host(args.url_host_mode, args.interface)
    base_url = f"http://{url_host}:{args.port}"

    num_files = write_csv(root, base_url, csv_path)
    print(f"Wrote {num_files} file URL(s) to: {csv_path}")
    print(f"Base URL: {base_url}")

    serve(root, args.port)


if __name__ == "__main__":
    main()
