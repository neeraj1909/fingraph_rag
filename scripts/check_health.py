#!/usr/bin/env python3
"""Check the host-facing readiness boundary for local dependencies."""

from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Callable
from urllib.error import URLError
from urllib.request import urlopen

ServiceCheck = Callable[[], None]


def check_tcp(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=2):
        return


def check_http(url: str) -> None:
    with urlopen(url, timeout=3) as response:  # noqa: S310 - fixed local URLs only
        if response.status >= 500:
            raise RuntimeError(f"HTTP {response.status}")


def checks() -> dict[str, ServiceCheck]:
    return {
        "postgres": lambda: check_tcp("localhost", 5432),
        "redis": lambda: check_tcp("localhost", 6379),
        "neo4j": lambda: check_http("http://localhost:7474"),
        "qdrant": lambda: check_http("http://localhost:6333/readyz"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required", nargs="+", choices=sorted(checks()), required=True)
    args = parser.parse_args()

    failed = False
    available = checks()
    for name in args.required:
        try:
            available[name]()
        except (OSError, URLError, RuntimeError) as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
        else:
            print(f"OK   {name}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
