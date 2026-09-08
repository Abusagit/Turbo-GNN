#!/usr/bin/env python3
"""Fetch benchmark graphs and lay them out where the loaders look for them.

Datasets normally download themselves on first use.  On a machine whose egress is filtered
that fails -- zenodo.org and snap.stanford.edu are commonly unreachable where github.com is
fine -- and the failure surfaces as an SSL error buried under a DGL fallback.  This script
downloads and unpacks the same archives, so it can be run somewhere with network and the
resulting ``data/`` tree copied across.

    # on a machine with network
    python3 scripts/download_datasets.py --datasets all --out data --archive graphs.tar.gz

    # then, on the machine with the GPU
    tar xzf graphs.tar.gz -C /path/to/Turbo-GNN/

Standard library only, and it never imports torch, so it runs on a laptop with nothing
installed.  Existing datasets are left alone unless --force is given.

Layout, which is what makes the copied tree work:

* GraphLand -- ``data/<name>/raw/<name>/`` holding info.yaml, features.csv and the rest;
* OGB -- ``data/<underscored name>/`` holding raw/, split/, mapping/ (the archive's top-level
  directory is the short name, e.g. ``products``, and gets renamed);
* Planetoid (cora, citeseer, pubmed) comes from github and usually needs no help, so it is
  not handled here.
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

GRAPHLAND_URL = "https://zenodo.org/records/16895532/files/{name}.zip"
OGB_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/{short}.zip"

# The graphs the kernel benchmark report covers, minus the Planetoid ones. Approximate sizes
# are the download, so a transfer can be planned before starting it.
GRAPHLAND = {
    "artnet-exp": 60,
    "avazu-ctr": 350,
    "city-reviews": 130,
    "city-roads-L": 110,
    "city-roads-M": 40,
    "hm-categories": 260,
    "pokec-regions": 900,
    "tolokers-2": 25,
    "twitch-views": 400,
    "web-fraud": 700,
    "web-traffic": 200,
}
# Short name inside the archive, and the download size. The short name is OGB's own
# `download_name` from ogb/nodeproppred/master.csv.
OGB = {"ogbn-arxiv": ("arxiv", 90), "ogbn-products": ("products", 1500), "ogbn-proteins": ("proteins", 1500)}


def target_dir(name: str, out: Path) -> Path:
    """Where a finished dataset lives, and therefore what marks it as already present."""
    return out / "_".join(name.split("-")) if name in OGB else out / name


def download(url: str, destination: Path, insecure: bool) -> None:
    context = ssl._create_unverified_context() if insecure else None
    print(f"    {url}")
    with urllib.request.urlopen(url, context=context, timeout=120) as response, destination.open("wb") as sink:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            sink.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print("\r" + " " * 40 + "\r", end="")


def fetch_graphland(name: str, out: Path, insecure: bool) -> None:
    raw = out / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / f"{name}.zip"
        download(GRAPHLAND_URL.format(name=name), archive, insecure)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(raw)


def fetch_ogb(name: str, out: Path, insecure: bool) -> None:
    short = OGB[name][0]
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / f"{short}.zip"
        download(OGB_URL.format(short=short), archive, insecure)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(scratch)
        # The archive unpacks under its short name; OGB expects the underscored dataset name.
        extracted = Path(scratch) / short
        if not extracted.is_dir():
            candidates = [entry for entry in Path(scratch).iterdir() if entry.is_dir()]
            if len(candidates) != 1:
                raise RuntimeError(f"{name}: expected one directory in the archive, found {candidates}")
            extracted = candidates[0]
        destination = target_dir(name, out)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted), str(destination))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset names, or 'all' for every graph this script knows",
    )
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Re-download datasets that are already present")
    parser.add_argument("--archive", type=Path, help="Also write a tar.gz of --out, ready to copy to another machine")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip certificate verification. For a TLS-intercepting proxy, not for an unreachable host",
    )
    parser.add_argument("--list", action="store_true", help="Print the known datasets and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    known: dict[str, tuple[str, int]] = {name: ("graphland", size) for name, size in GRAPHLAND.items()}
    known |= {name: ("ogb", size) for name, (_, size) in OGB.items()}

    if args.list:
        for name, (kind, size) in sorted(known.items()):
            print(f"  {name:<16} {kind:<10} ~{size} MB")
        return 0

    requested = sorted(known) if args.datasets == ["all"] else args.datasets
    unknown = [name for name in requested if name not in known]
    if unknown:
        print(f"unknown: {unknown}. Run --list to see the names this script knows.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in requested:
        kind, size = known[name]
        destination = target_dir(name, args.out)
        if destination.exists() and any(destination.iterdir()) and not args.force:
            print(f"{name}: already at {destination}")
            continue
        print(f"{name}: downloading ~{size} MB")
        try:
            (fetch_ogb if kind == "ogb" else fetch_graphland)(name, args.out, args.insecure)
            print(f"{name}: ready at {destination}")
        except Exception as error:  # noqa: BLE001 - one unreachable host must not end the run
            print(f"{name}: FAILED -- {type(error).__name__}: {str(error)[:120]}", file=sys.stderr)
            failed.append(name)

    if args.archive:
        print(f"\npacking {args.out} into {args.archive}")
        with tarfile.open(args.archive, "w:gz") as tar:
            tar.add(args.out, arcname=args.out.name)
        print(f"copy it over and unpack with: tar xzf {args.archive.name} -C /path/to/Turbo-GNN/")

    if failed:
        print(f"\nfailed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
