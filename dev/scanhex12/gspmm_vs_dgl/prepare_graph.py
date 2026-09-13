"""Turn a named graph into one edge-list cache both sides of the comparison read.

    python prepare_graph.py cora tolokers-2 ogbn-proteins --out-dir <cache>

Writes <cache>/<name>.npz holding `edge_index` ([2, E] int64, as stored by the
source -- no self-loops, no reverse edges) and `num_nodes`.  Both dgl_side.py
and bench.py load that, so the two sides cannot end up measuring subtly
different graphs, and neither has to depend on torch_geometric or the ogb
package at measurement time.

Sources, picked by name:
  * GraphLand (hm-categories, city-roads-M, web-fraud, ...): the edgelist.csv
    of the Zenodo archive, under --graphland-root, in the direction it stores
    (the repo's own loader symmetrizes optionally; nothing here does).
  * OGB (ogbn-*): the raw edge.csv.gz of the download, under --ogb-root, with
    reverse edges added for the datasets ogb marks add_inverse_edge.
  * Planetoid (cora, citeseer, pubmed): via DGL, which ships them.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path

import numpy as np

GRAPHLAND = {
    "hm-categories", "hm-prices", "pokec-regions", "web-topics", "web-fraud", "web-traffic",
    "tolokers-2", "city-reviews", "city-roads-M", "city-roads-L", "artnet-exp", "artnet-views",
    "avazu-ctr", "twitch-views",
}
PLANETOID = {"cora": "CoraGraphDataset", "citeseer": "CiteseerGraphDataset", "pubmed": "PubmedGraphDataset"}


def _read_csv_pairs(path: Path, skip_header: bool) -> np.ndarray:
    """[E, 2] int64 from a two-column csv, gzipped or not.

    pandas if it is around (tens of millions of rows otherwise take minutes),
    else a manual parse that still avoids building a Python list per field.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        import pandas as pd

        return pd.read_csv(path, header=0 if skip_header else None, dtype=np.int64).to_numpy()
    except ImportError:
        with opener(path, "rt") as fh:
            if skip_header:
                fh.readline()
            return np.loadtxt(fh, delimiter=",", dtype=np.int64)


def load_graphland(name: str, root: str) -> tuple[np.ndarray, int]:
    d = Path(root) / name
    edgelist = d / "edgelist.csv"
    if not edgelist.exists():
        raise SystemExit(f"{name}: no {edgelist}. Unpack the Zenodo archive for it under {root}.")
    edges = _read_csv_pairs(edgelist, skip_header=True)
    # The archives carry no node count; the features table is the authority on
    # it, since isolated nodes exist and would be lost by taking max(edge) + 1.
    features = d / "features.csv"
    if features.exists():
        with open(features) as fh:
            num_nodes = sum(1 for _ in fh) - 1
    else:
        num_nodes = int(edges.max()) + 1
    return edges.T.copy(), num_nodes


# Whether an OGB dataset stores each undirected edge once and expects the
# reverse to be added.  ogb's own master.csv is the authority and is consulted
# when the package is importable; this map covers the datasets used here so the
# preparation does not require it.
OGB_ADD_INVERSE = {"ogbn-arxiv": False, "ogbn-products": True, "ogbn-proteins": True}


def _ogb_adds_inverse(name: str) -> bool:
    try:
        import os.path as osp

        import ogb.nodeproppred
        import pandas as pd

        master = pd.read_csv(osp.join(osp.dirname(ogb.nodeproppred.__file__), "master.csv"), index_col=0)
        return str(master[name]["add_inverse_edge"]).strip().lower() == "true"
    except Exception:
        if name not in OGB_ADD_INVERSE:
            raise SystemExit(f"{name}: unknown whether it needs reverse edges, and ogb is not importable")
        return OGB_ADD_INVERSE[name]


def load_ogb(name: str, root: str) -> tuple[np.ndarray, int]:
    raw = Path(root) / name.replace("-", "_") / "raw"
    if not raw.is_dir():
        raise SystemExit(f"{name}: no {raw}. The OGB archive unpacks straight into this layout.")
    edges = _read_csv_pairs(raw / "edge.csv.gz", skip_header=False).T.copy()
    with gzip.open(raw / "num-node-list.csv.gz", "rt") as fh:
        num_nodes = int(fh.readline().strip())

    if _ogb_adds_inverse(name):
        # An undirected dataset stores each edge once; ogb's loader hands back
        # both directions, and the DGL side used to go through that loader, so
        # the cache has to agree or the two runs measure different graphs.
        edges = np.concatenate([edges, edges[::-1]], axis=1)
    return edges, num_nodes


def load_planetoid(name: str) -> tuple[np.ndarray, int]:
    import dgl  # only needed for these three, and only at preparation time

    ds = getattr(dgl.data, PLANETOID[name])(verbose=False)
    g = ds[0]
    src, dst = g.edges()
    return np.stack([src.numpy().astype(np.int64), dst.numpy().astype(np.int64)]), int(g.num_nodes())


def prepare(name: str, args) -> Path:
    out = Path(args.out_dir) / f"{name}.npz"
    if out.exists() and not args.force:
        blob = np.load(out)
        print(f"{name}: cached, N={int(blob['num_nodes']):,} E={blob['edge_index'].shape[1]:,}")
        return out

    if name in PLANETOID:
        edge_index, num_nodes = load_planetoid(name)
    elif name.startswith("ogbn-"):
        edge_index, num_nodes = load_ogb(name, args.ogb_root)
    elif name in GRAPHLAND:
        edge_index, num_nodes = load_graphland(name, args.graphland_root)
    else:
        raise SystemExit(f"{name}: unknown graph. Known: GraphLand, ogbn-*, {', '.join(PLANETOID)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edge_index=edge_index, num_nodes=np.int64(num_nodes))
    deg = np.bincount(edge_index[1], minlength=num_nodes)
    print(f"{name}: N={num_nodes:,} E={edge_index.shape[1]:,} "
          f"max_in_deg={deg.max():,} mean={deg.mean():.1f} -> {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+")
    p.add_argument("--out-dir", default=os.environ.get("GRAPH_CACHE", "data/graph_cache"))
    p.add_argument("--graphland-root", default=os.environ.get("GRAPHLAND_ROOT", "data/graphland"))
    p.add_argument("--ogb-root", default=os.environ.get("OGB_ROOT", "data/ogb"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for name in args.names:
        prepare(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
