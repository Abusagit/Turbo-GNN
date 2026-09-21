"""Edge count of a prepared graph, or 0 if it is not in the cache.

    python edge_count.py <cache dir> <graph name>

run.sh uses it to decide whether an [E, d] edge operand fits in memory.
"""

import os
import sys

import numpy as np

cache = os.path.join(sys.argv[1], f"{sys.argv[2]}.npz")
print(np.load(cache)["edge_index"].shape[1] if os.path.exists(cache) else 0)
