"""Serve RESULTS_ROOT read-only over HTTP, only under /<token>/ (everything else is 404).

Runpod exposes the port as https://<pod-id>-8000.proxy.runpod.net/<token>/...; without the
token nothing -- not even a directory listing -- is reachable.

    python serve_results.py <root> <token> [port]
"""

import functools
import http.server
import sys
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, token: str, **kwargs):
        self.token = token
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        prefix = f"/{self.token}"
        if not (path == prefix or path.startswith(prefix + "/")):
            return "/nonexistent"
        return super().translate_path(path[len(prefix) :] or "/")

    def log_message(self, fmt, *args):  # keep the token out of the pod log
        pass


def main() -> None:
    root, token = sys.argv[1], sys.argv[2]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    Path(root).mkdir(parents=True, exist_ok=True)
    handler = functools.partial(Handler, directory=root, token=token)
    http.server.ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


if __name__ == "__main__":
    main()
