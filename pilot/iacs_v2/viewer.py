"""Read-only loopback report viewer. No experiment execution or source access."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from compression_study.datasets import inside, sha256
from .common import read_json, safe_run


def make_handler(report, port):
    report = Path(report).resolve(strict=True)
    manifest = read_json(report / "report_manifest.json")
    allowed = {row["name"]: row["sha256"] for row in manifest["files"]}
    allowed["report_manifest.json"] = sha256(report / "report_manifest.json")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.headers.get("Host") not in ("127.0.0.1:%d" % port, "localhost:%d" % port):
                self.send_error(403)
                return
            route = urlparse(self.path).path
            if route == "/":
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:8765/")
                self.end_headers()
                return
            name = "dashboard.html" if route == "/iacs-v2" else route[7:] if route.startswith("/files/") else None
            if name not in allowed:
                self.send_error(404)
                return
            target = (report / name).resolve()
            if target.parent != report or not target.is_file() or sha256(target) != allowed[name]:
                self.send_error(409)
                return
            import mimetypes
            content = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", default="publication_v2")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    root = safe_run(read_json(args.config))
    report = (root / "reports" / args.report).resolve(strict=True)
    if report.parent != root / "reports":
        raise ValueError("Viewer accepts only a direct immutable report child")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(report, args.port))
    print("Read-only IACS_v2 viewer: http://127.0.0.1:%d/iacs-v2" % args.port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
