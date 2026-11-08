import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
from compression_study.datasets import sha256
from iacs_v2.viewer import make_handler


@pytest.fixture
def server(tmp_path):
    (tmp_path / "dashboard.html").write_text("<h1>Synthetic report fixture</h1>")
    manifest = dict(files=[dict(name="dashboard.html", sha256=sha256(tmp_path / "dashboard.html"))])
    (tmp_path / "report_manifest.json").write_text(json.dumps(manifest))
    instance = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    instance.RequestHandlerClass = make_handler(tmp_path, instance.server_port)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % instance.server_port, tmp_path
    instance.shutdown()
    instance.server_close()
    thread.join()


def test_viewer_only_serves_manifest_and_detects_changes(server):
    url, root = server
    assert b"Synthetic report" in urlopen(url + "/iacs-v2").read()
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/files/unknown.txt")
    assert error.value.code == 404
    (root / "dashboard.html").write_text("altered")
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/iacs-v2")
    assert error.value.code == 409


def test_viewer_rejects_foreign_host_and_has_no_write_endpoint(server):
    url, _ = server
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url + "/iacs-v2", headers={"Host": "untrusted.example"}))
    assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url + "/iacs-v2", method="POST", data=b"no write"))
    assert error.value.code == 501
