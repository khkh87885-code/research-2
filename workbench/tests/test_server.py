"""HTTP boundary tests; no patient images are used in these fixtures."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from research_platform.server import make_handler


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = {"output_root": self.temp.name}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a: None)
        self.port = self.server.server_port
        self.server.RequestHandlerClass = make_handler(self.config, self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_eight_page_shell_and_no_external_assets(self):
        html = urlopen(self.base).read().decode("utf-8")
        self.assertEqual(html.count('<section class="page'), 8)
        self.assertNotIn('src="https://', html)

    def test_output_traversal_rejected(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base + "/api/files?path=../secret.json")
        self.assertEqual(error.exception.code, 400)

    def test_cross_origin_mutation_rejected(self):
        req = Request(self.base + "/api/scan", data=b"{}", headers={"Origin": "https://untrusted.example", "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as error:
            urlopen(req)
        self.assertEqual(error.exception.code, 400)

    def test_real_empty_dataset_status(self):
        data = json.load(urlopen(self.base + "/api/datasets"))
        self.assertEqual(data["records"], [])
        self.assertEqual(data["summary"]["status"], "not_scanned")
