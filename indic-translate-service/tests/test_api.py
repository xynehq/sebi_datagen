"""The HTTP contract lighton-ocr-server's translate_client relies on, with a stand-in translator."""

import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # fastapi/httpx not installed locally
    TestClient = None

if TestClient is not None:
    import main


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, texts, src_lang, *, fix_numbers_on=True):
        self.calls.append((texts, src_lang, fix_numbers_on))
        return [f"EN({src_lang}) {t}" for t in texts]

    def health(self):
        return {"model": "fake", "device": "cpu"}


@unittest.skipIf(TestClient is None, "fastapi not installed")
class ApiTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeTranslator()
        main.STATE["translator"] = self.fake
        self.client = TestClient(main.app)  # no context manager: the real model is never loaded

    def tearDown(self):
        main.STATE.clear()

    def test_translate_keeps_order_and_count(self):
        r = self.client.post("/translate", json={"texts": ["एक", "दोन"], "src_lang": "mr", "fix_numbers": True})
        assert r.status_code == 200
        assert r.json() == {"translations": ["EN(mr) एक", "EN(mr) दोन"]}
        assert self.fake.calls == [(["एक", "दोन"], "mr", True)]

    def test_fix_numbers_defaults_to_true(self):
        self.client.post("/translate", json={"texts": ["x"], "src_lang": "gu"})
        assert self.fake.calls[0][2] is True

    def test_unknown_language_is_rejected(self):
        r = self.client.post("/translate", json={"texts": ["x"], "src_lang": "ta"})
        assert r.status_code == 400

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "mr" in r.json()["languages"]


if __name__ == "__main__":
    unittest.main()
