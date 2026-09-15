import unittest

from fastapi.testclient import TestClient

from src.api.app import app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_analysis_requires_existing_source(self) -> None:
        response = self.client.post("/analyses", json={"source": "missing.mp4"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
