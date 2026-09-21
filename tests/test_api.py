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

    def test_analysis_rejects_path_outside_input_root(self) -> None:
        response = self.client.post("/analyses", json={"source": "../../README.md"})
        self.assertEqual(response.status_code, 403)

    def test_analysis_rejects_client_model_path(self) -> None:
        response = self.client.post(
            "/analyses",
            json={"source": "missing.mp4", "model_path": "malicious.pt"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
