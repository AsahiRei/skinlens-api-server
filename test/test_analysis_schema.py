import unittest

from app.schemas.common import AnalyzeTaskStatus


class AnalyzeTaskStatusTests(unittest.TestCase):
    def test_allows_error_result(self):
        task = {
            "progress": 100,
            "status": "No face detected",
            "result": {"error": "No face detected in the image"},
            "heatmap": None,
        }

        status = AnalyzeTaskStatus(**task)

        self.assertIsNotNone(status.result)
        self.assertEqual(status.result.error, "No face detected in the image")


if __name__ == "__main__":
    unittest.main()
