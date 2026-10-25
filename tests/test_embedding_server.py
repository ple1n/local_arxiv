import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import embedding_server


class ProgressFieldTests(unittest.TestCase):
    def test_integer_progress_is_preserved(self):
        self.assertEqual(
            embedding_server._progress_fields(
                SimpleNamespace(stage="Encoding", completed=4, total=9)
            ),
            ("Encoding", 4, 9),
        )

    def test_fractional_progress_is_normalized(self):
        self.assertEqual(
            embedding_server._progress_fields(
                SimpleNamespace(stage="Training", progress=0.25)
            ),
            ("Training", 250, 1000),
        )


class IndexJobRequestTests(unittest.TestCase):
    def setUp(self):
        embedding_server.index_jobs.clear()
        self.handler = object.__new__(embedding_server.EmbeddingHandler)
        self.handler._send_json = Mock()

    def test_invalid_request_is_rejected(self):
        self.handler._start_index_job({"job_id": "bad/id"})

        self.handler._send_json.assert_called_once()
        status, payload = self.handler._send_json.call_args.args
        self.assertEqual(status, embedding_server.HTTPStatus.BAD_REQUEST)
        self.assertIn("error", payload)

    def test_valid_request_is_queued(self):
        with patch.object(embedding_server, "index_executor") as executor:
            self.handler._start_index_job(
                {
                    "job_id": "job-1",
                    "database_path": "/tmp/lance",
                    "table_name": "chunks",
                    "column_name": "vector",
                }
            )

        executor.submit.assert_called_once()
        self.assertEqual(embedding_server.index_jobs["job-1"]["state"], "queued")
        status, payload = self.handler._send_json.call_args.args
        self.assertEqual(status, embedding_server.HTTPStatus.ACCEPTED)
        self.assertEqual(payload, {"job_id": "job-1"})


if __name__ == "__main__":
    unittest.main()
