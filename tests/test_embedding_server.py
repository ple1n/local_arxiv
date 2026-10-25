import unittest
import warnings
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


class ProductQuantizationTests(unittest.TestCase):
    def test_quantizer_iterations_update_job_progress(self):
        job_id = "quantizer-progress-test"
        embedding_server.index_jobs[job_id] = {"stage": "train_quantizer"}
        with patch.object(embedding_server, "_push_index_progress") as push:
            report = embedding_server._make_quantizer_progress_tqdm(
                job_id, 10, "http://backend/progress", lambda iterable, **kwargs: iterable
            )

            for _ in range(10):
                self.assertEqual(list(report(range(50))), list(range(50)))
            self.assertEqual(push.call_count, 50)
            self.assertEqual(push.call_args_list[-1].args[1]["completed"], 500)
        self.assertEqual(embedding_server.index_jobs[job_id]["stage"], "PQ codebook 10/10")
        self.assertEqual(embedding_server.index_jobs[job_id]["completed"], 500)
        self.assertEqual(embedding_server.index_jobs[job_id]["total"], 500)
        del embedding_server.index_jobs[job_id]

    def test_subvector_count_matches_lance_dimension_defaults(self):
        self.assertEqual(embedding_server._suggest_num_sub_vectors(1536), 96)
        self.assertEqual(embedding_server._suggest_num_sub_vectors(768), 48)
        self.assertEqual(embedding_server._suggest_num_sub_vectors(33), 1)

    def test_nonpositive_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            embedding_server._suggest_num_sub_vectors(0)

    def test_partition_count_uses_available_vector_count(self):
        self.assertEqual(embedding_server._suggest_num_partitions(3354), 1)
        self.assertEqual(embedding_server._suggest_num_partitions(8192), 1)
        self.assertEqual(embedding_server._suggest_num_partitions(16384), 2)
        self.assertEqual(embedding_server._suggest_num_partitions(8192 * 5000), 4096)

    def test_nonpositive_vector_count_is_rejected(self):
        with self.assertRaises(ValueError):
            embedding_server._suggest_num_partitions(0)

    def test_fixed_size_list_conversion_uses_writable_numpy_memory(self):
        import pyarrow as pa

        vectors = pa.FixedSizeListArray.from_arrays(
            pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 3
        )
        batch = pa.record_batch(
            [vectors, pa.array([10, 11], type=pa.uint64())],
            names=["vector", "_rowid"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tensors = embedding_server._writable_lance_to_tensor(batch)

        self.assertEqual(
            tensors["vector"].tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        self.assertEqual(tensors["_rowid"].tolist(), [10, 11])
        self.assertFalse(
            any("not writable" in str(warning.message).lower() for warning in caught)
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
                    "callback_url": "http://127.0.0.1:8250/index-builds/job-1/progress",
                }
            )

        executor.submit.assert_called_once()
        self.assertEqual(embedding_server.index_jobs["job-1"]["state"], "queued")
        status, payload = self.handler._send_json.call_args.args
        self.assertEqual(status, embedding_server.HTTPStatus.ACCEPTED)
        self.assertEqual(payload, {"job_id": "job-1"})

    def test_non_loopback_callback_is_rejected(self):
        self.handler._start_index_job(
            {
                "job_id": "job-2",
                "database_path": "/tmp/lance",
                "table_name": "chunks",
                "column_name": "vector",
                "callback_url": "http://example.invalid/index-builds/job-2/progress",
            }
        )

        status, payload = self.handler._send_json.call_args.args
        self.assertEqual(status, embedding_server.HTTPStatus.BAD_REQUEST)
        self.assertIn("callback_url", payload["error"])


if __name__ == "__main__":
    unittest.main()
