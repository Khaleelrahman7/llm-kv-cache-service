from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from kv_service.model import FrozenCausalLM
from kv_service.service import GenerationService
from kv_service.tokenizer import Tokenizer


class ServiceTests(unittest.TestCase):
    def test_cached_generation_matches_uncached_generation(self) -> None:
        document = {
            "batch_size": 1,
            "requests": [
                {"request_id": "a", "prompt": "shared prefix", "max_new_tokens": 6},
                {"request_id": "b", "prompt": "shared prefix extended", "max_new_tokens": 8},
            ],
        }
        cached = GenerationService().run_document(document, cache_enabled=True)
        uncached = GenerationService().run_document(document, cache_enabled=False)
        self.assertEqual(
            [item["generated_token_ids"] for item in cached["responses"]],
            [item["generated_token_ids"] for item in uncached["responses"]],
        )
        self.assertEqual(cached["responses"][1]["cache"]["status"], "prefix_hit")

    def test_namespaces_do_not_share_cache_state(self) -> None:
        report = GenerationService().run_document({
            "batch_size": 1,
            "requests": [
                {"request_id": "a", "namespace": "red", "prompt": "same prompt", "max_new_tokens": 3},
                {"request_id": "b", "namespace": "blue", "prompt": "same prompt", "max_new_tokens": 3},
            ],
        })
        self.assertEqual([r["cache"]["status"] for r in report["responses"]], ["miss", "miss"])

    def test_resume_survives_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cache.json"
            first = GenerationService()
            first.run_document({"batch_size": 1, "requests": [
                {"request_id": "job", "prompt": "restart stream", "max_new_tokens": 3},
            ]})
            first.save_checkpoint(checkpoint)

            second = GenerationService()
            second.load_checkpoint(checkpoint)
            resumed = second.run_document({"batch_size": 1, "requests": [
                {"request_id": "job", "prompt": "restart stream", "max_new_tokens": 9, "resume": True},
            ]})
            fresh = GenerationService().run_document({"requests": [
                {"request_id": "fresh", "prompt": "restart stream", "max_new_tokens": 9},
            ]}, cache_enabled=False)
            self.assertEqual(resumed["responses"][0]["generated_token_ids"],
                             fresh["responses"][0]["generated_token_ids"])

    def test_padding_does_not_consume_sequence_positions(self) -> None:
        tokenizer = Tokenizer()
        model = FrozenCausalLM(tokenizer.vocab_size)
        token_ids = tokenizer.encode("padding")
        width = len(token_ids) + 3
        ids = torch.tensor([[0, 0, 0, *token_ids]])
        mask = torch.tensor([[0, 0, 0, *([1] * len(token_ids))]])
        padded = model.forward_padded(ids, mask)
        _, direct = model.advance(token_ids)
        self.assertTrue(torch.equal(padded[0, width - 1], direct))


if __name__ == "__main__":
    unittest.main()

