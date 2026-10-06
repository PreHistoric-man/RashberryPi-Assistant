"""Tests for the TinyLlama/Piper latency benchmark's controlled matrix."""

import unittest

from tools.benchmark_llm_tts import _summarize, configuration_matrix


class LlmTtsBenchmarkTests(unittest.TestCase):
    def test_matrix_covers_requested_values_without_full_cartesian_product(self):
        configurations = configuration_matrix()

        self.assertEqual(len(configurations), 9)
        self.assertEqual({item.threads for item in configurations}, {2, 3, 4})
        self.assertEqual({item.context_size for item in configurations}, {1024, 1536, 2048})
        self.assertEqual({item.max_tokens for item in configurations}, {32, 48, 64})
        self.assertEqual(len({item.name for item in configurations}), len(configurations))

    def test_summary_separates_representative_prompts_from_cached_repeats(self):
        representative = {
            "configuration": "baseline-full-prompt",
            "run_kind": "cold",
            "prompt_tokens": 172,
            "generated_tokens": 9,
            "prompt_eval_seconds": 0.2,
            "generation_seconds": 0.5,
            "tokens_per_second": 18.0,
            "llm_wall_seconds": 0.7,
            "tts_synthesis_seconds": 0.1,
            "tts_audio_duration_seconds": 2.0,
            "tts_rtf": 0.05,
            "llm_to_playback_handoff_seconds": 0.8,
            "reached_max_tokens": False,
        }
        quality = dict(representative, run_kind="quality", prompt_tokens=10, prompt_eval_seconds=0.1)
        repeat = dict(
            representative,
            run_kind="repeat",
            prompt_eval_seconds=0.0,
            llm_wall_seconds=0.4,
            llm_to_playback_handoff_seconds=0.5,
        )

        summary = _summarize([representative, quality, repeat])[0]

        self.assertEqual(summary["representative_prompt_runs"], 2)
        self.assertEqual(summary["warm_repeat_runs"], 1)
        self.assertEqual(summary["median_prompt_tokens"], 91)
        self.assertAlmostEqual(summary["median_prompt_eval_seconds"], 0.15)
        self.assertEqual(summary["median_warm_repeat_handoff_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
