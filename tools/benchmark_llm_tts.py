"""Measure local TinyLlama and Piper response latency without audio hardware."""

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Dict, List, Optional

from ai.response_engine import LocalResponseEngine
from ai.tinyllama import TinyLlama
from voice.tts.piper_tts import PiperTextToSpeechEngine

CURRENT_SYSTEM_PROMPT = TinyLlama.DEFAULT_SYSTEM_INSTRUCTION
COMPACT_SYSTEM_PROMPT = (
    "You are a concise personal AI assistant.\n"
    "Answer directly and naturally.\n"
    "Usually respond in 1–3 short sentences.\n"
    "For simple questions, use one sentence.\n"
    "Do not repeat the question or add unnecessary introductions.\n"
    "Give more detail only when asked."
)
REPRESENTATIVE_PROMPTS = (
    "What is the capital of India?",
    "Explain what Python is.",
    "Set a reminder for tomorrow.",
    "Tell me a joke.",
    "What can you help me with?",
    "What's 25 times 16?",
)
REPEAT_PROMPT = REPRESENTATIVE_PROMPTS[0]


@dataclass(frozen=True)
class Configuration:
    name: str
    threads: int
    context_size: int
    max_tokens: int
    system_prompt: str


def configuration_matrix() -> List[Configuration]:
    """Return a small one-factor-at-a-time matrix plus the current baseline."""
    return [
        Configuration("baseline-full-prompt", 4, 2048, 64, CURRENT_SYSTEM_PROMPT),
        Configuration("baseline-compact-prompt", 4, 2048, 64, COMPACT_SYSTEM_PROMPT),
        Configuration("threads-2", 2, 1024, 48, COMPACT_SYSTEM_PROMPT),
        Configuration("threads-3", 3, 1024, 48, COMPACT_SYSTEM_PROMPT),
        Configuration("threads-4", 4, 1024, 48, COMPACT_SYSTEM_PROMPT),
        Configuration("context-1536", 3, 1536, 48, COMPACT_SYSTEM_PROMPT),
        Configuration("context-2048", 3, 2048, 48, COMPACT_SYSTEM_PROMPT),
        Configuration("tokens-32", 3, 1024, 32, COMPACT_SYSTEM_PROMPT),
        Configuration("tokens-64", 3, 1024, 64, COMPACT_SYSTEM_PROMPT),
    ]


class NullAudioPlayer:
    """Accept prepared PCM without requiring an output device or playing audio."""

    def __init__(self):
        self.calls = 0
        self.audio_sample_counts: List[int] = []

    def play(self, _audio) -> None:
        self.calls += 1
        self.audio_sample_counts.append(int(_audio.samples.size))

    def stop(self) -> None:
        pass


def _run_sample(
    engine: LocalResponseEngine,
    tts: PiperTextToSpeechEngine,
    player: NullAudioPlayer,
    prompt: str,
    run_kind: str,
    configuration: Configuration,
    model_load_seconds: float,
    include_cold_total: bool,
) -> Dict[str, Any]:
    start = time.perf_counter()
    response = engine.generate_response(prompt)
    llm_wall_seconds = time.perf_counter() - start
    audio = tts.speak(response)
    end_to_end_seconds = time.perf_counter() - start
    metrics = engine.last_generation_metrics
    tts_metrics = tts.last_synthesis_metrics
    row = {
        "configuration": configuration.name,
        "threads": configuration.threads,
        "context_size": configuration.context_size,
        "max_tokens": configuration.max_tokens,
        "system_prompt": "compact" if configuration.system_prompt == COMPACT_SYSTEM_PROMPT else "current",
        "run_kind": run_kind,
        "prompt": prompt,
        "response": response,
        "prompt_tokens": metrics.get("prompt_tokens"),
        "generated_tokens": metrics.get("generated_tokens"),
        "prompt_eval_seconds": metrics.get("prompt_eval_time_seconds"),
        "generation_seconds": metrics.get("generation_time_seconds"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "llm_wall_seconds": metrics.get("llm_wall_time_seconds", llm_wall_seconds),
        "time_to_first_token_seconds": metrics.get("time_to_first_token_seconds"),
        "finish_reason": metrics.get("finish_reason"),
        "reached_max_tokens": metrics.get("reached_max_tokens"),
        "tts_synthesis_seconds": tts_metrics.get("synthesis_time_seconds"),
        "tts_audio_duration_seconds": audio.duration_seconds,
        "tts_rtf": tts_metrics.get("real_time_factor"),
        "playback_handoff_seconds": tts_metrics.get("playback_handoff_seconds"),
        "llm_to_playback_handoff_seconds": end_to_end_seconds,
        "audio_player_calls": player.calls,
        "audio_samples_handed_off": player.audio_sample_counts[-1],
        "cold_model_load_seconds": model_load_seconds if run_kind == "cold" else None,
        "cold_total_with_model_load_seconds": (
            model_load_seconds + end_to_end_seconds if include_cold_total else None
        ),
        "tts_model_load_seconds": tts_metrics.get("load_time_seconds"),
    }
    return row


def _streaming_probe(
    llm: TinyLlama, configuration: Configuration, prompt: str = REPEAT_PROMPT
) -> Dict[str, Any]:
    """Measure first non-empty streamed text separately from normal generation."""
    messages = [
        {"role": "system", "content": configuration.system_prompt},
        {"role": "user", "content": prompt},
    ]
    start = time.perf_counter()
    first_token_seconds: Optional[float] = None
    chunks: List[str] = []
    finish_reason = None
    stream = llm._model.create_chat_completion(
        messages=messages,
        max_tokens=configuration.max_tokens,
        temperature=llm.temperature,
        top_p=llm.top_p,
        stream=True,
    )
    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")
        if content:
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - start
            chunks.append(content)
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
    return {
        "configuration": configuration.name,
        "prompt": prompt,
        "time_to_first_token_seconds": first_token_seconds,
        "stream_total_seconds": time.perf_counter() - start,
        "finish_reason": finish_reason,
        "response": "".join(chunks).strip(),
    }


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for configuration in configuration_matrix():
        representative_samples = [
            row for row in rows
            if row["configuration"] == configuration.name and row["run_kind"] != "repeat"
        ]
        repeat_samples = [
            row for row in rows
            if row["configuration"] == configuration.name and row["run_kind"] == "repeat"
        ]
        if not representative_samples:
            continue

        def median(samples: List[Dict[str, Any]], field: str) -> Optional[float]:
            values = [row[field] for row in samples if row.get(field) is not None]
            return statistics.median(values) if values else None

        summary.append({
            "configuration": configuration.name,
            "threads": configuration.threads,
            "context_size": configuration.context_size,
            "max_tokens": configuration.max_tokens,
            "representative_prompt_runs": len(representative_samples),
            "warm_repeat_runs": len(repeat_samples),
            "median_prompt_tokens": median(representative_samples, "prompt_tokens"),
            "median_generated_tokens": median(representative_samples, "generated_tokens"),
            "median_prompt_eval_seconds": median(representative_samples, "prompt_eval_seconds"),
            "median_generation_seconds": median(representative_samples, "generation_seconds"),
            "median_tokens_per_second": median(representative_samples, "tokens_per_second"),
            "median_llm_wall_seconds": median(representative_samples, "llm_wall_seconds"),
            "median_tts_synthesis_seconds": median(representative_samples, "tts_synthesis_seconds"),
            "median_tts_audio_duration_seconds": median(representative_samples, "tts_audio_duration_seconds"),
            "median_tts_rtf": median(representative_samples, "tts_rtf"),
            "median_llm_to_playback_handoff_seconds": median(
                representative_samples, "llm_to_playback_handoff_seconds"
            ),
            "median_warm_repeat_llm_wall_seconds": median(repeat_samples, "llm_wall_seconds"),
            "median_warm_repeat_handoff_seconds": median(
                repeat_samples, "llm_to_playback_handoff_seconds"
            ),
            "max_token_limit_hits": sum(row.get("reached_max_tokens") is True for row in rows
                                        if row["configuration"] == configuration.name),
        })
    return summary


def _print_summary(summary: List[Dict[str, Any]]) -> None:
    columns = (
        "configuration", "threads", "context_size", "max_tokens",
        "representative_prompt_runs", "warm_repeat_runs",
        "median_prompt_tokens", "median_generated_tokens", "median_prompt_eval_seconds",
        "median_generation_seconds", "median_tokens_per_second",
        "median_llm_wall_seconds", "median_tts_synthesis_seconds",
        "median_llm_to_playback_handoff_seconds",
        "median_warm_repeat_handoff_seconds", "max_token_limit_hits",
    )
    print("\t".join(columns))
    for row in summary:
        print("\t".join(
            "--" if row.get(column) is None else str(row[column])
            for column in columns
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "matrix"),
        default="matrix",
        help="Run only the current baseline or the controlled configuration matrix.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Warm repeated runs of the capital-of-India prompt per configuration (minimum 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the detailed JSON report.",
    )
    parser.add_argument(
        "--no-streaming-probe",
        action="store_true",
        help="Skip the separate raw llama.cpp time-to-first-token probe.",
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 so warm performance has repeated measurements.")

    model_path = os.getenv("PI_ASSISTANT_LLM_MODEL_PATH")
    tts_path = os.getenv("PI_ASSISTANT_TTS_MODEL_PATH")
    if not model_path or not Path(model_path).is_file():
        parser.error("Set PI_ASSISTANT_LLM_MODEL_PATH to an existing local TinyLlama GGUF.")
    if not tts_path or not Path(tts_path).is_file():
        parser.error("Set PI_ASSISTANT_TTS_MODEL_PATH to an existing local Piper ONNX model.")

    configurations = configuration_matrix()
    if args.mode == "baseline":
        configurations = configurations[:1]

    player = NullAudioPlayer()
    tts = PiperTextToSpeechEngine(model_path=tts_path, device="cpu", audio_player=player)
    rows: List[Dict[str, Any]] = []
    streaming_probe = None
    streaming_configuration = next(
        config for config in configurations if config.name == "threads-3"
    ) if args.mode == "matrix" else None

    for configuration in configurations:
        print(
            f"Loading {configuration.name}: threads={configuration.threads}, "
            f"context={configuration.context_size}, max_tokens={configuration.max_tokens}",
            flush=True,
        )
        llm = TinyLlama(
            model_path=model_path,
            threads=configuration.threads,
            context_size=configuration.context_size,
            max_tokens=configuration.max_tokens,
            temperature=0.5,
            top_p=0.9,
            system_instruction=configuration.system_prompt,
        )
        engine = LocalResponseEngine(
            llm=llm,
            system_instruction=configuration.system_prompt,
        )
        load_start = time.perf_counter()
        llm.load()
        model_load_seconds = time.perf_counter() - load_start
        model_identity = id(llm._model)

        for index, prompt in enumerate(REPRESENTATIVE_PROMPTS):
            rows.append(_run_sample(
                engine, tts, player, prompt,
                "cold" if index == 0 else "quality",
                configuration, model_load_seconds, include_cold_total=index == 0,
            ))

        for _ in range(args.runs):
            rows.append(_run_sample(
                engine, tts, player, REPEAT_PROMPT, "repeat",
                configuration, model_load_seconds, include_cold_total=False,
            ))

        if model_identity != id(llm._model) or llm.model_load_count != 1:
            raise RuntimeError(
                f"{configuration.name}: TinyLlama was not reused as one loaded model instance."
            )
        if (
            streaming_configuration is configuration
            and not args.no_streaming_probe
        ):
            streaming_probe = _streaming_probe(llm, configuration)
        print(
            f"Completed {configuration.name}: load={model_load_seconds:.3f}s, "
            f"one model load, {len(REPRESENTATIVE_PROMPTS) + args.runs} interactions",
            flush=True,
        )
        llm.unload()
        del engine, llm
        gc.collect()

    if tts.model_load_count != 1 or player.calls != len(rows):
        raise RuntimeError("Piper reuse or playback-handoff count did not match benchmark calls.")

    summary = _summarize(rows)
    report = {
        "environment": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "unavailable",
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "llama_cpp_model": Path(model_path).name,
            "piper_model": Path(tts_path).name,
            "llm_backend": "llama.cpp CPU",
            "tts_backend": "Piper CPU",
            "playback": "No-op audio sink; measures audio handoff, not audible output",
            "raspberry_pi_performance": "not measured",
        },
        "mode": args.mode,
        "warm_repeat_count": args.runs,
        "llm_model_load_count_per_configuration": 1,
        "piper_model_load_count": tts.model_load_count,
        "summary": summary,
        "streaming_probe": streaming_probe,
        "samples": rows,
    }
    _print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Detailed report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
