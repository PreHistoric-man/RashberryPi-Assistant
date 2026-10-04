"""Standalone faster-whisper WAV benchmark for desktop and Raspberry Pi."""

import argparse
import ctypes
import gc
import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import soundfile as sf

try:
    import resource
except ImportError:
    resource = None

from faster_whisper import WhisperModel

logger = logging.getLogger("pi_assistant.stt_benchmark")
SAMPLE_RATE = 16000


def calculate_real_time_factor(transcription_seconds: float, audio_duration: float) -> Optional[float]:
    """Return inference time divided by audio duration, or None for invalid duration."""
    if audio_duration <= 0:
        return None
    return transcription_seconds / audio_duration


def calculate_word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    """Calculate word error rate using normalized whitespace-delimited words."""
    reference_words = re.findall(r"[\w']+", reference.lower())
    hypothesis_words = re.findall(r"[\w']+", hypothesis.lower())
    if not reference_words:
        return None

    previous_row = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[hypothesis_index] + 1,
                    previous_row[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1] / len(reference_words)


def discover_wav_files(input_path: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Resolve one WAV file or all WAV files in a directory, reporting input issues."""
    if not input_path.exists():
        return [], [{"path": str(input_path), "error": "Input path does not exist."}]
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            return [], [{"path": str(input_path), "error": "Input file must have a .wav extension."}]
        return [input_path], []
    if not input_path.is_dir():
        return [], [{"path": str(input_path), "error": "Input path is not a file or directory."}]

    wav_files = sorted(
        child for child in input_path.rglob("*") if child.is_file() and child.suffix.lower() == ".wav"
    )
    if not wav_files:
        return [], [{"path": str(input_path), "error": "No WAV files found in directory."}]
    return wav_files, []


def load_wav(path: Path) -> tuple[np.ndarray, float]:
    """Load non-empty, finite, 16 kHz mono WAV audio for faster-whisper."""
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:
        raise ValueError(f"Could not read WAV: {error}") from error

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio, got {audio.shape[1]} channels.")
    if audio.shape[0] == 0:
        raise ValueError("WAV contains no audio samples.")
    if not np.isfinite(audio).all():
        raise ValueError("WAV contains non-finite audio samples.")

    mono_audio = audio[:, 0]
    return mono_audio, len(mono_audio) / sample_rate


def get_available_memory_mb() -> Optional[float]:
    """Return available system RAM when the platform exposes a reliable value."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
        except (OSError, ValueError, IndexError):
            return None

    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024 * 1024)
    return None


def get_peak_process_rss_mb() -> Optional[float]:
    """Return an approximate process high-water RSS where the OS exposes it."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
        except (OSError, ValueError, IndexError):
            return None

    if resource is not None:
        try:
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak_rss / (1024 * 1024) if sys.platform == "darwin" else peak_rss / 1024
        except (AttributeError, OSError, ValueError):
            return None
    return None


def get_cpu_description() -> str:
    """Report the Pi model when discoverable, otherwise the host CPU description."""
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            model = model_path.read_bytes().replace(b"\x00", b"").decode("utf-8").strip()
            if model:
                return model
        except (OSError, UnicodeDecodeError):
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def get_environment_report() -> dict[str, Any]:
    """Describe the machine so desktop timings cannot be mistaken for Pi results."""
    system = platform.system()
    if system == "Windows":
        label = "Windows development machine (not Raspberry Pi timing)"
    elif "Raspberry Pi" in get_cpu_description():
        label = "Raspberry Pi"
    else:
        label = f"{system} host (verify hardware before treating timings as Pi results)"

    if sys.platform.startswith("linux"):
        peak_rss_source = "Linux /proc/self/status VmHWM; cumulative for this process"
    elif resource is not None:
        peak_rss_source = "resource.getrusage process high-water RSS; cumulative for this process"
    else:
        peak_rss_source = "unavailable on this platform without an optional process-memory dependency"

    return {
        "label": label,
        "platform": platform.platform(),
        "cpu": get_cpu_description(),
        "logical_cpu_count": os.cpu_count(),
        "available_system_memory_mb_before_benchmark": get_available_memory_mb(),
        "peak_process_rss_source": peak_rss_source,
    }


def _transcribe(model: Any, audio: np.ndarray, beam_size: int) -> tuple[str, int, Optional[int], float]:
    started = time.perf_counter()
    segment_iterator, _info = model.transcribe(
        audio,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    segments = list(segment_iterator)
    elapsed = time.perf_counter() - started
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    token_lists = [getattr(segment, "tokens", None) for segment in segments]
    token_count = sum(len(tokens) for tokens in token_lists) if all(tokens is not None for tokens in token_lists) else None
    return text, len(segments), token_count, elapsed


def run_benchmark(
    audio_paths: Sequence[Path],
    configurations: Sequence[tuple[str, int]],
    *,
    runs: int = 3,
    compute_type: str = "int8",
    beam_size: int = 5,
    references: Optional[dict[str, str]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Benchmark valid recordings, isolating model-load and inference measurements."""
    errors: list[dict[str, str]] = []
    audio_items: list[tuple[Path, np.ndarray, float]] = []
    for path in audio_paths:
        try:
            audio, duration = load_wav(path)
            audio_items.append((path, audio, duration))
        except ValueError as error:
            errors.append({"path": str(path), "error": str(error)})

    results: list[dict[str, Any]] = []
    if not audio_items:
        return {"results": results, "errors": errors}

    factory = model_factory or WhisperModel
    for model_name, cpu_threads in configurations:
        try:
            load_started = time.perf_counter()
            model = factory(
                model_name,
                device="cpu",
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            model_load_seconds = time.perf_counter() - load_started
        except Exception as error:
            errors.append({
                "model": model_name,
                "cpu_threads": str(cpu_threads),
                "error": f"Model load failed: {error}",
            })
            continue

        warmup_path, warmup_audio, _warmup_duration = audio_items[0]
        try:
            _warmup_text, _warmup_segments, _warmup_tokens, warmup_seconds = _transcribe(
                model, warmup_audio, beam_size
            )
        except Exception as error:
            errors.append({
                "model": model_name,
                "cpu_threads": str(cpu_threads),
                "path": str(warmup_path),
                "error": f"Warm-up transcription failed: {error}",
            })
            del model
            gc.collect()
            continue

        for path, audio, duration in audio_items:
            for run_index in range(1, runs + 1):
                try:
                    text, segment_count, token_count, inference_seconds = _transcribe(model, audio, beam_size)
                except Exception as error:
                    errors.append({
                        "model": model_name,
                        "cpu_threads": str(cpu_threads),
                        "path": str(path),
                        "error": f"Inference run {run_index} failed: {error}",
                    })
                    continue

                result = {
                    "model": model_name,
                    "compute_type": compute_type,
                    "cpu_threads": cpu_threads,
                    "model_load_seconds": model_load_seconds,
                    "warmup_seconds": warmup_seconds,
                    "audio_file": str(path),
                    "audio_duration_seconds": duration,
                    "run": run_index,
                    "transcription_seconds": inference_seconds,
                    "real_time_factor": calculate_real_time_factor(inference_seconds, duration),
                    "segment_count": segment_count,
                    "token_count": token_count,
                    "approximate_peak_process_rss_mb": get_peak_process_rss_mb(),
                    "transcribed_text": text,
                }
                if references is not None and path.name in references:
                    result["word_error_rate"] = calculate_word_error_rate(references[path.name], text)
                results.append(result)

        del model
        gc.collect()

    return {"results": results, "errors": errors}


def _build_configurations(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.mode == "models":
        models = args.models
        threads = args.threads or [4]
    elif args.mode == "threads":
        models = [args.model or "tiny.en"]
        threads = args.threads or [2, 3, 4]
    else:
        models = [args.model or "tiny.en"]
        threads = args.threads or [4]
        threads = threads[:1]
    return [(model, thread_count) for model in models for thread_count in threads]


def _load_references(path: Optional[Path]) -> Optional[dict[str, str]]:
    if path is None:
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in loaded.items()
    ):
        raise ValueError("Reference JSON must map WAV basenames to transcript strings.")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", "--input", required=True, type=Path, help="A 16 kHz mono WAV file or directory of WAV files.")
    parser.add_argument("--mode", choices=("models", "threads", "repeat"), default="models")
    parser.add_argument("--models", nargs="+", default=["tiny.en", "base.en", "small"], help="Models for models mode.")
    parser.add_argument("--model", help="One model for threads/repeat mode; defaults to tiny.en.")
    parser.add_argument("--threads", nargs="+", type=int, help="CPU thread counts; defaults by benchmark mode.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per WAV and configuration.")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--references", type=Path, help="Optional JSON mapping WAV basenames to reference transcripts.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path instead of stdout.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.beam_size < 1:
        parser.error("--beam-size must be at least 1")
    if args.threads and any(thread_count < 1 for thread_count in args.threads):
        parser.error("benchmark thread counts must be positive integers")

    try:
        references = _load_references(args.references)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"Could not read references: {error}")

    audio_paths, input_errors = discover_wav_files(args.audio)
    report = {
        "environment": get_environment_report(),
        "benchmark": {
            "mode": args.mode,
            "device": "cpu",
            "compute_type": args.compute_type,
            "runs_per_audio": args.runs,
            "warmup_transcriptions_per_configuration": 1,
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
        },
    }
    report.update(
        run_benchmark(
            audio_paths,
            _build_configurations(args),
            runs=args.runs,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            references=references,
        )
    )
    report["errors"] = input_errors + report["errors"]

    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"Benchmark report written to {args.output}")
    else:
        print(output_text)
    return 0 if report["results"] else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())