"""Run an offline Piper synthesis smoke test.

Install piper-tts, download a Piper voice, and set
PI_ASSISTANT_TTS_MODEL_PATH to its .onnx model, then run
``python -m tools.test_piper_tts`` from the project root.
"""

import argparse
import wave
from pathlib import Path

from voice.tts.piper_tts import PiperTextToSpeechEngine

TEST_TEXT = "Hello! I'm your personal assistant."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("piper_tts_smoke.wav"),
        help="WAV file to write (default: piper_tts_smoke.wav)",
    )
    args = parser.parse_args()

    engine = PiperTextToSpeechEngine()
    if not engine.is_loaded:
        raise RuntimeError("Piper voice model did not load.")

    audio = engine.synthesize(TEST_TEXT)
    engine.synthesize(TEST_TEXT)
    if engine.model_load_count != 1:
        raise RuntimeError(f"Piper model was loaded {engine.model_load_count} times; expected exactly once.")
    if audio.samples.size == 0 or audio.sample_rate <= 0:
        raise RuntimeError("Piper produced empty or invalid audio.")

    channels = 1 if audio.samples.ndim == 1 else audio.samples.shape[1]
    with wave.open(str(args.output), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.samples.astype("<i2", copy=False).tobytes())

    with wave.open(str(args.output), "rb") as wav_file:
        if wav_file.getnframes() == 0 or wav_file.getframerate() <= 0 or wav_file.getsampwidth() != 2:
            raise RuntimeError(f"Generated file '{args.output}' is not valid PCM WAV audio.")

    metrics = engine.last_synthesis_metrics
    print(f"Audio file: {args.output.resolve()}")
    print(f"Voice/model: {metrics['voice']} / {metrics['model_name']}")
    print(f"Synthesis time: {metrics['synthesis_time_seconds']:.3f}s")
    print(f"Audio duration: {metrics['audio_duration_seconds']:.3f}s")
    print(f"Real-time factor: {metrics['real_time_factor']:.3f}")
    print(f"Model loads: {engine.model_load_count}")


if __name__ == "__main__":
    main()
