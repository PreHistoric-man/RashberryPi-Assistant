# Faster-Whisper Benchmarking

`tools/benchmark_stt.py` runs independently of the PySide6 UI. It accepts one WAV file or a directory of WAV files, requires 16 kHz mono audio, warms each model/thread configuration once, and reports model load time separately from measured transcription runs. It does not generate audio or invent recognition results.

## Prepare Audio

Provide a directory containing several real recordings, including short and longer utterances. Useful reference prompts are:

- "Hello, this is a test of my Raspberry Pi AI assistant."
- "What is the weather today?"
- "Tell me a short joke."
- "Explain what Python is in simple words."
- "Please remind me to study at seven in the evening."

Save each recording as a 16 kHz mono WAV. The benchmark validates those properties and reports unreadable, empty, stereo, or wrong-rate files instead of using them. To calculate WER, provide a JSON file keyed by WAV basename:

```json
{
  "short.wav": "What is the weather today?",
  "long.wav": "Please remind me to study at seven in the evening."
}
```

Without references, compare the `transcribed_text` entries manually; no accuracy claim is made.

## Modes

Run from the project root. Each mode defaults to three measured repetitions after one warm-up:

```powershell
python tools/benchmark_stt.py --audio path\to\wav-directory --mode models --output models.json
python tools/benchmark_stt.py --audio path\to\wav-directory --mode threads --model tiny.en --output threads.json
python tools/benchmark_stt.py --audio path\to\wav-directory --mode repeat --model base.en --threads 3 --runs 3 --output repeat.json
```

`models` compares `tiny.en`, `base.en`, and `small` at four threads. `threads` compares 2, 3, and 4 threads on `tiny.en`. `repeat` measures the selected model/thread combination three times. Override models, threads, and repetitions with `--models`, `--threads`, and `--runs`. Model downloads are performed by faster-whisper as needed; the script does not change system settings.

For a single file, pass that WAV path to `--audio` instead of a directory. Add `--references references.json` to include per-file WER. Reports include environment/CPU information, available system memory before benchmarking, approximate process peak RSS when the OS exposes it, warm-up duration, model load duration, audio duration, inference time, RTF, segment/token counts when available, and transcription text. Windows output is labeled as development-machine data and is not a Raspberry Pi result.

## Raspberry Pi 3B+

Run on Raspberry Pi OS from the project root after installing the project's Python dependencies and downloading/caching the selected faster-whisper model:

```bash
python3 tools/benchmark_stt.py --audio /path/to/16khz-mono-wavs --mode models --output pi-models.json
python3 tools/benchmark_stt.py --audio /path/to/16khz-mono-wavs --mode threads --model tiny.en --output pi-threads.json
python3 tools/benchmark_stt.py --audio /path/to/16khz-mono-wavs --mode repeat --model base.en --threads 3 --runs 3 --output pi-repeat.json
```

Record model, CPU threads, model-load time, warm-up time, audio duration, each inference duration, RTF, available RAM, approximate peak process RSS, and recognized text. RTF is `transcription time / audio duration`: below 1.0 is faster than real time; lower is faster. Compare transcription text (or WER with references) as well as speed. Recommend `tiny.en` only if its command-critical recognition is acceptable; otherwise prefer `base.en` if it remains practically real-time. Keep `small` as the comparison/fallback model, and do not select a model from Windows timings as though they were Pi timings.

## Application Configuration

The application keeps `small` as its default until Pi recordings establish a better choice, and defaults to three CPU threads to leave headroom for the UI and OS. Configure it before launching with environment variables:

```bash
PI_ASSISTANT_STT_MODEL=base.en PI_ASSISTANT_STT_CPU_THREADS=3 python3 main.py
```

On PowerShell:

```powershell
$env:PI_ASSISTANT_STT_MODEL = "base.en"
$env:PI_ASSISTANT_STT_CPU_THREADS = "3"
python main.py
```

The model is loaded once when the STT engine is constructed and reused for subsequent transcriptions. The Developer Mode STT label includes the active model, compute type, and CPU thread count.