# Faster-Whisper Benchmarking

`tools/benchmark_stt.py` runs independently of the PySide6 UI. It accepts one WAV file or a directory of WAV files, requires 16 kHz mono audio, warms each model/thread configuration once, and reports model load time separately from measured transcription runs. It does not generate audio or invent recognition results.

## TinyLlama and Piper Response Latency

`tools/benchmark_llm_tts.py` measures the current TinyLlama → Piper path using
the locally configured models. It requires `PI_ASSISTANT_LLM_MODEL_PATH` and
`PI_ASSISTANT_TTS_MODEL_PATH`; it never downloads models. A no-op audio player
accepts the synthesized PCM, so the report measures audio handoff preparation,
not audible playback or hardware latency.

Run the current full-prompt baseline or the small controlled matrix from the
project root:

```powershell
python -m tools.benchmark_llm_tts --mode baseline --runs 3 --output "$env:TEMP\assistant-baseline.json"
python -m tools.benchmark_llm_tts --mode matrix --runs 3 --output "$env:TEMP\assistant-matrix.json"
```

The matrix compares the current and compact system prompts at 4 threads,
2048 context, and 64 output tokens; threads 2/3/4 at context 1024 and 48
output tokens; contexts 1024/1536/2048 at 3 threads and 48 output tokens; and
output caps 32/48/64 at 3 threads and context 1024. Each candidate also runs
all six representative prompts once and the India question three additional
times for warm repeat measurements. Configurations are loaded sequentially,
one TinyLlama at a time; the Piper voice is loaded once and reused throughout.
The six-prompt medians represent varied prompts; the repeated identical India
question is reported separately because llama.cpp may reuse its cached prompt.

Reports include model load duration, actual prompt/completion token counts,
llama.cpp prompt-evaluation and token-generation timings, generated tokens per
second, completion finish reason and max-token warnings, Piper synthesis time,
audio duration, RTF, and total LLM-to-audio-handoff preparation time. The
optional raw llama.cpp streaming probe measures time to first non-empty token
for the 3-thread/1024/48 compact-prompt candidate; it does not change the
application's non-streaming behavior or indicate that the UI is streaming.
Inspect the saved response text for correctness, naturalness, unsupported
claims, and sentence-boundary truncation before selecting a configuration.

Windows numbers are for configuration comparison only. Run the same command
on Raspberry Pi OS with the model paths set there for target measurements.
The script labels its no-op playback sink and does not report Windows results
as Raspberry Pi results.

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