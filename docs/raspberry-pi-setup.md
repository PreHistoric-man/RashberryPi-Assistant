# Raspberry Pi setup

This guide prepares a Raspberry Pi 3B+ as a deployment target. It does not
claim that the speech or language models meet performance expectations on the
Pi.

## Target

- Raspberry Pi OS 64-bit with the desktop enabled; Bookworm is the conservative
  starting point for Python 3.11.
- A 64-bit kernel/userspace (`aarch64`) and Python 3.11 are recommended.
- A local desktop session is required for the PySide6 window. The Pi's 3.5-inch
  display must already be configured and visible to that session.
- This project has not yet been installed or hardware-verified on a Pi.

Check the actual OS, architecture, and Python on the Pi before installing:

```sh
cat /etc/os-release
uname -m
python3 --version
```

Use the OS-provided Python; do not replace the system interpreter or use
system-wide `pip`.

## System packages

Install build tools and the audio/Qt runtime libraries:

```sh
sudo apt update
sudo apt install -y \
  git python3 python3-full python3-dev python3-venv \
  build-essential cmake pkg-config libopenblas-dev \
  libportaudio2 portaudio19-dev libasound2-dev \
  libsndfile1 libsndfile1-dev espeak-ng libespeak-ng1 \
  libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0
```

`sounddevice` uses PortAudio/ALSA; `soundfile` uses libsndfile. `pyttsx3`'s
Linux fallback may use eSpeak. The pip PySide6 distribution supplies Qt Python
bindings, while the OS libraries provide common desktop display dependencies.

## Get the source and create the virtual environment

Use the repository URL configured for your GitHub account:

```sh
git clone <repository-url> "$HOME/AiAssistant"
cd "$HOME/AiAssistant"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

All Python packages must be installed inside `.venv`.

## Install Python dependencies

The project keeps a shared `requirements.txt` for Windows and Linux. On ARM64,
install the listed dependencies in the virtual environment:

```sh
export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
export CMAKE_BUILD_PARALLEL_LEVEL=1
python -m pip install -r requirements.txt
python -m pip check
```

`llama-cpp-python` may need to compile llama.cpp because PyPI does not publish
an ARM64 Linux wheel for the current release checked while preparing this
guide. This build can take a long time and may exceed the Pi 3B+'s 1 GB RAM.
Single-job compilation reduces peak build pressure but does not guarantee it
will fit. Do not silently skip the dependency. If the build is killed for
memory, stop and arrange a compatible ARM64 wheel/build artifact for this
Python and OS, or revisit the deployment target before continuing. Do not
replace the system Python, enable CUDA, or download model files as a workaround.

Package compatibility checked for this guide:

| Dependency | ARM64/Linux consideration |
|---|---|
| PySide6 | ARM64 wheels exist. Bookworm's glibc is compatible with the 6.7.x `manylinux_2_31` wheel; newer releases may require newer glibc. If pip cannot select a compatible release, try `python -m pip install "PySide6==6.7.3"` before installing requirements. |
| faster-whisper | The package is Python-only, but relies on native CTranslate2 and PyAV dependencies; install their compatible ARM64 wheels. |
| Piper TTS | ARM64 Linux wheels exist for current releases; runtime libraries and the supplied voice files are still required. |
| sounddevice / soundfile | sounddevice needs system PortAudio; soundfile needs libsndfile. ARM64 soundfile wheels are available for modern manylinux baselines. |
| llama-cpp-python | No ARM64 Linux wheel was listed on PyPI for the current release checked; source compilation is the main installation risk on this 1 GB device. |

These are packaging observations, not a successful install on the Pi. If pip
reports that no compatible distribution exists, record the exact OS/Python/
package version and resolve compatibility explicitly; do not remove
dependencies from `requirements.txt` to make installation appear successful.

## Copy the model files manually

Model weights are intentionally excluded from Git and application startup never
downloads them. Transfer the files to the Pi out-of-band (for example, with
`scp` from a trusted machine), then arrange this structure:

```text
$HOME/AiAssistant/
├── models/
│   ├── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
│   └── tts/
│       ├── en_US-lessac-medium.onnx
│       └── en_US-lessac-medium.onnx.json
└── .venv/
```

The Piper `.onnx.json` companion must be beside the `.onnx` file. Keep the
model path under `$HOME`; do not copy a Windows drive path into the Pi
configuration.

## Configure environment variables

There is no `.env` loader in the application. Export variables in the shell
used to launch it, or source a private shell configuration file. Do not commit
secrets or machine-specific paths.

```sh
export PI_ASSISTANT_LLM_MODEL_PATH="$HOME/AiAssistant/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
export PI_ASSISTANT_LLM_THREADS=3
export PI_ASSISTANT_LLM_CONTEXT_SIZE=2048
export PI_ASSISTANT_LLM_MAX_TOKENS=64
export PI_ASSISTANT_LLM_TEMPERATURE=0.5
export PI_ASSISTANT_LLM_TOP_P=0.9

export PI_ASSISTANT_STT_MODEL=base.en
export PI_ASSISTANT_STT_CPU_THREADS=3

export PI_ASSISTANT_TTS_MODEL_PATH="$HOME/AiAssistant/models/tts/en_US-lessac-medium.onnx"
export PI_ASSISTANT_TTS_VOICE=en_US-lessac-medium
export PI_ASSISTANT_TTS_DEVICE=cpu
```

`PI_ASSISTANT_AUDIO_OUTPUT_DEVICE` is optional. Leave it unset to use the
system default, or set it to a PortAudio device index/name after identifying
the device on the Pi.

## Verify imports and run

From a terminal in the Pi desktop session:

```sh
cd "$HOME/AiAssistant"
source .venv/bin/activate
python -c "import PySide6, sounddevice, soundfile, faster_whisper, llama_cpp, piper"
python -m compileall -q core voice ai ui tests
python -m unittest discover -s tests -q
python main.py
```

The real-model tests in the suite only run when their model environment
variables point to existing files. A test suite pass is not proof of real-time
performance. To verify the display, confirm that the application window opens
on the configured TFT while running within the graphical desktop session.

## Troubleshooting

- **No window / Qt platform plugin error:** run inside the desktop session,
  check `echo "$DISPLAY"`, and confirm the Qt/X11 system libraries above are
  installed. Do not set `QT_QPA_PLATFORM=offscreen` for the real display test.
- **Microphone or speaker unavailable:** inspect PortAudio device names and
  indices from the Pi; install/check ALSA and verify the selected device.
  Hardware audio is outside this setup phase.
- **Model not found:** check the exported absolute path and file permissions.
  The app intentionally does not fetch models.
- **`llama-cpp-python` build fails or is killed:** retain the full pip/CMake
  error. A 1 GB Pi may not have enough memory for the native build; resolve
  that compatibility/build constraint before attempting application startup.
- **Import missing after installation:** ensure `.venv` is active, rerun
  `python -m pip check`, and retain the package/version and error output.

## Compatibility references

- [llama-cpp-python installation](https://github.com/abetlen/llama-cpp-python#installation)
- [faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#requirements)
- [Qt for Python / PySide6](https://doc.qt.io/qtforpython/)
