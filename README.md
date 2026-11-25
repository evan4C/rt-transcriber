# rt-transcriber

A Rust CLI that transcribes the default microphone with Whisper. It prints each recognized segment to standard output and appends the same line to a UTF-8 log file. Audio is processed locally in memory.

## Setup

Install stable Rust, CMake, and a C/C++ toolchain. On macOS, install the Xcode command-line tools. On Debian or Ubuntu, install `build-essential`, `libclang-dev`, `libasound2-dev`, and `pkg-config`.

Download the [multilingual Whisper base model](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin), then build and run in release mode:

```sh
mkdir -p models
curl -fL https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -o models/ggml-base.bin
cargo build --release
./target/release/rt-transcriber --language en
```

Allow microphone access if the operating system prompts for it. Use `--model PATH` to select another multilingual Whisper GGML model. If decoding falls behind on your machine, try the multilingual `ggml-tiny.bin` model.

## Usage

```sh
# English, until Ctrl+C
./target/release/rt-transcriber --language en --duration forever

# Chinese, stop after 15 seconds without recognized words
./target/release/rt-transcriber --language zh --duration auto --timeout 15

# Detect the language of each segment and append to a custom log
./target/release/rt-transcriber --language auto --log-file meeting.log
```

The default language mode is `auto`, the default duration is `forever`, and the default log file is `transcript.log`. In `--duration auto`, the timeout defaults to 30 seconds and starts when listening begins. Recognized words reset it. Ctrl+C works in either mode and processes any buffered speech before exiting.

The CLI emits finalized lines after speech pauses or after about four seconds of continuous speech, plus the time Whisper takes to decode the segment. Use `--help` to see all options.
