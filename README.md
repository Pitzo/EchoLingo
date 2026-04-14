# EchoLingo

**Real-time system audio translator for Windows.** Listens to everything playing through your speakers — streams, voice chat, videos, games — detects the language, and translates it to English on the fly.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)

---

## What it does

EchoLingo captures your PC's audio output via WASAPI loopback (no virtual cables needed), runs voice activity detection to isolate speech, then uses OpenAI's Whisper model to transcribe and translate any detected language into English — all in near real-time.

```
  [SV 97% -> EN] I don't even know how to play this game!
  [SV 52% -> EN] You just have to take a good position
  [EN 95%] Would you do that?
  [JA 91% -> EN] Today's weather is really nice, isn't it?
```

## Features

- **Zero-config audio capture** — automatically finds your default WASAPI loopback device. No virtual audio cables, no routing setup.
- **99+ languages** — powered by Whisper large-v3, supporting automatic language detection across 99 languages.
- **GPU accelerated** — auto-detects CUDA and uses your NVIDIA GPU for fast inference. Falls back to CPU gracefully.
- **Smart VAD** — Silero Voice Activity Detection filters silence and non-speech audio so Whisper only processes actual speech.
- **Streaming architecture** — segments are flushed as soon as a pause is detected, with lag-dropping to stay live.
- **Context-aware** — feeds previous translations back into Whisper for better coherence across sentences.
- **Temperature fallback** — retries with increasing randomness if the first decoding pass produces garbage.

## Requirements

- **OS:** Windows 10/11
- **Python:** 3.10+
- **GPU (recommended):** NVIDIA GPU with CUDA support (RTX series recommended)
- **CPU fallback:** Works without a GPU, but uses a smaller model and is slower

## Installation

```bash
# Clone the repo
git clone https://github.com/pitzo/echolingo.git
cd echolingo

# Install dependencies (auto-pulls CUDA PyTorch)
pip install -r requirements-translate.txt
```

### Manual install (if you prefer)

```bash
# CUDA PyTorch (recommended — much faster)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install PyAudioWPatch faster-whisper numpy
```

## Usage

```bash
python EchoLingo.py
```

That's it. EchoLingo will:

1. Load the Whisper model (downloads ~3GB on first run)
2. Load the Silero VAD model
3. Find your default audio output device
4. Start listening and translating

Press `Ctrl+C` to stop.

### Example output

```
============================================================
  Real-Time Audio Translator  (streaming)
============================================================
  Whisper : large-v3  on  cuda  (float16)
  Latency : flush after 1.0s silence, max 12.0s segment
  Beam    : 10  (1=fastest, 5=best quality)

[1/3] Loading Whisper...  OK
[2/3] Loading VAD...  OK
[3/3] Opening audio stream...

------------------------------------------------------------
  Listening... Ctrl+C to stop
------------------------------------------------------------
  Device : Speakers (PRO X 2 LIGHTSPEED) [Loopback]
  Format : 48000 Hz / 8ch / 1536 frames (32ms)

  [SV 97% -> EN] I am the only one fighting
  [EN 58%] This is already in English, no translation needed
  [SV 88% -> EN] That's the bomb, go for it!
```

## Configuration

All settings are at the top of `EchoLingo.py`:

| Setting | Default (GPU) | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Whisper model size. Auto-selects `small` on CPU. |
| `BEAM_SIZE` | `10` | Beam search width. Higher = more accurate, slower. |
| `MAX_SEGMENT_SEC` | `12.0` | Max speech duration before force-flushing to Whisper. |
| `SILENCE_TRIGGER_SEC` | `1.0` | Seconds of silence before translating buffered speech. |
| `VAD_THRESHOLD` | `0.35` | Speech detection sensitivity (0.0–1.0). Lower = more sensitive. |
| `MAX_QUEUE_LAG` | `120` | Max queued chunks before dropping old audio to stay live. |

### Tuning for speed vs accuracy

**Fastest (live subtitles):**
```python
BEAM_SIZE = 1
MAX_SEGMENT_SEC = 5.0
SILENCE_TRIGGER_SEC = 0.6
```

**Most accurate (best translation):**
```python
BEAM_SIZE = 10
MAX_SEGMENT_SEC = 12.0
SILENCE_TRIGGER_SEC = 1.0
```

## How it works

```
┌──────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐
│ WASAPI       │───>│ Resample   │───>│ Silero VAD │───>│ Buffer until │
│ Loopback     │    │ to 16kHz   │    │ Speech?    │    │ pause / max  │
│ (48kHz/8ch)  │    │ Mono       │    │            │    │              │
└──────────────┘    └────────────┘    └────────────┘    └──────┬───────┘
                                                               │
                                                               v
                                                        ┌──────────────┐
                                                        │ Whisper      │
                                                        │ large-v3     │
                                                        │ translate    │
                                                        │ -> English   │
                                                        └──────┬───────┘
                                                               │
                                                               v
                                                        ┌──────────────┐
                                                        │ [SV 97%->EN] │
                                                        │ Console out  │
                                                        └──────────────┘
```


## Tech stack

| Component | Library | Purpose |
|---|---|---|
| Audio capture | [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) | WASAPI loopback recording |
| Speech detection | [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection |
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2-accelerated Whisper |
| GPU acceleration | [PyTorch](https://pytorch.org/) + CUDA | GPU inference |

## Supported languages

Whisper large-v3 supports 99 languages including: Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, and more.

## License

MIT
