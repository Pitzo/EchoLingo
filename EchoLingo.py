import sys
import time
import queue
import threading
import traceback
import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    sys.exit("Install PyAudioWPatch:  pip install PyAudioWPatch")

try:
    import torch
except ImportError:
    sys.exit("Install PyTorch:  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("Install faster-whisper:  pip install faster-whisper")


def log(msg: str):
    print(msg, flush=True)


# ── Device auto-detection ────────────────────────────────────────────────────

def _pick_device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda", "float16", "large-v3"
    except Exception:
        pass
    return "cpu", "int8", "small"

DEVICE, COMPUTE_TYPE, WHISPER_MODEL = _pick_device()

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_SR = 16000
VAD_THRESHOLD = 0.35
SILENCE_TRIGGER_SEC = 1.0          # wait for a proper pause before translating
MAX_SEGMENT_SEC = 12.0             # more context = much better translation quality
MIN_AUDIO_SEC = 0.5                # ignore segments shorter than this
BEAM_SIZE = 10                     # max quality beam search
MAX_QUEUE_LAG = 120                # allow more buffering since processing is heavier
MIN_CONFIDENCE = 0.0               # language detection confidence filter (0.0 = show all, 0.5 = 50%+, etc.)

# ── Globals ──────────────────────────────────────────────────────────────────

audio_q: queue.Queue = queue.Queue()
stop_ev = threading.Event()


# ── Audio helpers ────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    target_len = int(len(audio) * target_sr / orig_sr)
    if target_len == 0:
        return np.array([], dtype=np.float32)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def to_mono_f32(data: bytes, channels: int) -> np.ndarray:
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        trim = len(audio) - (len(audio) % channels)
        audio = audio[:trim].reshape(-1, channels).mean(axis=1)
    return audio


# ── WASAPI loopback ──────────────────────────────────────────────────────────

def find_loopback(p: pyaudio.PyAudio) -> dict:
    wasapi = None
    for i in range(p.get_host_api_count()):
        info = p.get_host_api_info_by_index(i)
        if info["name"] == "Windows WASAPI":
            wasapi = info
            break
    if wasapi is None:
        sys.exit("Windows WASAPI not found.")

    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if (
            dev["hostApi"] == wasapi["index"]
            and dev["name"].endswith("[Loopback]")
            and default_out["name"] in dev["name"]
        ):
            return dev
    sys.exit("No WASAPI loopback device found.")


# ── VAD ──────────────────────────────────────────────────────────────────────

def load_vad():
    model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    return model


def vad_check(vad_model, audio_16k: np.ndarray) -> bool:
    """Return True if any 512-sample frame in audio_16k contains speech."""
    fs = 512
    if len(audio_16k) < fs:
        return False
    for i in range(0, len(audio_16k) - fs + 1, fs):
        t = torch.from_numpy(audio_16k[i : i + fs]).float()
        if vad_model(t, TARGET_SR).item() > VAD_THRESHOLD:
            return True
    return False


# ── Capture thread ───────────────────────────────────────────────────────────

def capture_thread(dev: dict, pa: pyaudio.PyAudio):
    sr = int(dev["defaultSampleRate"])
    ch = int(dev["maxInputChannels"])
    chunk = int(sr * (512 / TARGET_SR))

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=ch,
        rate=sr,
        input=True,
        input_device_index=int(dev["index"]),
        frames_per_buffer=chunk,
    )
    log(f"  Device : {dev['name']}")
    log(f"  Format : {sr} Hz / {ch}ch / {chunk} frames ({chunk/sr*1000:.0f}ms)")

    try:
        while not stop_ev.is_set():
            try:
                raw = stream.read(chunk, exception_on_overflow=False)
                mono = to_mono_f32(raw, ch)
                mono = resample(mono, sr, TARGET_SR)
                audio_q.put(mono)
            except OSError:
                time.sleep(0.005)
    except Exception as e:
        log(f"  [Capture Error] {e}")
    finally:
        stream.stop_stream()
        stream.close()


# ── Processor thread ─────────────────────────────────────────────────────────

def process_thread(whisper: WhisperModel, vad):
    buf: list[np.ndarray] = []
    buf_sec = 0.0
    sil_sec = 0.0
    speaking = False

    vad.reset_states()

    while not stop_ev.is_set():
        # Drain queue — if we're lagging, skip old audio to stay live
        chunks = []
        try:
            while True:
                chunks.append(audio_q.get_nowait())
        except queue.Empty:
            pass

        if not chunks:
            # Nothing in queue — check if we should flush on timeout
            if speaking and buf_sec >= MIN_AUDIO_SEC:
                sil_sec += 0.05
                if sil_sec >= SILENCE_TRIGGER_SEC:
                    _flush(whisper, buf, buf_sec)
                    buf, buf_sec, sil_sec, speaking = [], 0.0, 0.0, False
                    vad.reset_states()
            time.sleep(0.05)
            continue

        # If severely behind, drop all but the last MAX_QUEUE_LAG chunks
        if len(chunks) > MAX_QUEUE_LAG:
            dropped = len(chunks) - MAX_QUEUE_LAG
            chunks = chunks[-MAX_QUEUE_LAG:]
            log(f"  [Lag] Dropped {dropped} chunks to stay live")
            # Also discard current buffer since it's stale
            if buf:
                buf, buf_sec, sil_sec, speaking = [], 0.0, 0.0, False
                vad.reset_states()

        for chunk in chunks:
            csec = len(chunk) / TARGET_SR
            has_speech = vad_check(vad, chunk)

            if has_speech:
                buf.append(chunk)
                buf_sec += csec
                sil_sec = 0.0
                speaking = True

                # Force-flush long segments to keep latency low
                if buf_sec >= MAX_SEGMENT_SEC:
                    _flush(whisper, buf, buf_sec)
                    buf, buf_sec, sil_sec, speaking = [], 0.0, 0.0, False
                    vad.reset_states()

            elif speaking:
                buf.append(chunk)
                buf_sec += csec
                sil_sec += csec

                if sil_sec >= SILENCE_TRIGGER_SEC:
                    if buf_sec >= MIN_AUDIO_SEC:
                        _flush(whisper, buf, buf_sec)
                    buf, buf_sec, sil_sec, speaking = [], 0.0, 0.0, False
                    vad.reset_states()


_prev_text = ""

def _flush(model: WhisperModel, buf: list[np.ndarray], dur: float):
    global _prev_text
    audio = np.concatenate(buf)
    try:
        segs, info = model.transcribe(
            audio,
            task="translate",
            beam_size=BEAM_SIZE,
            best_of=5,                  # sample 5 candidates, pick the best
            patience=2.0,               # very patient beam search — explores more paths
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=True,
            initial_prompt=_prev_text[-300:] if _prev_text else None,
            temperature=[0.0, 0.2, 0.4, 0.6],  # retry with higher temps if first pass fails
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            repetition_penalty=1.1,     # reduce repeated words/phrases
        )

        text = " ".join(s.text.strip() for s in segs).strip()
        if not text:
            return

        lang = info.language.upper()
        prob = info.language_probability

        if prob < MIN_CONFIDENCE:
            log(f"  [Skipped] {lang} {prob:.0%} below {MIN_CONFIDENCE:.0%} threshold")
            return

        _prev_text = text

        if info.language == "en":
            log(f"  [{lang} {prob:.0%}] {text}")
        else:
            log(f"  [{lang} {prob:.0%} -> EN] {text}")

    except Exception as e:
        log(f"  [Error] {e}")
        traceback.print_exc()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Real-Time Audio Translator  (streaming)")
    log("=" * 60)
    log(f"  Whisper : {WHISPER_MODEL}  on  {DEVICE}  ({COMPUTE_TYPE})")
    log(f"  Latency : flush after {SILENCE_TRIGGER_SEC}s silence, max {MAX_SEGMENT_SEC}s segment")
    log(f"  Beam    : {BEAM_SIZE}  (1=fastest, 5=best quality)")
    log(f"  Filter  : {MIN_CONFIDENCE:.0%} min confidence  (0%=show all)")
    log("")

    log("[1/3] Loading Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    log("  OK")

    log("[2/3] Loading VAD...")
    vad = load_vad()
    log("  OK")

    log("[3/3] Opening audio stream...")
    pa = pyaudio.PyAudio()
    try:
        dev = find_loopback(pa)
    except SystemExit as e:
        pa.terminate()
        raise e

    t_cap = threading.Thread(target=capture_thread, args=(dev, pa), daemon=True)
    t_proc = threading.Thread(target=process_thread, args=(whisper, vad), daemon=True)
    t_cap.start()
    t_proc.start()

    log("")
    log("-" * 60)
    log("  Listening... Ctrl+C to stop")
    log("-" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nShutting down...")
        stop_ev.set()
        t_cap.join(timeout=2)
        t_proc.join(timeout=2)
        pa.terminate()
        log("Done.")


if __name__ == "__main__":
    main()
