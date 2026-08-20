"""The Silent Co-Driver - read driver stress from team radio.

Three Hugging Face Hub models chained:
  whisper-small ............. audio -> transcript
  hubert-large-superb-er .... audio -> vocal emotion (how it was said)
  distilroberta-emotion ..... text  -> word emotion (what was said)
Blended into one 0-100 stress score, then lined up against lap times.
"""
import os

# Once the models are cached, never call the Hub again: a cached model still
# triggers a HEAD check per file, and on flaky or captive wifi each one retries
# with backoff, so startup stalls for minutes instead of failing fast.
# On a fresh clone the cache is empty, so we must stay online to download.
from pathlib import Path as _P
_HUB = _P(os.environ.get("HF_HOME", _P.home() / ".cache/huggingface")) / "hub"
if (_HUB / "models--Systran--faster-whisper-large-v3").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import functools
import io
import json
import statistics
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# imported here, not inside _pipe: transformers' lazy loader is not thread-safe
# and a background warm-up would race the first request for it.
from transformers import pipeline
from faster_whisper import WhisperModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
CLIPS = ROOT / "clips"
CACHE_FILE = ROOT / "results.json"
SR = 16000

# Measured across 21 real team-radio clips (2025 Abu Dhabi): RMS ran
# 0.021-0.036, median 0.027 - nothing like the 0.06 of clean studio speech.
# Retune this if your clips come from a different source.
ENERGY_REF = 0.027
SILENT = 0.004          # below this there is no usable voice signal
STRESS_AT = 55          # p90 of stress across those clips
ELEVATED_AT = 40        # p70 - not calm any more, not yet stressed
TIRED_AT = 30           # p90 of fatigue

DATASET = "MikCil/f1-team-radio"

RACE = {"title": "2019 Brazilian Grand Prix", "dataset": DATASET,
        "race_id": "2019_Brazilian_Grand_Prix",
        "note": "Four drivers, one race. Real lap times, real team radio, joined on timestamp."}

# Finishing order that day, which is also the order they appear in the UI.
# (our code, the dataset's own driver_id, display name, what happened that day)
DRIVERS = [("VER", "MAXVER01", "Max Verstappen", "Won it"),
           ("SAI", "CARSAI01", "Carlos Sainz", "First podium, P3"),
           ("HAM", "LEWHAM01", "Lewis Hamilton", "Penalised after contact"),
           ("VET", "SEBVET01", "Sebastian Vettel", "Retired, lap 66")]

ASR_MODEL = "large-v3"          # Systran/faster-whisper-large-v3 on the Hub
VOICE_MODEL = "superb/hubert-large-superb-er"
TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

app = FastAPI(title="The Silent Co-Driver")

# Cache survives restarts, so a crash mid-demo costs nothing.
CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


# ---------------------------------------------------------------- models
# Loaded on first use, not at import, so the server starts instantly.
@functools.cache
def _pipe(task, model):
    return pipeline(task, model=model)


@functools.cache
def _asr():
    # int8 on CPU: same weights as whisper large-v3, ~7x faster than running it
    # through transformers, and it fits comfortably in RAM.
    return WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")


def _scores(rows):
    """Pipeline output [{label, score}, ...] -> {label: score} dict."""
    return {r["label"].lower(): float(r["score"]) for r in rows}


# ---------------------------------------------------------------- scoring
def stress_score(voice, text, energy):
    """Blend two emotion models + vocal energy into one 0-100 stress score.

    Vocal tone is weighted over word choice: a driver saying "I'm fine"
    through gritted teeth is not fine. Energy separates the two ways of
    not being okay - stress is loud, fatigue is flat.
    """
    # Agitation: anger in the voice, anger/fear/disgust/surprise in the words.
    v_hot = voice.get("ang", 0.0)
    t_hot = (text.get("anger", 0.0) + text.get("fear", 0.0)
             + text.get("disgust", 0.0) + 0.5 * text.get("surprise", 0.0))
    stress = 100 * (0.65 * v_hot + 0.35 * min(t_hot, 1.0))

    # Fatigue: flat, sad delivery - amplified when the voice is also quiet.
    v_flat = voice.get("sad", 0.0)
    t_flat = text.get("sadness", 0.0)
    quiet = max(0.0, 1.0 - energy / ENERGY_REF)
    fatigue = 100 * (0.60 * v_flat + 0.30 * t_flat) * (1 + 0.3 * quiet)

    if energy < SILENT:
        return "CALM", 0          # no signal is not the same as no stress
    # Thresholds are the 90th percentile of each axis measured over 63 real
    # team-radio clips - not round numbers picked by feel. Team radio is
    # flatter than ordinary speech, so a "feels right" cut labels nothing.
    if stress >= STRESS_AT:
        label = "STRESSED"
    elif fatigue >= TIRED_AT:
        label = "TIRED"
    elif stress >= ELEVATED_AT:
        # a 51 next to a green "CALM" badge reads as a broken gauge - name it
        label = "ELEVATED"
    else:
        label = "CALM"
    # The headline number is whichever pressure is actually present.
    return label, round(min(100.0, max(stress, fatigue if label == "TIRED" else 0)))


def _transcribe(y):
    """Transcript + per-word timings, degrading rather than failing.

    language is pinned: on short, noisy radio whisper's auto-detect drifts
    into Dutch/Thai/Welsh and returns confident nonsense. An upload must
    never 500 because the audio was junk, so failure yields empty text.
    """
    try:
        # vad_filter is not optional: without it whisper silently drops whole
        # speakers. On real team radio the driver's question was being cut and
        # only the engineer's reply kept - the exact thing this project reads.
        segments, _ = _asr().transcribe(y, language="en", word_timestamps=True,
                                        vad_filter=True)
        parts, words = [], []
        for seg in segments:                       # generator - drives the work
            parts.append(seg.text)
            for w in (seg.words or []):
                words.append({"w": w.word, "t": round(w.start, 2)})
        text = "".join(parts).strip()
        # large-v3 emits "... ..." for unintelligible radio rather than
        # hallucinating a sentence. Keep the honest version.
        if not any(c.isalnum() for c in text):
            return "", []
        return text, words
    except Exception:
        return "", []


def analyse_audio(y):
    """np.float32 mono @16k -> full analysis dict."""
    transcript, words = _transcribe(y)

    voice = _scores(_pipe("audio-classification", VOICE_MODEL)(y, top_k=None))
    text = _scores(
        _pipe("text-classification", TEXT_MODEL)(transcript or "ok", top_k=None)
    )
    energy = float(np.sqrt(np.mean(y**2)))

    label, score = stress_score(voice, text, energy)
    return {
        # no words does not mean no signal: the voice models still ran, and a
        # wordless shout is exactly the case this project exists for.
        "transcript": transcript or "(no intelligible words \u2014 tone only)",
        "words": words,
        "label": label,
        "score": score,
        "voice": voice,
        "text": text,
        "energy": round(energy, 4),
    }


# ---------------------------------------------------------------- lap data
def load_laps(driver):
    with open(ROOT / "laps.csv") as f:
        return [{"lap": int(r["lap"]), "time": float(r["lap_time_s"]),
                 "compound": r["compound"], "position": int(r["position"])}
                for r in csv.DictReader(f) if r["driver"] == driver]


def load_radio(driver):
    """One row per radio call, already joined to the lap it was sent on."""
    with open(ROOT / "radio.csv") as f:
        return [{"lap": int(r["lap"]), "clip": r["clip"], "utc": r["utc"],
                 # the dataset's own identifiers, shown in the UI as provenance
                 "driver_id": r["driver_id"], "dataset_id": r["dataset_id"],
                 "row": int(r["row"]),
                 # featured calls carry the demo; the rest hide behind a toggle
                 "featured": r.get("featured") == "1",
                 # named in the csv but no file on disk - the UI says so
                 "have": (CLIPS / r["clip"]).is_file()}
                for r in csv.DictReader(f) if r["driver"] == driver]


def lap_deltas(laps):
    """Each lap's gap to the stint median - the baseline a slow lap stands out from."""
    median = statistics.median(l["time"] for l in laps)
    for l in laps:
        l["delta"] = round(l["time"] - median, 3)
    return round(median, 3)


# ---------------------------------------------------------------- api
@app.get("/api/laps")
def api_laps():
    """Everything for every driver in one call - it is only a few hundred rows,
    and it means switching driver in the UI needs no round trip."""
    drivers = []
    for code, driver_id, name, note in DRIVERS:
        laps = load_laps(code)
        drivers.append({"code": code, "driver_id": driver_id, "name": name,
                        "note": note, "laps": laps, "median": lap_deltas(laps),
                        "radio": load_radio(code)})
    return {"drivers": drivers, "results": CACHE, "race": RACE}


@app.post("/api/analyse")
async def api_analyse(clip: str = Form(None), file: UploadFile = File(None)):
    if clip:
        if clip in CACHE:
            return CACHE[clip]
        path = CLIPS / Path(clip).name          # no traversal out of clips/
        if not path.is_file():
            raise HTTPException(404, f"no such clip: {clip}")
        source, key = str(path), clip
    elif file:
        source, key = io.BytesIO(await file.read()), f"upload:{file.filename}"
    else:
        raise HTTPException(400, "send a clip name or a file")

    try:
        y, _ = librosa.load(source, sr=SR, mono=True)
    except Exception as e:
        raise HTTPException(400, f"could not read audio: {e}")
    if len(y) < SR // 2:
        raise HTTPException(400, "clip is under half a second")

    result = analyse_audio(y) | {"clip": key}
    CACHE[key] = result
    CACHE_FILE.write_text(json.dumps(CACHE, indent=1))
    return result


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/chart.min.js")
def chartjs():
    return FileResponse(ROOT / "chart.min.js")


CLIPS.mkdir(exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS), name="clips")


# Load all three models now, on the main thread, before serving anything.
# Cached clips never touch the models, so without this the first live upload
# pays a 15s load in front of whoever is watching. Do NOT move this to a
# background thread: functools.cache is not atomic, and two threads building
# the same pipeline at once yields a half-initialised model that transcribes
# pure garbage.
analyse_audio(np.zeros(SR, dtype=np.float32))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
