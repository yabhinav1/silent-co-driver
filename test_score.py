"""Self-check for the blend. No models, no network - runs in a second.

    .venv/bin/python test_score.py
"""
from app import stress_score

NEUTRAL_TEXT = {"neutral": 0.9, "joy": 0.1}
NORMAL_ENERGY = 0.027       # measured median of real team radio

angry = stress_score({"ang": 0.9, "neu": 0.1},
                     {"anger": 0.8, "neutral": 0.2}, NORMAL_ENERGY)
calm = stress_score({"neu": 0.9, "hap": 0.1}, NEUTRAL_TEXT, NORMAL_ENERGY)
tired = stress_score({"sad": 0.8, "neu": 0.2},
                     {"sadness": 0.7, "neutral": 0.3}, 0.010)
gritted = stress_score({"ang": 0.85, "neu": 0.15}, NEUTRAL_TEXT, NORMAL_ENERGY)

assert angry[0] == "STRESSED", angry
assert calm[0] == "CALM", calm
assert tired[0] == "TIRED", tired

# The whole point: an angry clip must outscore a calm one by a wide margin.
assert angry[1] - calm[1] > 40, (angry, calm)

# Tone beats words - "I'm fine" said furiously still reads as stressed.
assert gritted[0] == "STRESSED", gritted

# Quiet + flat is tired, the same delivery at normal volume is less so.
loud_flat = stress_score({"sad": 0.8, "neu": 0.2},
                         {"sadness": 0.7, "neutral": 0.3}, 0.045)
assert tired[1] > loud_flat[1], (tired, loud_flat)

# A near-silent clip must not read as fatigue - the emotion models return
# confident nonsense on a clip with no voice in it.
silent = stress_score({"sad": 0.76, "neu": 0.2}, {"sadness": 0.6}, 0.0001)
assert silent == ("CALM", 0), silent

# Scores stay in range whatever the models return.
for s in (angry, calm, tired, gritted, loud_flat, silent):
    assert 0 <= s[1] <= 100, s

print(f"ok  angry={angry}  calm={calm}  tired={tired}  gritted={gritted}")
