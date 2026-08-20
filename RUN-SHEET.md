# Run sheet — read this at the venue

## Start it (two commands)

```bash
cd ~/Downloads/hk
.venv/bin/python app.py
```

Wait ~16 seconds for `Application startup complete`, then open
**http://127.0.0.1:8000** in Brave or Chrome.

**Start it before the judges reach your table.** Those 16 seconds are the three
models loading. Once it says ready, everything is instant.

## Internet

You do not need any. Verified with the network physically cut: the page, the
chart, all 27 cached calls, and a live upload of an unseen clip all work with
zero connectivity.

`app.py` forces `HF_HUB_OFFLINE=1`, so it reads models from
`~/.cache/huggingface` and never calls out. This is not just for a dead
connection — **bad wifi is worse than none**, because the Hub retries with
backoff and startup stalls for minutes instead of failing fast.

Only turn that off if you deliberately want to download a new model:

```bash
HF_HUB_OFFLINE=0 .venv/bin/python app.py
```

## If something goes wrong

| Symptom | Fix |
|---|---|
| `Address already in use` | It is already running. Just open the browser. |
| Need to kill it | `kill $(ss -ltnp \| grep -oP ':8000.*pid=\K[0-9]+' \| head -1)` |
| Page loads, clips do nothing | Hard-refresh with Ctrl+Shift+R |
| A clip shows "no file" | You are not in `~/Downloads/hk`. `cd` there and restart. |
| Upload spins forever | Models are still loading. Wait for startup, then retry. |
| Everything is broken | `git`-less fallback: the deck PDF has real screenshots. Present those. |

Laptop settings: disable sleep and notifications before you present.

## The 90-second walkthrough

1. **Say the problem first.** "A pit wall watches temperatures, fuel, sector
   deltas. The one thing nobody has time to process is the driver's own voice."
2. **Click lap 5.** Green, 8. Let the transcript light up word by word as the
   audio plays. "This is him fine."
3. **Click lap 42.** Red, 90. "Same driver, twenty-five minutes later."
4. **Point at the two breakdowns.** "Two models, not one. The voice heard anger
   at 86%. The words heard disgust at 70%. Independently."
5. **Click lap 57.** Amber. Then point at the chart. "We never told it about
   the safety car. That lap took 120 seconds instead of 73."
7. **Now upload `~/Downloads/clips/lewis-hamilton_TIRED.mp3`.** Amber, TIRED 48.
   "This is just ridiculous man." Say: **"The words sound angry. The voice is
   flat and quiet — vocal sadness 0.78, anger zero. So it calls it tired, not
   stressed. Those need different responses from a pit wall."**
8. **Stop talking.** Let them ask.

Step 6 matters for two reasons: it is the only way to show the amber TIRED
state — the loaded race has no tired call in it — and it is the clearest proof
that the tone is doing the work, not the words.

If they want one more, upload `valtteri-bottas_ELEVATED.mp3` — different driver,
different season, never seen by the app. The transcript comes out completely
clean and it still flags amber, from the voice alone.

## Questions they will ask

**"Isn't this just a sentiment model?"**
No. Sentiment reads words. Half our signal is the sound. Two clips prove it:
Bottas transcribes clean and still reads STRESSED, and Hamilton's "this is just
ridiculous" reads TIRED rather than angry, because the delivery is flat and
quiet. A word-only model gets both of those backwards.

**"Your legend shows three colours — where's the tired one?"**
Not in this race; none of Hamilton's 27 calls that day were flat enough. Upload
`lewis-hamilton_TIRED.mp3` and it appears. Say that plainly rather than hunting
for one in the list.

**"How do you know the lap numbers are right?"**
We do not align them by hand — it is a timestamp join. And it checks itself:
the call that says "outlap critical" lands on lap 44, an 89-second out-lap. The
safety car talk lands on laps that took two minutes instead of 73 seconds. Two
unrelated sources, and they agree.

**"Where's the AI? Isn't this just an API call?"**
Three models chained, plus the blend, which is ours. Tone is weighted 65/35
over words, and vocal energy separates stress (loud) from fatigue (flat). Both
thresholds are the 90th percentile measured over 63 real clips, not numbers we
picked.

**"Does it work on any race?"**
Yes — swap `laps.csv` and `radio.csv`. Nothing in `app.py` is specific to this
race. Upload a clip and see.

**"What would you do next?"**
Live radio instead of recorded clips, and per-driver baselines — some drivers
just sound angrier than others, so the honest version scores each driver
against himself.

**If you do not know an answer, say so.** "We did not test that" beats guessing;
judges probe hardest when they smell bluffing.

## Do not claim

- Not real-time — it is recorded clips, ~8 seconds per unseen upload.
- Not medically or psychologically validated — it is a signal, not a diagnosis.
- Whisper still mis-hears names sometimes. It is on screen; own it if it shows.
