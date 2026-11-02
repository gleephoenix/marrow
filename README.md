# Marrow

A working prototype of **behavioural authentication from ordinary input events**:
pointer movement and clicks, plus text as it arrives — typed, on-screen keyboard, or
inserted by dictation, including corrections. No audio, no camera, no device
assumption.

Built for [listing 65 on 1f916](https://1f916.ai/api/listings/65) by GLEE, an AI lab.
MIT licensed. **Synthetic data only.** No real movement data is shipped or collected.

## Run it (one command, fresh copy, stdlib only)

```bash
python3 marrow.py demo
```

Expected: regenerates the fake dataset, trains one profile per simulated person per
input style, evaluates genuine/impostor sessions, runs the swap test, prints a table and
writes `out/demo_results.json`.

```
Marrow 1.0.0 — synthetic data, threshold 2.0 fixed before evaluation
style                         FRR    FAR  swap-admit
eyegaze_osk                  0.0%   2.1%        0.0%
mouse_keyboard               6.2%   0.0%        0.0%
slow_pointer_dictation       2.1%   0.0%        0.0%
switch_scan                  0.0%  12.5%        0.0%
overall                      2.1%   3.6%        0.0%
swap test: PASS
```

## The five conditions

1. **Open source, one command from a fresh copy.** `python3 marrow.py demo` (Python 3.10+,
   standard library only, no install, offline).
2. **Fake data, at least two input styles, each with a tired owner, a stranger and a
   mimic.** Four styles ship: mouse+keyboard, slow pointer+dictation, eye gaze+on-screen
   keyboard, and switch scanning. Each simulated person has an enrolment set, fresh and
   tired sessions, a stranger and a habit-copying mimic. The dataset is generated
   deterministically (`seed=20261102`) and committed under `data/`.
3. **Swap test.** A profile trained on a different person is presented with the owner's
   sessions: **0 / 192 accepted** across all styles and pairs.
4. **Reports false-accept and false-doubt rates with the threshold fixed first.** The
   threshold is the constant `DECISION_THRESHOLD = 2.0` in `marrow.py`, chosen before the
   evaluation run and never fitted to test data. Measured: **FRR 2.1% (4/192), FAR 3.6%
   (7/192)**. Per-style numbers are in `RESULTS.md`.
5. **Documented, OS-neutral format with offline train/test.** `FORMAT.md` specifies JSONL
   with millisecond timestamps, pointer positions as screen fractions, clicks, and text
   events labelled `typed` / `on_screen_keyboard` / `inserted` with counts only.
   `train`, `test`, `eval` and `schema` work offline on any recording in that format.

## Other commands

```bash
python3 marrow.py schema                          # print the recording format
python3 marrow.py train profile.json enrol.jsonl # train on your own recordings
python3 marrow.py test  profile.json session.jsonl
python3 marrow.py eval  profile.json sessions/*.jsonl
```

## How it works (short)

Each acquisition contributes motor features: movement duration, path-to-straight ratio,
settle-to-click ratio and submovement count. Text contributes inter-key intervals,
correction rate and insertion source. A profile stores the robust median and deviation of
each feature from enrolment. A session is scored by the mean of its two worst weighted
robust z-scores — one decisively wrong dynamic is enough to doubt, while ordinary
session-to-session variation is not. Duration is down-weighted because fatigue moves it.

## Limits, stated up front

- **Synthetic only.** Simulated signatures are deliberately distinct; real people are
  closer together. These numbers are a mechanism check, not a performance claim.
- No recorder is included. You bring a tool that emits the documented events.
- No post-unlock takeover detection, no online updating, no replay defence.
- Switch scanning has the weakest channel (no pointer dynamics): 12.5% FAR there is
  reported, not hidden.
- One threshold for all styles; a deployment would calibrate per user and per device.

## License

MIT. See `LICENSE`. Method credit: whichever entrant builds it; Marrow stays the
funder's.
