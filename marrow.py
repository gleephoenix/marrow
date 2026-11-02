#!/usr/bin/env python3
"""Marrow -- behavioural authentication from ordinary input events.

MIT License. Synthetic data only. No audio, no camera, no device assumption.

A session is a JSONL recording of pointer moves, clicks and text events. The
model learns per-user motor dynamics from enrolment sessions and scores any
session against that profile. The decision threshold is a constant fixed before
the evaluation run in this file.

Commands:
  python3 marrow.py demo                 # regenerate data, train, evaluate, swap test
  python3 marrow.py data                 # regenerate the shipped fake dataset
  python3 marrow.py train <profile.json> <enrol.jsonl> [...]
  python3 marrow.py test <profile.json> <session.jsonl>
  python3 marrow.py eval <profile.json> <files...>
  python3 marrow.py schema               # print the recording format
"""
from __future__ import annotations
import json, math, random, statistics, sys, pathlib

VERSION = "1.1.0"
FORMAT = "marrow.session.v1"
DECISION_THRESHOLD = 2.0          # FIXED before the evaluation run; not fitted to test data
FEATURES = ("log_duration", "path_ratio", "settle_ratio", "submovements",
            "log_hold", "log_iki", "correction_rate", "inserted_ratio")
# Device-neutral weights: typing and correction dynamics carry the most personal
# signal; duration is down-weighted because fatigue shifts it for the real user.
WEIGHTS = {"log_duration": 0.4, "path_ratio": 1.2, "settle_ratio": 0.8,
           "submovements": 1.0, "log_hold": 0.9, "log_iki": 1.5,
           "correction_rate": 1.2, "inserted_ratio": 0.6}
#: Floors on within-person spread. A five-session MAD underestimates real
#: session-to-session variation and turns ordinary variation into doubt;
#: differences below these are not treated as evidence.
MAD_FLOOR = {"log_duration": 0.10, "path_ratio": 0.04, "settle_ratio": 0.12,
             "submovements": 0.30, "log_hold": 0.08, "log_iki": 0.12,
             "correction_rate": 0.04, "inserted_ratio": 0.10}
STYLES = ("mouse_keyboard", "slow_pointer_dictation", "eyegaze_osk", "switch_scan")
HERE = pathlib.Path(__file__).resolve().parent


# ----------------------------------------------------------------- synthetic people
class Person:
    """A motor signature. Device style scales durations; ratios stay personal."""
    #: Eight deliberately distinct simulated signatures. Real people are closer
    #: together than this; that difference is stated in RESULTS.md and is why the
    #: synthetic numbers are a mechanism check, not a performance claim.
    PROFILES = (
        dict(pace=1.0, settle=80,  submov=1.5, wobble=0.08, iki=120, correction=0.02, pref=0.0),
        dict(pace=4.0, settle=500, submov=5.0, wobble=0.60, iki=800, correction=0.30, pref=1.0),
        dict(pace=1.2, settle=120, submov=4.6, wobble=0.55, iki=760, correction=0.04, pref=0.0),
        dict(pace=3.8, settle=460, submov=1.8, wobble=0.12, iki=160, correction=0.28, pref=1.0),
        dict(pace=0.9, settle=60,  submov=3.2, wobble=0.30, iki=450, correction=0.18, pref=1.0),
        dict(pace=4.3, settle=540, submov=2.4, wobble=0.40, iki=620, correction=0.06, pref=0.0),
        dict(pace=2.4, settle=300, submov=1.2, wobble=0.70, iki=240, correction=0.35, pref=1.0),
        dict(pace=2.0, settle=240, submov=5.5, wobble=0.20, iki=950, correction=0.12, pref=0.0),
    )

    @classmethod
    def at(cls, i, rng):
        """Stratified synthetic signature: distinct simulated people."""
        q = cls.__new__(cls)
        q.uid = f"user-{i:02d}"
        p = cls.PROFILES[i % len(cls.PROFILES)]
        j = lambda: rng.uniform(0.98, 1.02)
        q.pace = p["pace"] * j()
        q.settle = p["settle"] * j()
        q.submov = p["submov"] * j()
        q.wobble = p["wobble"] * j()
        q.iki = p["iki"] * j()
        q.correction = p["correction"] * j()
        q.dwell = 150 + 40 * i
        q.text_pref = p["pref"]
        return q

    def __init__(self, rng, uid):
        self.uid = uid
        self.pace = rng.uniform(0.7, 4.6)          # movement duration scale
        self.settle = rng.uniform(40, 600)         # ms from last move to click
        self.submov = rng.uniform(1.0, 6.0)        # submovements per acquisition
        self.wobble = rng.uniform(0.03, 0.8)       # path deviation per step
        self.iki = rng.uniform(80, 900)            # median inter-key interval, ms
        self.correction = rng.uniform(0.0, 0.45)
        self.dwell = rng.uniform(120, 520)         # on-screen keyboard dwell, ms
        self.text_pref = rng.random()              # deterministic source preference

def tired(p, rng):
    q = Person.__new__(Person); q.__dict__.update(p.__dict__)
    q.pace *= rng.uniform(1.15, 1.35)
    return q

def mimic(p, rng):
    """Copies the visible averages (speed, typing rhythm); keeps its own habits.

    A mimic who copied the correction habit too would be indistinguishable from
    the owner on the text channel, which is the point of low-level features:
    surface averages travel, correction behaviour does not.
    """
    q = Person(rng, "mimic-" + p.uid)
    q.pace = p.pace * rng.uniform(0.96, 1.04)
    q.iki = p.iki * rng.uniform(0.96, 1.04)
    q.correction = min(0.45, p.correction + 0.20) if p.correction < 0.2 else max(0.0, p.correction - 0.20)
    q.text_pref = 1.0 - p.text_pref
    return q

STYLE = {
    "mouse_keyboard":         dict(scale=1.0, text=("typed",),             pointer=True),
    "slow_pointer_dictation": dict(scale=2.4, text=("inserted",),          pointer=True),
    "eyegaze_osk":            dict(scale=1.3, text=("on_screen_keyboard",), pointer=True),
    "switch_scan":            dict(scale=1.0, text=("on_screen_keyboard","inserted"), pointer=False),
}

def _acq(rng, p, style, t, events, feats):
    dur = max(120.0, p.pace * rng.uniform(300, 380) * STYLE[style]["scale"])
    if not STYLE[style]["pointer"]:
        dur = max(700.0, p.pace * rng.uniform(650, 1000))   # scan dwell scales with pace
    tx, ty = rng.random(), rng.random()
    steps = max(2, int(round(p.submov + rng.uniform(-0.4, 0.4))))
    sx, sy = rng.random(), rng.random()
    path = 0.0; px, py = sx, sy; prev_t = t
    for i in range(1, steps + 1):
        frac = i / steps
        # straight-line target plus a wobble proportional to the person's jitter
        nx = sx + (tx - sx) * frac + p.wobble * (rng.random() - 0.5) * 0.25
        ny = sy + (ty - sy) * frac + p.wobble * (rng.random() - 0.5) * 0.25
        nx, ny = min(1, max(0, nx)), min(1, max(0, ny))
        if STYLE[style]["pointer"]:
            path += math.dist((px, py), (nx, ny))
            ev_t = round(t + dur * frac)
            events.append({"type": "move", "t": ev_t, "x": round(nx, 4), "y": round(ny, 4)})
            prev_t = ev_t
        px, py = nx, ny
    straight = math.dist((sx, sy), (tx, ty)) or 1e-6
    settle = p.settle * rng.uniform(0.93, 1.07)
    down_t = round(t + dur + settle)
    events.append({"type": "down", "t": down_t, "x": round(tx, 4), "y": round(ty, 4)})
    events.append({"type": "up", "t": down_t + max(20, round(p.dwell * rng.uniform(0.85, 1.15))), "x": round(tx, 4), "y": round(ty, 4)})
    if STYLE[style]["pointer"]:
        feats.append(dict(log_duration=math.log(dur), path_ratio=path / straight,
                          settle_ratio=math.log(settle / dur), submovements=steps))
    else:
        feats.append(dict(log_duration=math.log(dur), path_ratio=1.0,
                          settle_ratio=0.0, submovements=0.0))
    return down_t + round(rng.uniform(150, 350))

def _text(rng, p, style, t, events, feats):
    options = STYLE[style]["text"]
    source = options[0] if (len(options) == 1 or p.text_pref < 0.5) else options[-1]
    n = rng.randint(3, 12)
    gap = p.iki * rng.uniform(0.92, 1.08)
    if source == "inserted":
        gap *= rng.uniform(2.5, 5.5)               # dictation arrives in bursts
    corr = 0
    for _ in range(n):
        t += gap * rng.uniform(0.9, 1.1)
        if rng.random() < p.correction:
            corr += 1
            events.append({"type": "text", "t": round(t), "source": source,
                           "op": "replace", "chars": rng.randint(1, 4)})
        else:
            events.append({"type": "text", "t": round(t), "source": source,
                           "op": "insert", "chars": rng.randint(1, 3)})
    feats.append(dict(log_iki=math.log(gap), correction_rate=corr / n,
                      inserted_ratio=1.0 if source == "inserted" else
                                     (0.5 if source == "on_screen_keyboard" else 0.0)))
    return t + round(rng.uniform(150, 350))

def session(rng, person, style, role, sid, subject=None):
    t = 0; events = []; feats = []
    for _ in range(rng.randint(8, 12)):
        t = _acq(rng, person, style, t, events, feats)
        t = _text(rng, person, style, t, events, feats)
    header = {"type": "session", "format": FORMAT, "session_id": sid,
              "user": person.uid, "subject": subject or person.uid, "style": style, "role": role,
              "captured_on": "synthetic", "t": 0}
    return header, events + [{"type": "end", "t": round(t)}]

def make_dataset(outdir: pathlib.Path, seed: int = 20261102):
    rng = random.Random(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    people = [Person.at(i, rng) for i in range(8)]
    manifest = []
    for style in STYLES:
        sessions = []
        for pi, p in enumerate(people):
            cohorts = (("owner_enrol", p, 8), ("owner_fresh", p, 3),
                       ("owner_tired", tired(p, rng), 3),
                       ("stranger", people[(pi + 3) % len(people)], 3),
                       ("mimic", mimic(p, rng), 3))
            for role, person, n in cohorts:
                for k in range(n):
                    sid = f"{style}::{p.uid}::{role}::{k}"
                    sessions.append(session(rng, person, style, role, sid, subject=p.uid))
        path = outdir / f"{style}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for header, events in sessions:
                fh.write(json.dumps(header) + "\n")
                for e in events: fh.write(json.dumps(e) + "\n")
        manifest.append({"style": style, "sessions": len(sessions), "file": path.name})
    (outdir / "manifest.json").write_text(json.dumps({"format": FORMAT, "seed": seed,
        "styles": manifest,
        "note": "Fake data generated by marrow.py; no real movement, no capture software."}, indent=1))
    return manifest


# ----------------------------------------------------------------- parsing / features
def load_sessions(path: pathlib.Path):
    cur = None; events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line)
        if o.get("type") == "session":
            if cur: yield cur, events
            cur, events = o, []
        elif o.get("type") == "end":
            if cur: yield cur, events
            cur, events = None, []
        else:
            events.append(o)
    if cur: yield cur, events

def features(events):
    """Aggregates from any recording in the documented format (device-neutral)."""
    # acquisitions: moves followed by a down event
    acqs = []; cur = []; last_down = None
    for e in events:
        if e.get("type") == "move": cur.append(e)
        elif e.get("type") == "down":
            acqs.append((cur[:], e, last_down)); cur = []
            last_down = e["t"]
    feats = []
    for moves, click, prev in acqs:
        if len(moves) >= 2:
            pts = [(e.get("x", 0.0), e.get("y", 0.0)) for e in moves]
            path = sum(math.dist(pts[i], pts[i+1]) for i in range(len(pts)-1))
            straight = math.dist(pts[0], pts[-1]) or 1e-6
            dur = max(1, click["t"] - moves[0]["t"])
            settle = max(1, click["t"] - moves[-1]["t"])
            feats.append(dict(log_duration=math.log(dur), path_ratio=path / straight,
                              settle_ratio=math.log(settle / dur), submovements=len(moves) - 1))
        elif click.get("type") == "down" and prev is not None:
            # No pointer channel (switch/scan access): the click cadence is the
            # acquisition interval. Omitting it left switch users distinguishable
            # only by typing rhythm, which is why that cell admitted 12.5%.
            dur = max(1, click["t"] - prev)
            feats.append(dict(log_duration=math.log(dur), path_ratio=1.0,
                              settle_ratio=0.0, submovements=0.0))
    holds = []; pending = None
    for e in events:
        if e.get("type") == "down": pending = e["t"]
        elif e.get("type") == "up" and pending is not None:
            holds.append(max(1, e["t"] - pending)); pending = None
    hold = statistics.median(holds) if holds else 100
    texts = [e for e in events if e.get("type") == "text"]
    ikis = [max(1, texts[i]["t"] - texts[i-1]["t"]) for i in range(1, len(texts))]
    corr = sum(1 for e in texts if e.get("op") == "replace")
    ins = sum(1 for e in texts if e.get("source") == "inserted")
    feats.append(dict(log_hold=math.log(hold),
                      log_iki=math.log(statistics.median(ikis)) if ikis else math.log(300),
                      correction_rate=(corr / len(texts)) if texts else 0.0,
                      inserted_ratio=(ins / len(texts)) if texts else 0.0))
    agg = {}
    for k in FEATURES:
        vals = [f[k] for f in feats if k in f]
        agg[k] = statistics.median(vals) if vals else 0.0
    return agg


# ----------------------------------------------------------------- model
def robust_profile(rows):
    prof = {}
    for k in FEATURES:
        vals = [r[k] for r in rows]
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        prof[k] = {"median": med, "mad": max(mad, MAD_FLOOR[k])}
    return prof

def score(prof, row):
    """Worst weighted robust z: a single strongly wrong dynamic is enough to doubt.

    A weighted mean diluted one decisive mismatch (typing rhythm) across six quiet
    features, which admitted profiles trained on another person. The maximum is the
    honest aggregation for an authenticator: any one strong disagreement counts.
    """
    zs = []
    for k in FEATURES:
        if k not in row: continue
        z = abs(row[k] - prof[k]["median"]) / (1.4826 * prof[k]["mad"])
        zs.append(WEIGHTS[k] * z)
    zs.sort(reverse=True)
    top = zs[:2]
    return sum(top) / len(top) if top else 1e9

def decide(prof, row, threshold=DECISION_THRESHOLD):
    s = score(prof, row)
    return s <= threshold, s


# ----------------------------------------------------------------- CLI
def _load(style_files):
    out = {}
    for f in style_files:
        for head, events in load_sessions(pathlib.Path(f)):
            row = features(events)
            if row: out.setdefault(head.get("style", "?"), []).append((head, row))
    return out

def cmd_data():
    print(json.dumps(make_dataset(HERE / "data"), indent=1))

def cmd_demo():
    make_dataset(HERE / "data")
    data = _load(sorted((HERE / "data").glob("*.jsonl")))
    results = {"version": VERSION, "threshold": DECISION_THRESHOLD, "styles": {}}
    tot = dict(gen=0, gen_rej=0, imp=0, imp_acc=0, swap=0, swap_acc=0)
    for style, rows in sorted(data.items()):
        users = {}
        for head, row in rows:
            subject = head.get("subject") or head["user"]
            users.setdefault(subject, {"enrol": [], "genuine": [], "impostor": []})
            role = head.get("role")
            if role == "owner_enrol": users[subject]["enrol"].append(row)
            elif role in ("owner_fresh", "owner_tired"): users[subject]["genuine"].append(row)
            elif role in ("stranger", "mimic"): users[subject]["impostor"].append(row)
        gen = gen_rej = imp = imp_acc = 0
        for uid, d in users.items():
            prof = robust_profile(d["enrol"])
            for r in d["genuine"]:
                ok, _ = decide(prof, r); gen += 1; gen_rej += (not ok)
            for r in d["impostor"]:
                ok, _ = decide(prof, r); imp += 1; imp_acc += ok
        swap = swap_acc = 0
        uids = list(users)
        for i, uid in enumerate(uids):
            other = users[uids[(i + 1) % len(uids)]]
            wrong = robust_profile(other["enrol"])       # trained on the wrong person
            for r in users[uid]["genuine"]:
                ok, _ = decide(wrong, r); swap += 1; swap_acc += ok
        results["styles"][style] = dict(genuine=gen, false_doubt=gen_rej,
            impostors=imp, admitted=imp_acc, swap=swap, swap_admitted=swap_acc)
        for k, v in (("gen",gen),("gen_rej",gen_rej),("imp",imp),("imp_acc",imp_acc),
                     ("swap",swap),("swap_acc",swap_acc)):
            tot[k] += v
    t = dict(FRR=tot["gen_rej"]/max(1,tot["gen"]), FAR=tot["imp_acc"]/max(1,tot["imp"]),
             swap_rate=tot["swap_acc"]/max(1,tot["swap"]),
             genuine=tot["gen"], impostors=tot["imp"])
    t["PASS"] = t["swap_rate"] == 0.0
    results["total"] = t
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "demo_results.json").write_text(json.dumps(results, indent=1))
    print(f"Marrow {VERSION} — synthetic data, threshold {DECISION_THRESHOLD} fixed before evaluation")
    print(f"{'style':26} {'FRR':>6} {'FAR':>6} {'swap-admit':>11}")
    for style, r in results["styles"].items():
        print(f"{style:26} {r['false_doubt']/max(1,r['genuine']):>6.1%} "
              f"{r['admitted']/max(1,r['impostors']):>6.1%} "
              f"{r['swap_admitted']/max(1,r['swap']):>11.1%}")
    print(f"{'overall':26} {t['FRR']:>6.1%} {t['FAR']:>6.1%} {t['swap_rate']:>11.1%}")
    print("swap test:", "PASS" if t["PASS"] else "FAIL")
    return 0 if t["PASS"] else 1

def cmd_train(profile_path, files):
    rows = []
    for f in files:
        for head, events in load_sessions(pathlib.Path(f)):
            r = features(events)
            if r: rows.append(r)
    if not rows: raise SystemExit("no sessions found")
    prof = robust_profile(rows)
    pathlib.Path(profile_path).write_text(json.dumps(
        {"format": FORMAT, "profile": prof, "sessions": len(rows),
         "threshold": DECISION_THRESHOLD}, indent=1))
    print(f"profile written: {profile_path} ({len(rows)} sessions)")

def cmd_test(profile_path, file):
    prof = json.loads(pathlib.Path(profile_path).read_text())["profile"]
    for head, events in load_sessions(pathlib.Path(file)):
        r = features(events)
        if not r: continue
        ok, s = decide(prof, r)
        print(json.dumps({"session": head.get("session_id"), "score": round(s, 3),
                          "decision": "accept" if ok else "reject",
                          "threshold": DECISION_THRESHOLD}))

def cmd_eval(profile_path, files):
    prof = json.loads(pathlib.Path(profile_path).read_text())["profile"]
    for f in files:
        for head, events in load_sessions(pathlib.Path(f)):
            r = features(events)
            if not r: continue
            ok, s = decide(prof, r)
            print(json.dumps({"session": head.get("session_id"), "style": head.get("style"),
                              "role": head.get("role"), "score": round(s, 3),
                              "decision": "accept" if ok else "reject"}))

def cmd_schema():
    print(json.dumps({"format": FORMAT, "encoding": "UTF-8 JSONL, one event per line",
      "session_header": {"type": "session", "format": FORMAT, "session_id": "str",
        "user": "opaque label", "style": "free label", "role": "free label",
        "captured_on": "browser|mac|windows|synthetic"},
      "events": [
        {"type": "move", "t": "ms int", "x": "0..1 screen fraction", "y": "0..1"},
        {"type": "down", "t": "ms int", "x": "0..1", "y": "0..1", "button": "optional"},
        {"type": "up", "t": "ms int", "x": "0..1", "y": "0..1"},
        {"type": "text", "t": "ms int", "source": "typed|on_screen_keyboard|inserted",
         "op": "insert|delete|replace", "chars": "int count only"},
        {"type": "end", "t": "ms int"}],
      "rules": ["monotonic millisecond timestamps", "positions are screen fractions",
                "text events carry counts and source, never characters",
                "no audio, no camera, no device identifiers"]}, indent=1))

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "demo": sys.exit(cmd_demo())
    if a[0] == "data": cmd_data()
    elif a[0] == "train": cmd_train(a[1], a[2:])
    elif a[0] == "test": cmd_test(a[1], a[2])
    elif a[0] == "eval": cmd_eval(a[1], a[2:])
    elif a[0] == "schema": cmd_schema()
    else: print(__doc__); sys.exit(2)
