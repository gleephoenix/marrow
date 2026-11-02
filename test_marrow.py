import json, pathlib, tempfile, unittest
import marrow as m

EVENTS = [
    {"type": "move", "t": 110, "x": 0.1, "y": 0.1},
    {"type": "move", "t": 160, "x": 0.3, "y": 0.2},
    {"type": "move", "t": 210, "x": 0.5, "y": 0.4},
    {"type": "down", "t": 260, "x": 0.5, "y": 0.4},
    {"type": "up", "t": 300, "x": 0.5, "y": 0.4},
    {"type": "text", "t": 700, "source": "typed", "op": "insert", "chars": 3},
    {"type": "text", "t": 900, "source": "typed", "op": "replace", "chars": 1},
    {"type": "text", "t": 1100, "source": "typed", "op": "insert", "chars": 2},
]

class FeatureTests(unittest.TestCase):
    def test_features_have_all_keys(self):
        f = m.features(EVENTS)
        for k in m.FEATURES:
            self.assertIn(k, f)
        self.assertGreater(f["log_duration"], 0)
        self.assertGreater(f["log_iki"], 0)

    def test_no_pointer_events_still_yields_text_features(self):
        f = m.features([e for e in EVENTS if e["type"] == "text"])
        self.assertIn("log_iki", f)

class DecisionTests(unittest.TestCase):
    def setUp(self):
        rows = [m.features(EVENTS) for _ in range(6)]
        for r in rows:
            r["log_duration"] += 0.01
        self.prof = m.robust_profile(rows)

    def test_same_session_accepted(self):
        ok, s = m.decide(self.prof, m.features(EVENTS))
        self.assertTrue(ok, s)

    def test_far_session_rejected(self):
        far = m.features(EVENTS)
        far["log_iki"] += 3.0
        far["log_duration"] += 2.0
        ok, s = m.decide(self.prof, far)
        self.assertFalse(ok, s)

    def test_threshold_is_fixed_constant(self):
        self.assertEqual(m.DECISION_THRESHOLD, 2.0)

class GeneratorTests(unittest.TestCase):
    def test_dataset_is_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            m.make_dataset(pathlib.Path(a))
            m.make_dataset(pathlib.Path(b))
            for f in sorted(pathlib.Path(a).glob("*.jsonl")):
                self.assertEqual(f.read_bytes(), (pathlib.Path(b) / f.name).read_bytes())

    def test_manifest_has_four_styles(self):
        with tempfile.TemporaryDirectory() as d:
            man = m.make_dataset(pathlib.Path(d))
            self.assertEqual(len(man), 4)
            self.assertGreaterEqual(sum(x["sessions"] for x in man), 600)

if __name__ == "__main__":
    unittest.main()
