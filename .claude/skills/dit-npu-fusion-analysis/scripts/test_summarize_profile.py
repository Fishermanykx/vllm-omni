"""Portable behavioral checks using synthetic data, not measured NPU traces."""

import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("summarize_profile.py")


class ProfileSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trace = self.root / "trace.json"
        self.events = [
            {"ph": "M", "pid": 4, "name": "process_name", "args": {"name": "NPU 0"}},
            {"ph": "X", "cat": "device", "pid": 4, "tid": 1, "name": "RmsNorm", "ts": 0, "dur": 2000},
            {"ph": "X", "cat": "device", "pid": 4, "tid": 2, "name": "RmsNorm", "ts": 1000, "dur": 2000},
            {"ph": "X", "cat": "cpu_op", "pid": 4, "tid": 1, "name": "host", "ts": 0, "dur": 9000},
            {"ph": "X", "cat": "device", "pid": 5, "tid": 1, "name": "other-device", "ts": 0, "dur": 8000},
            {"ph": "B", "cat": "device", "pid": 4, "tid": 1, "name": "unpaired", "ts": 3000},
        ]
        self.trace.write_text(json.dumps({"traceEvents": self.events}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, success=True):
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                                text=True, encoding="utf-8", capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_overlap_and_host_device_selection(self):
        out = self.run_cli("trace", self.trace, "--category", "device", "--pid", "4", "--unit", "us")
        self.assertEqual(out["selected_count"], 2)
        self.assertEqual(out["sum_ms"], 4)
        self.assertEqual(out["selected_intervals_union_ms"], 3)
        self.assertEqual(out["observed_span_ms"], 3)
        self.assertEqual(out["ignored_selected_phases"], {"B": 1})
        self.assertEqual(out["top"][0]["samples"][0]["event_index"], 1)

    def test_gzip_list_input_and_tid(self):
        compressed = self.root / "trace.json.gz"
        with gzip.open(compressed, "wt", encoding="utf-8") as stream:
            json.dump(self.events, stream)
        out = self.run_cli("trace", compressed, "--category", "device", "--pid", "4", "--tid", "2", "--unit", "us")
        self.assertEqual(out["selected_count"], 1)
        self.assertEqual(out["sum_ms"], 2)

    def test_inventory_and_missing_selection(self):
        out = self.run_cli("trace", self.trace, "--inventory")
        self.assertEqual(out["metadata"][0]["args"]["name"], "NPU 0")
        self.assertIn("--category", self.run_cli("trace", self.trace, "--unit", "us", success=False))
        self.assertIn("--unit", self.run_cli("trace", self.trace, "--category", "device", "--pid", "4", success=False))

    def test_empty_selection_is_error(self):
        error = self.run_cli("trace", self.trace, "--category", "missing", "--pid", "4", "--unit", "us", success=False)
        self.assertIn("No matching", error)

    def test_csv_bom_units_shapes_and_output(self):
        source = self.root / "kernels.csv"
        source.write_text('Name,Duration(ns),Shape\n融合,2000000,"[1, 128]"\n融合,3000000,"[1, 128]"\n', encoding="utf-8-sig")
        self.assertEqual(self.run_cli("csv", source, "--inventory")["columns"][0], "Name")
        output = self.root / "summary.json"
        result = subprocess.run([sys.executable, str(SCRIPT), "csv", str(source), "--name-column", "Name",
                                 "--duration-column", "Duration(ns)", "--shape-column", "Shape", "--unit", "ns",
                                 "--output", str(output)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(out["sum_ms"], 5)
        self.assertEqual(out["top"][0]["mean_ms"], 2.5)
        self.assertEqual(out["top"][0]["samples"][0]["csv_line_end"], 2)
        self.assertNotIn("selected_intervals_union_ms", out)

    def test_invalid_duration_and_columns(self):
        source = self.root / "bad.csv"
        source.write_text("Name,Duration\nbad,NaN\n", encoding="utf-8")
        self.assertIn("finite", self.run_cli("csv", source, "--name-column", "Name", "--duration-column", "Duration", "--unit", "ms", success=False))
        self.assertIn("exact", self.run_cli("csv", source, "--name-column", "missing", "--duration-column", "Duration", "--unit", "ms", success=False))


if __name__ == "__main__":
    unittest.main()
