"""Compile CPU protocol fixtures against this experiment's actual headers.

Small logical plans here test state transitions, never B1 performance.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


class ExecProtocolTest(unittest.TestCase):
    def compile_run(self, source, shared=False):
        with tempfile.TemporaryDirectory(prefix="only-scheduler-test-") as directory:
            output = Path(directory) / "test"
            flags = ["-DPTO_FDWIC_SHARED_MAP=1", "-DPA_BUILD_PERF_CLOCK=1"] if shared else []
            command = ["g++", "-O2", "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                       *flags, "-I", str(ROOT / "common"), "-I", str(ROOT),
                       "-x", "c++", "-", "-o", str(output)]
            build = subprocess.run(command, input=source, text=True, capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(output)], text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("[FAIL]", result.stdout + result.stderr)
            return result.stdout

    def test_fanin_reference_equivalence(self):
        result = self.compile_run((ROOT / "test_fanin_snapshot.cpp").read_text())
        self.assertIn("4080 reference-equivalence checks", result)

    def test_shared_protocol(self):
        source = (FIXTURES / "test_shared_exec_protocol.cpp").read_text()
        self.assertIn("[PASS] cross-core shared execution protocol", self.compile_run(source))

    def test_execute_scan_and_route_cache_lifetime(self):
        source = (FIXTURES / "test_cross_core_exec_scan.cpp").read_text()
        result = self.compile_run(source, shared=True)
        self.assertIn("[PASS] owner-route-cache-lifetime", result)
        self.assertIn("[PASS] cross-core Execute ticket and drain closure", result)


if __name__ == "__main__":
    unittest.main()
