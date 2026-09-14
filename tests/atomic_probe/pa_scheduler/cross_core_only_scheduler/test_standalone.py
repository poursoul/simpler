"""Exercise the tools with no parent checkout or sibling experiment available."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class StandaloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="only-scheduler-relocation-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name) / "detached experiment"
        shutil.copytree(
            ROOT, cls.root,
            ignore=shutil.ignore_patterns("build", ".venv", "__pycache__", "*.pyc", "test_record"),
        )

    def run_checked(self, *arguments):
        result = subprocess.run(
            [sys.executable, "-I", "-B", *map(str, arguments)],
            cwd=self.temporary.name, text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_converter_without_parent_repository(self):
        output = self.run_checked(self.root / "swimlane_converter.py", "--help")
        self.assertIn("usage:", output)

    def test_protocol_without_sibling_directory(self):
        output = self.run_checked(
            "-m", "unittest", "discover", "-s", self.root,
            "-p", "test_exec_protocol.py", "-v",
        )
        self.assertIn("Ran 3 tests", output)
        self.assertIn("OK", output)


if __name__ == "__main__":
    unittest.main()
