import os
import glob
import subprocess
import sys
import pytest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
SNIPPET_FILES = sorted(glob.glob(os.path.join(CODE_DIR, "*.py")))

@pytest.mark.parametrize("filepath", SNIPPET_FILES, ids=lambda p: os.path.basename(p))
def test_snippet_execution(filepath):
    """
    Executes each Python code snippet in code/ and asserts clean exit code 0.
    """
    res = subprocess.run([sys.executable, filepath], capture_output=True, text=True)
    assert res.returncode == 0, f"Error executing {os.path.basename(filepath)}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
