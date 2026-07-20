import subprocess
import sys


def test_preprocessing_does_not_eagerly_import_d4rl():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import diffuser.datasets.preprocessing; "
                "assert 'd4rl' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
