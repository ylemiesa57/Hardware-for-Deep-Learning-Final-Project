"""Regression test for deterministic series/stack ordering in plots.py.

bar_side_by_side() and bar_stacked() used to build their series/stack key
list via `list(set(...))`. Python randomizes string hash order per process
(PYTHONHASHSEED), so that ordering -- and therefore which color a given
series/stack gets and where it lands in the legend -- could silently differ
between two runs of the same notebook. This test runs the ordering logic in
two subprocesses with different PYTHONHASHSEED values and asserts the
resulting order is identical, which would have failed before the
dict.fromkeys() fix.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]

_PROBE = """
import sys
sys.path.insert(0, {scripts_dir!r})
from plots import bar_side_by_side, bar_stacked

data = {{
    "layer1": {{"sram": 1.0, "rram": 2.0, "dram": 3.0, "disk": 4.0}},
    "layer2": {{"sram": 5.0, "rram": 6.0, "dram": 7.0, "disk": 8.0}},
}}

ax = bar_side_by_side(data)
side_by_side_order = [t.get_text() for t in ax.get_legend().get_texts()]

ax2 = bar_stacked(data)
stacked_order = [t.get_text() for t in ax2.get_legend().get_texts()]

print(side_by_side_order)
print(stacked_order)
"""


def _run_with_hashseed(seed: str) -> str:
    env = {"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}
    code = _PROBE.format(scripts_dir=str(SCRIPTS_DIR))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_series_and_stack_order_is_stable_across_hash_seeds():
    out_seed0 = _run_with_hashseed("0")
    out_seed1 = _run_with_hashseed("1")
    assert out_seed0 == out_seed1, (
        "bar_side_by_side/bar_stacked ordering changed between "
        "PYTHONHASHSEED=0 and PYTHONHASHSEED=1 -- ordering is not "
        "process-independent.\n"
        f"seed=0:\n{out_seed0}\nseed=1:\n{out_seed1}"
    )
