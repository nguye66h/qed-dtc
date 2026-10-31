"""Short integrations of the three solvers.

These are the same commands as the local example in the README. They check
that each script starts, writes a file, and returns finite numbers. They do
not reproduce the paper figures.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _run(script, args, workdir, extra_env):
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / script), *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{script} failed with code {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )


def _load(path):
    data = np.loadtxt(path, dtype=complex)
    assert data.ndim == 2 and data.shape[0] >= 1
    assert np.isfinite(data.real).all()
    assert np.isfinite(data.imag).all()
    return data


def test_cumulant_mean_field_and_second_order(tmp_path):
    # One drive period. t_eval is once per period, so the file has two rows.
    common = [
        "0.0", "0.05", "lambda", "1.0", "1.0", "2",
        "0.0", "1.0", "0.0", "0.0",
    ]
    for method in ("mf", "psc", "fsc"):
        outfile = tmp_path / f"{method}.txt"
        _run(
            "cumulants.py",
            common + [method, "rk", "1", "True"],
            tmp_path,
            {"QED_DTC_OUTFILE": str(outfile)},
        )
        data = _load(outfile)
        assert data.shape == (2, 2)
        # Column 1 is jx (mean field) or sx (cumulants); both start at sx_init.
        assert data[0, 1].real == 1.0


def test_exact_master_equation_and_one_trajectory(tmp_path):
    # N=1, photon cutoff 3, 0.2 periods at dt=0.2 -> two time samples.
    # Positional args: ep k type lambda per N a sx sy sz tmax nphoton
    # evolving dt modulated ntraj i_
    base = [
        "0.0", "0.05", "lambda", "1.0", "1.0", "1",
        "0.0", "1.0", "0.0", "0.0",
        "0.2", "3", "True", "0.2", "True",
    ]
    outdir = tmp_path / "exact"
    _run(
        "exact.py",
        base + ["0", "0"],
        tmp_path,
        {"QED_DTC_OUTDIR": str(outdir)},
    )
    mesolve = _load(outdir / "me-results.txt")
    assert mesolve.shape[0] == 2
    assert mesolve.shape[1] == 6
    # Vacuum cavity and the +Jx eigenstate: <Jx>(0) = 1/2 for N=1.
    assert mesolve[0, 0].real == 0.0
    assert abs(mesolve[0, 2].real - 0.5) < 1e-8

    _run(
        "exact.py",
        base + ["1", "0"],
        tmp_path,
        {"QED_DTC_OUTDIR": str(outdir)},
    )
    mcwf = _load(outdir / "i=0.txt")
    assert mcwf.shape[0] == 2
    assert mcwf.shape[1] == 6
    assert abs(mcwf[0, 2].real - 0.5) < 1e-8


def test_truncated_wigner_one_trajectory(tmp_path):
    # Two samples (t = 0 and t = 0.05 T), one trajectory, N=2.
    # Positional args: w w0 k type lambda per N a sx sy sz tmax dt ntraj task modulated
    outdir = tmp_path / "twa"
    _run(
        "twa.py",
        [
            "1.0", "1.0", "0.05", "lambda", "1.0", "1.0", "2",
            "0.0", "1.0", "0.0", "0.0",
            "0.1", "0.05", "1", "0", "True",
        ],
        tmp_path,
        {"QED_DTC_OUTDIR": str(outdir)},
    )
    data = _load(outdir / "i=0.txt")
    assert data.shape == (2, 6)
    assert data[0, 0].real == 0.0
    # The discrete Wigner points are rotated onto +x, so <sx>(0) = 1.
    assert abs(data[0, 2].real - 1.0) < 1e-8
