# Finite-size discrete time crystals in cavity QED

Code for the preprint

> Haimi Nguyen and David R. Reichman, *Fate of Discrete Time Crystals in Finite-Size Cavity QED Systems*, arXiv:2610.27589 (20 October 2026), submitted to Physical Review Letters.
> <https://arxiv.org/abs/2610.27589>

The preprint studies a periodically driven, lossy Dicke model: \(N\) two-level atoms coupled to one cavity mode. In the thermodynamic limit a discrete time crystal (DTC) can be stable. At finite \(N\) the DTC amplitude and its decay rate depend on \(N\) non-monotonically. The second-order cumulant expansion does not capture that finite-size dynamics. A stochastic truncated Wigner approximation (TWA) does, including at atom numbers that are out of reach of an exact wave-function simulation. Exact integration and the TWA together show three dynamical regimes (unstable, transient DTC, and pre-thermal DTC).

Author of this repository: Haimi Nguyen, [ORCID 0000-0003-1685-445X](https://orcid.org/0000-0003-1685-445X), Department of Chemistry, Columbia University. Coauthor of the preprint: David R. Reichman, Department of Chemistry, Columbia University.

Please cite the preprint if you use the code or the data in `figures/`. A BibTeX entry and a `CITATION.cff` file are below.

```bibtex
@article{nguyen2026fate,
  title   = {Fate of Discrete Time Crystals in Finite-Size Cavity {QED} Systems},
  author  = {Nguyen, Haimi and Reichman, David R.},
  journal = {arXiv preprint arXiv:2610.27589},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.27589},
  url     = {https://arxiv.org/abs/2610.27589}
}
```

## Layout

| Path | Role |
| --- | --- |
| `src/exact.py` | QuTiP master equation, or Monte Carlo wave functions |
| `src/twa.py` | Stochastic truncated Wigner approximation |
| `src/cumulants.py` | Mean field, partial second cumulant, and full second cumulant |
| `batchsubmit_exact.py`, `jobsubmit_exact_cmd.sh` | SLURM launcher for `exact.py` / `cumulants.py` |
| `batchsubmit_twa.py`, `jobsubmit_twa_cmd.sh` | SLURM launcher for `twa.py` |
| `inputs_exact.txt`, `inputs_twa.txt` | Example cluster parameter tables (not laptop-sized) |
| `figures/` | Plotting scripts, rendered figures, and the trajectories behind them |
| `tests/test_smoke.py` | Short integrations of all three solvers |

The equations of motion are the ones used for the preprint. This repository does not change them. `src/exact.py` has three compatibility edits so the same calls run on QuTiP 4 and QuTiP 5: `jspin` is imported from whichever module exports it, the spin matrices are diagonalized through `.full()`, and `e_ops` is passed by keyword because QuTiP 5 made that argument keyword-only. The operators, the initial state, and the solvers (`mesolve`, `mcsolve`) are unchanged.

## What the three solvers actually do

Time is measured in units of the drive period \(T = 2\pi\), with the reference frequency set to 1. The detuning parameter in `exact.py` and `cumulants.py` is \(\varepsilon\): the atomic frequency is \(\omega = 1+\varepsilon\) and the cavity frequency is \(\omega_c = 1-\varepsilon\). In `twa.py` those two frequencies are passed directly as `w0` and `w`. The critical coupling used when `type_of_coupling` is `per` is

\[
\lambda_c = \frac12\sqrt{\frac{\omega}{\omega_c}\left(\omega_c^2+\frac{\kappa^2}{4}\right)}.
\]

`modulated=True` (the string `True`) makes \(\lambda(t)\) a square wave: \(\lambda_0\) on the first half of each period and 0 on the second half. Any other string keeps \(\lambda(t)=\lambda_0\).

### `src/exact.py` — symmetric-subspace QuTiP

This is the exact open-system solver, in the Dicke manifold of maximum spin \(j=N/2\) (dimension \(N+1\)) tensored with a Fock space of cutoff `nphoton`. Collective operators come from QuTiP's `jspin`, restricted to that symmetric block.

- `ntraj=0` calls `qutip.mesolve` (deterministic master equation with cavity loss \(\kappa\)). The output file is `me-results.txt`.
- `ntraj>0` calls `qutip.mcsolve` with that many Monte Carlo wave-function trajectories. The output file is `i=<i_>.txt`. This is the MCWF method cited in the preprint.

The initial cavity state is the Fock state `|a_init>` (`0` is vacuum). Exactly one of `sx_init`, `sy_init`, `sz_init` must be `+1` or `-1`. That value is a sign flag, not an eigenvalue: `+1` selects the highest eigenstate of the corresponding collective spin, which is the fully polarized state along that axis.

`evolving` is accepted on the command line and is not read. The column named `integrator` in `inputs_exact.txt` (the sample value is `piqs`) is also not read by this script. The shell launcher uses only the `method` column to decide between `exact.py` and `cumulants.py`. The `piqs` import is used solely to build \(J_x,J_y,J_z\).

Columns of the saved array, all complex: time, \(\langle a\rangle\), \(\langle J_x\rangle\), \(\langle J_y\rangle\), \(\langle J_z\rangle\), \(\langle a^\dagger a\rangle\). The spins are collective. Plotting scripts divide \(\langle J_x\rangle\) by \(N/2\) when they want a single-atom \(\langle\sigma_x\rangle\).

### `src/twa.py` — stochastic truncated Wigner

Classical stochastic trajectories for the cavity amplitude and the spins, averaged over `ntraj` samples.

- The cavity amplitude is drawn from a complex Gaussian of width \(1/2\) centered on `a_init` (vacuum Wigner noise when `a_init` is 0).
- Each spin is one of the four points of the single-spin discrete Wigner function, rotated so the mean Bloch vector lies along whichever of \(x,y,z\) was given a nonzero initial value. The \(N\) atoms are assigned to those four points uniformly at random.
- The ODEs are integrated with `scipy.integrate.solve_ivp` (`RK45`), one step at a time, with a fresh complex cavity-noise sample frozen on each step. The noise variance is set by \(\kappa\).
- The random seed of trajectory `traj` on task `tasknum` is `tasknum * 10000 + traj`.

`type_of_coupling` and `per` are accepted and ignored. The coupling that enters the ODEs is the positional `lambda_` argument, used as \(\lambda_0\).

Columns: time, trajectory-averaged \(a\), \(\langle s_x\rangle\), \(\langle s_y\rangle\), \(\langle s_z\rangle\), and \(|a|^2 - 1/2\). The spins here are single-atom expectations, already divided by \(N\). The last column is the Wigner estimate of the photon number. Output file: `i=<tasknum>.txt`.

### `src/cumulants.py` — mean field and second-order cumulants

One shared square-wave drive, three closures selected by `method`:

| `method` | Equations |
| --- | --- |
| `mf` | Mean-field ODEs. The field is the rescaled amplitude \(a/\sqrt{N}\), and \(j_\gamma\) is the collective spin per atom (\(s_\gamma/2\)). |
| `psc` | Partial second-order cumulant. The symmetry-breaking factorization is dropped. |
| `fsc` | Full second-order cumulant, including those terms. |

`integrator` is `rk` (`RK45`) or `bdf` (`BDF`), with absolute and relative tolerances \(10^{-12}\). `a_init` may be the string `sqrtN` or a float. The output file has two columns only: time, and the \(x\) component of the spin (`jx` for `mf`, `sx` otherwise). The rest of the cumulant vector is integrated and not saved.

## Install

Python 3.10 or newer. The smoke test below was run with Python 3.12, QuTiP 5, NumPy 2, and Numba 0.68.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -r requirements.txt` installs the same runtime libraries without pytest. Figure scripts in `figures/` also import pandas and seaborn, which are included. Regenerating the plots calls `matplotlib` with `text.usetex`, so a TeX installation is required for that step and is not required for the solvers.

## Run a small example locally

The cluster scripts are not needed for this. From the repository root:

```bash
mkdir -p out
export QED_DTC_OUTDIR="$PWD/out"
export QED_DTC_OUTFILE="$PWD/out/cumulant_mf.txt"

# Mean field, one drive period, N=2, epsilon=0, kappa=0.05, lambda=1, spin +x.
python src/cumulants.py 0.0 0.05 lambda 1.0 1.0 2 0.0 1.0 0.0 0.0 mf rk 1 True

# Full second-order cumulant, same parameters.
export QED_DTC_OUTFILE="$PWD/out/cumulant_fsc.txt"
python src/cumulants.py 0.0 0.05 lambda 1.0 1.0 2 0.0 1.0 0.0 0.0 fsc rk 1 True

# Exact master equation. N=1, photon cutoff 3, 0.2 periods.
# Args: ep k type lambda per N a sx sy sz tmax nphoton evolving dt modulated ntraj task
python src/exact.py 0.0 0.05 lambda 1.0 1.0 1 0.0 1.0 0.0 0.0 0.2 3 True 0.2 True 0 0

# One Monte Carlo wave-function trajectory (same system).
python src/exact.py 0.0 0.05 lambda 1.0 1.0 1 0.0 1.0 0.0 0.0 0.2 3 True 0.2 True 1 0

# Stochastic TWA. One trajectory, N=2, two time samples.
# Args: w w0 k type lambda per N a sx sy sz tmax dt ntraj task modulated
python src/twa.py 1.0 1.0 0.05 lambda 1.0 1.0 2 0.0 1.0 0.0 0.0 0.1 0.05 1 0 True
```

Outputs:

- `out/cumulant_mf.txt`, `out/cumulant_fsc.txt`
- `out/me-results.txt` (master equation)
- `out/i=0.txt` is written by the MCWF call and then overwritten by the TWA call. Run them with different `QED_DTC_OUTDIR` values if you want both.

`tests/test_smoke.py` runs these integrations in a temporary directory and checks that the files are finite. It does not reproduce a paper figure.

```bash
python -m pytest -q
```

Paper-scale runs are much longer. The preprint uses exact propagation up to 1500 periods for \(N\le 20\) and TWA up to 5000 periods for \(N>20\), with thousands of trajectories. `inputs_exact.txt` and `inputs_twa.txt` are examples of those cluster jobs (`N=17`, photon cutoff 550, and 1000 TWA trajectories per task).

## SLURM

On the cluster the launchers submit one `sbatch` script per task. They still `module load conda` and, unless `CONDA_ENV` is set, activate `/global/common/software/py38`, which is the NERSC environment used for the paper. `QED_DTC_ROOT` overrides the repository location; the default is the directory that contains the shell script, not `$HOME/qed-dtc`.

```bash
# One row of inputs_exact.txt, expanded by batchsubmit_exact.py into ntasks jobs.
python batchsubmit_exact.py                 # reads inputs_exact.txt
python batchsubmit_exact.py path/to/table.txt

python batchsubmit_twa.py                   # reads inputs_twa.txt
```

`batchsubmit_exact.py` used to open `inputs.txt`, which is not in the repository. It now defaults to `inputs_exact.txt`.

Solver output still lands in `foldername/` (or `filename.txt` for cumulants) relative to the process working directory. The launchers `cd` to `$QED_DTC_ROOT/msgs` before `srun`, so cluster output appears under `msgs/foldername/` unless `QED_DTC_OUTDIR` / `QED_DTC_OUTFILE` is set in the job environment.

## Scripts and figures

The mapping below is from the plotting scripts and from the figure captions that are intact in the arXiv HTML. A few main-text captions are garbled in that HTML, so those rows are marked as inferred. Plot scripts expect to be run from their own directory and read `raw_data/` or `plot_data/` next to them. They are not called by `src/`.

| Paper figure | Directory | How it is produced |
| --- | --- | --- |
| Fig. 1 (inferred) | `figures/Figure1_SM1/` | `plot.ipynb` compares exact, TWA, and full second cumulants for the undriven model at \(N=10\) and \(N=55\), initial spin along \(x\) and along \(y\). The notebook writes `unmodulated_dynamics.*`. The committed image is `Fig1.png`. |
| SM 1 (inferred) | `figures/Figure1_SM1/` | Same notebook, initial spin along \(z\), now including the partial second cumulant. Committed image: `SM1.png`. |
| Fig. 2 (inferred) | `figures/Figure2/` | `plot_dtc_regimes.py` plots stroboscopic \(\langle\sigma_x\rangle\) for three regimes: unstable (\(\varepsilon=-0.065\), \(N=30\)), transient (\(\varepsilon=-0.056\), \(N=10\)), pre-thermal (\(\varepsilon=0.01\), \(N=20\)), overlaying exact, TWA, and full second cumulants. |
| Fig. 3 (inferred; the 0.05 line is in the preprint) | `figures/Figure3/` | `plot_error.py` plots the mean absolute difference between exact and TWA \(\langle\sigma_x\rangle\) for \(N=2\ldots 30\) at \(\varepsilon\in\{-0.056,0.01,0.051,0.11\}\), with a horizontal line at 0.05. |
| Fig. 4 | `figures/Figure4/` | `get_floquet_multipliers.py` builds the largest Floquet multiplier of the linearized drive (panel a; data file `floquet_mul_heatmap_data.txt`). `plot_combined_heatmaps.py` draws that panel plus the DTC decay rate (panel b) and the finite-\(N\) amplitude ratio (panel c) from `raw_data/` and `plot_data/`. Parameters in the preprint: \(\lambda_0=1\), vacuum, spin along \(+x\), and for panels b–c a cut at \(\kappa=0.05\). Exact data are the \(N\le 20\) directories; TWA data are the rest. |
| SM 2 (inferred) | `figures/SM2/` | `plot_number_ratios.py` and `plot.ipynb` plot \(\langle a^\dagger a\rangle/N\) against \(N\) for exact, TWA, and cumulant data. Matches the supplement comparison cited to Kirton and Keeling. |
| SM 3 (inferred) | `figures/SM3/` | `plot_noise_effect.py` compares TWA with and without cavity noise at \(\varepsilon=-0.056\) and \(\varepsilon=0.01\), against a \(1/N\) guide. |
| SM 4 | `figures/SM4/` | `plot_sz_timescale.py` is a standalone TWA/mean-field script (it does not call `src/twa.py`). As committed it uses \(\varepsilon=0.03\), \(\kappa=0.05\), \(\lambda_0=1\), and \(N\in\{13,50,70\}\), and plots \(s_x(t)\) and \(s_z(t)\) against the mean-field steady \(\bar s_z\). The supplement caption for this panel was not recoverable from the arXiv HTML. |
| SM 5 | `figures/SM5/basins_twa_axes.html` | Interactive slice of the two stroboscopic mean-field basins. The preprint's SM 5 is this figure (\(\varepsilon=0\), \(\kappa=0.05\), \(\lambda_0=1\)). |
| SM 6 and SM 8 | `figures/SM6_SM8/` | `plot_phase_ratio.py` is another standalone TWA script. It averages in-phase and out-of-phase trajectories with and without noise and writes `phase_ratios.png`. The committed images are `SM6.png` and `SM8.png`. The script as committed has \(\varepsilon=0\) and \(N\in\{10,100,1000\}\) with 10,000 trajectories, which is the SM 6 setup only approximately: the preprint quotes \(N=30,80,1000\) for SM 6 and \(\varepsilon=0.09\), \(N=30,10^4,10^6\) for SM 8. |
| SM 7 | `figures/SM7/cavity-spin-visualization.html` | Interactive \((s_x,s_y)\) trajectories. The preprint's SM 7 is this visualization. |

## Data already in the repository

`figures/` is about 8.8 GB and about 1,760 files, almost all raw trajectories. Those files are intentionally still in git. `.gitignore` ignores *new* files under `figures/**/raw_data/`, `figures/**/plot_data/`, and `figures/SM3/ep=*/`, and it ignores the local output names `foldername/`, `filename.txt`, and `out/`. Ignoring a path does not remove files that are already tracked.

Rendered images (`Fig1.png`, `Fig2.png`, `Fig3.png`, `Fig4.png`, `SM1.png`–`SM4.png`, `SM6.png`, `SM8.png`, and the two HTML files) are small next to the trajectories and are the practical copy of the paper figures.

## License

[MIT](LICENSE). Copyright 2026 Haimi Nguyen.
