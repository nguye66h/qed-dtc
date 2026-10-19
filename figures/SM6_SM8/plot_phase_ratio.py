
import numpy as np
from numba import jit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Computer Modern Roman', size=16)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')


# --- Initial condition samplers ---
def sample_initial_spin(th0=np.pi/2, ph0=0):
    # Fixed-axis modes: all 4 spin configs set to the same unit vector
    if SPIN_INIT == 'sx':
        return np.ones(4), np.zeros(4), np.zeros(4)
    if SPIN_INIT == 'sy':
        return np.zeros(4), np.ones(4), np.zeros(4)
    if SPIN_INIT == 'sz':
        return np.zeros(4), np.zeros(4), np.ones(4)
    # Default 'twa': discrete Wigner 4-configuration sampling
    sx0 = np.array([1.0,  1.0, -1.0, -1.0])
    sy0 = np.array([1.0, -1.0,  1.0, -1.0])
    sz0 = np.ones(4)
    sx_i = sx0*np.cos(th0)*np.cos(ph0) - sy0*np.sin(ph0) + sz0*np.sin(th0)*np.cos(ph0)
    sy_i = sx0*np.cos(th0)*np.sin(ph0) + sy0*np.cos(ph0) + sz0*np.sin(th0)*np.sin(ph0)
    sz_i = -sx0*np.sin(th0) + sz0*np.cos(th0)
    return sx_i, sy_i, sz_i

def sample_a(rng):
    return (rng.normal(0, 0.5) + 1j * rng.normal(0, 0.5))


# --- JIT-compiled dynamics ---
@jit(nopython=True, cache=True)
def g_t(t, dt, modu):
    t_mod = (t + dt/2) % (2 * np.pi)
    if t_mod < np.pi:
        return g
    return 0.0 if modu == 'True' else g

# just generic eom
@jit(nopython=True, cache=True)
def _twa_rhs(t, y, noise, counts, sqrt_N, w, w0, gt):
    a  = y[0]
    sx = y[1:5]
    sy = y[5:9]
    sz = y[9:13]

    da  = -1j*w*a - 1j*gt/sqrt_N*np.sum(counts*sx) - kappa/2*a - noise/2
    dsx = -w0 * sy
    dsy = w0*sx - 4*gt/sqrt_N * sz * a.real
    dsz = 4*gt/sqrt_N * sy * a.real
    out = np.zeros(13, dtype=np.complex128)
    out[0]    = da
    out[1:5]  = dsx
    out[5:9]  = dsy
    out[9:13] = dsz
    return out

# integrate over a dt with frozen noise for n times or t=dt*n
@jit(nopython=True, cache=True)
def _integrate_twa(y0, noise_arr, counts, dt, w, w0, modu):
    """RK4 with frozen noise per step over the full trajectory."""
    n    = noise_arr.shape[0]
    sqN  = np.sqrt(np.sum(counts))
    a_out  = np.zeros(n + 1, dtype=np.complex128)
    sx_out = np.zeros((n + 1, 4), dtype=np.complex128)
    sy_out = np.zeros((n + 1, 4), dtype=np.complex128)
    sz_out = np.zeros((n + 1, 4), dtype=np.complex128)

    y = y0.copy()
    a_out[0]  = y[0]
    sx_out[0] = y[1:5]
    sy_out[0] = y[5:9]
    sz_out[0] = y[9:13]

    t = 0.0
    for i in range(n):
        noise = noise_arr[i]
        gt = g_t(t, dt, modu)
        k1 = _twa_rhs(t,        y,            noise, counts, sqN, w, w0, gt)
        k2 = _twa_rhs(t+dt/2,   y+dt/2*k1,   noise, counts, sqN, w, w0,  gt)
        k3 = _twa_rhs(t+dt/2,   y+dt/2*k2,   noise, counts, sqN, w, w0,  gt)
        k4 = _twa_rhs(t+dt,     y+dt*k3,     noise, counts, sqN, w, w0,  gt)
        y  = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        a_out[i+1]  = y[0]
        sx_out[i+1] = y[1:5]
        sy_out[i+1] = y[5:9]
        sz_out[i+1] = y[9:13]

    return np.linspace(0, t, n+1), a_out, sx_out, sy_out, sz_out


def _integrate_twa_ivp(y0, noise_arr, counts, dt, w, w0, modu):
    """solve_ivp (adaptive RK45) with frozen noise per step.
    Identical physics to _integrate_twa but uses scipy adaptive stepping.
    Note: much slower than the custom RK4 — use for cross-checking only.
    """
    n   = noise_arr.shape[0]
    sqN = np.sqrt(np.sum(counts))

    a_out  = np.zeros(n + 1, dtype=np.complex128)
    sx_out = np.zeros((n + 1, 4), dtype=np.complex128)
    sy_out = np.zeros((n + 1, 4), dtype=np.complex128)
    sz_out = np.zeros((n + 1, 4), dtype=np.complex128)

    y = y0.copy()
    a_out[0]  = y[0]
    sx_out[0] = y[1:5]
    sy_out[0] = y[5:9]
    sz_out[0] = y[9:13]

    t = 0.0
    for i in range(n):
        noise  = noise_arr[i]
        gt = g_t(t, dt, modu)
        t_next = t + dt
        sol = solve_ivp(
            _twa_rhs,
            (t, t_next),
            y,
            method='RK45',
            t_eval=[t_next],
            args=(noise, counts, sqN, w, w0, gt),
            atol=1e-12, rtol=1e-12,
        )
        y   = sol.y[:, -1]
        t   = t_next
        a_out[i+1]  = y[0]
        sx_out[i+1] = y[1:5]
        sy_out[i+1] = y[5:9]
        sz_out[i+1] = y[9:13]

    return np.linspace(0, t, n+1), a_out, sx_out, sy_out, sz_out


# --- Mean field (called once per N, solve_ivp overhead is negligible) ---
def _mf_rhs(t, y, w, w0, gt):
    a, jx, jy, jz = y
    return np.array([
        -(1j*w + kappa/2)*a - 2j*gt*jx,
        -w0*jy,
        w0*jx - 4*gt*a.real*jz,
        4*gt*a.real*jy,
    ])


def integrate_mf_ivp(y0, t, dt, w, w0, modu):
    
    n   = int (round (t/dt))

    a_out  = np.zeros(n + 1, dtype=np.complex128)
    jx_out = np.zeros((n + 1), dtype=np.complex128)
    jy_out = np.zeros((n + 1), dtype=np.complex128)
    jz_out = np.zeros((n + 1), dtype=np.complex128)

    y = y0.copy()
    a_out[0]  = y[0]
    jx_out[0] = y[1]
    jy_out[0] = y[2]
    jz_out[0] = y[3]

    t = 0.0
    for i in range(n):
        gt = g_t(t, dt, modu)
        t_next = t + dt
        sol = solve_ivp(
            _mf_rhs,
            (t, t_next),
            y,
            method='RK45',
            t_eval=[t_next],
            args=(w, w0, gt),
            atol=1e-12, rtol=1e-12,
        )
        y   = sol.y[:, -1]
        t   = t_next
        a_out[i+1]  = y[0]
        jx_out[i+1] = y[1]
        jy_out[i+1] = y[2]
        jz_out[i+1] = y[3]

    return np.linspace(0, t, n+1), a_out, jx_out, jy_out, jz_out


# input noise and the spins are still individuals
# --- Simulation runners ---
def run_trajectory(traj_idx, w, w0, zero_noise=False):
    rng = np.random.default_rng(seed=traj_idx*20 + 10**8)
    #rng = np.random.default_rng()
    sx0, sy0, sz0 = sample_initial_spin()
    a0 = sample_a(rng)

    y0 = np.zeros(13, dtype=np.complex128)
    y0[0]    = a0
    y0[1:5]  = sx0
    y0[5:9]  = sy0
    y0[9:13] = sz0

    counts = rng.multinomial(N, [0.25, 0.25, 0.25, 0.25]).astype(np.float64)

    n_steps   = int(round(t_max / timestep))
    scale     = np.sqrt(kappa / timestep)
    if zero_noise:
        noise_arr = np.zeros(n_steps, dtype=np.complex128)
    else:
        noise_arr = rng.normal(0, scale, n_steps) + 1j*rng.normal(0, scale, n_steps)

    if INTEGRATOR == 'rk4':
        t_arr, a_arr, sx_arr, sy_arr, sz_arr = _integrate_twa(
            y0, noise_arr, counts, timestep, w, w0, modulated
        )
    else:
        t_arr, a_arr, sx_arr, sy_arr, sz_arr = _integrate_twa_ivp(
            y0, noise_arr, counts, timestep, w, w0, modulated
        )
    results = {
        't':         t_arr,
        'a_complex': a_arr,
        'sx':        sx_arr,
        'sy':        sy_arr,
        'sz':        sz_arr,
    }
    return results, counts

# save averaged collective spins for each and all trajs
def run_simulation(num_trajectories, w, w0, zero_noise=False):
    results = {}
    for traj_ind in range(num_trajectories):
        traj, counts = run_trajectory(traj_ind, w, w0, zero_noise=zero_noise)
        if not results:
            n = len(traj['a_complex'])
            results = {
                't':         traj['t'],
                'a_complex': np.zeros((num_trajectories, n), dtype=complex),
                'sx':        np.zeros((num_trajectories, n), dtype=complex),
                'sy':        np.zeros((num_trajectories, n), dtype=complex),
                'sz':        np.zeros((num_trajectories, n), dtype=complex),
                'a_sq_mag':  np.zeros((num_trajectories, n))
            }
        results['a_complex'][traj_ind] = traj['a_complex']
        results['sx'][traj_ind] = np.sum(traj['sx'] * counts, axis=1) / N
        results['sy'][traj_ind] = np.sum(traj['sy'] * counts, axis=1) / N
        results['sz'][traj_ind] = np.sum(traj['sz'] * counts, axis=1) / N
        results['a_sq_mag'][traj_ind] = (traj['a_complex'].real**2
                                         + traj['a_complex'].imag**2)
    return results

# --- Parameters ---
ep    = 0.0
kappa = 0.05
g     = 1.0

modulated = 'True'
INTEGRATOR = 'rk4'     # 'rk4'  →  fast fixed-step RK4 (default)
                        # 'ivp'  →  scipy solve_ivp (adaptive, slower but cross-check)

SPIN_INIT  = 'twa'     # 'twa'  →  discrete Wigner 4-config sampling (default)
                        # 'sx'   →  all spins: sx=1, sy=0, sz=0
                        # 'sy'   →  all spins: sx=0, sy=1, sz=0
                        # 'sz'   →  all spins: sx=0, sy=0, sz=1


# --- Main execution ---
N_values         = [10, 100, 1000]
t_max            = 50 * 2 * np.pi
timestep         = 0.01 * 2 * np.pi
t_eval           = np.arange(0, t_max, timestep)
num_trajectories = 10000
btctrajn         = 1
t_sync_periods   = 30          # compare signs at t = t_sync_periods * 2π

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Trigger numba compilation before the timed loop
N = N_values[0]
print("Compiling JIT functions …")
run_trajectory(0, 1.0 - ep, 1.0 + ep)

al_val = 0
sx_val = 1

w  = 1.0 - ep
w0 = 1.0 + ep

mf_t, mf_a, mf_jx, mf_jy, mf_jz = integrate_mf_ivp(np.array([al_val, sx_val/2, 0.0, 0.0], dtype=complex),
                            t_max, timestep, w, w0, modulated)

lambda_c = 0.5 * np.sqrt (w0 / w * (w**2 + kappa**2/4))
sz_const = -1 * lambda_c**2 / g**2

for i, N in enumerate(N_values):

    t0 = time.time()
    twa_trajs = run_simulation(num_trajectories, w, w0)
    print(f"N={N}: TWA done in {time.time()-t0:.1f}s")

    t0 = time.time()
    twa_noiseless = run_simulation(num_trajectories, w, w0, zero_noise=True)
    print(f"N={N}: TWA w/o noise done in {time.time()-t0:.1f}s")

    ax = axes[i]

    ax.plot(twa_trajs['t'][::100] / (2*np.pi),
            np.mean(twa_trajs['sx'], axis=0)[::100].real,
            color='g', linewidth=1.0, label=fr'$TWA$')
        
    ax.plot(twa_noiseless['t'][::100] / (2*np.pi),
            np.mean(twa_noiseless['sx'], axis=0)[::100].real,
            color='r', linewidth=2, ls='--', label='TWA w/o noise')
    
    ax.plot(mf_t[::100] / (2*np.pi),
            mf_jx[::100].real * 2,
            color='b', linewidth=1, alpha=0.5, label='Mean Field')
    


    # --- Phase-sync inset ---
    t_sync_twa = twa_trajs['t'][::100][t_sync_periods] / (2 * np.pi)
    t_sync_mf  = mf_t[::100][t_sync_periods]     / (2 * np.pi)
    mf_sign    = np.sign(mf_jx[::100][t_sync_periods].real)
    twa_sx_sync = twa_trajs['sx'][:, ::100][:, t_sync_periods].real  # shape (num_trajectories,)
    n_in  = int(np.sum(np.sign(twa_sx_sync) == mf_sign))
    n_out = int(np.sum(np.sign(twa_sx_sync) != mf_sign))
    print(f"  sync at index {t_sync_periods} → t={t_sync_twa:.2f} periods: n_in={n_in}, n_out={n_out}")

    # mark the sync point on the plot
    # ax.plot(t_sync_twa, np.mean(twa_sx_sync),
    #         marker='o', color='g', markersize=7, zorder=5)
    # ax.plot(t_sync_mf,  mf_traj.y[1][::100][t_sync_periods].real * 2,
    #         marker='o', color='b', markersize=7, zorder=5)
    pct_in  = 100.0 * n_in  / (n_in + n_out)
    pct_out = 100.0 * n_out / (n_in + n_out)

    inset = ax.inset_axes([0.74, 0.02, 0.24, 0.33])
    inset.pie(
        [pct_in, pct_out],
        colors=['#4CAF50', '#E57373'],
        startangle=90,
        autopct=lambda p: f'{p:.0f}\\%',
        pctdistance=0.5,
        textprops={'fontsize': 12},
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
    )

    ax.set_title(fr'$N = {N:,}$', fontsize=20)
    ax.set_xlabel(fr'$t/T (1/\omega)$', fontsize=20)
    ax.set_ylabel(fr'$\langle s_x \rangle$', fontsize=20)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, t_max / (2*np.pi))
    ax.grid(True, linestyle='--', alpha=0.5)
axes[1].legend(fontsize=14, loc='upper left')
plt.tight_layout()
plt.savefig(f'phase_ratios.png', dpi=600)
