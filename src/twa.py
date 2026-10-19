import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, List
from numba import jit
import matplotlib.pyplot as plt
import time
import argparse
import os

def sample_initial_spin():
    """
    Sample the initial spin condition according to the specified distribution.
    
    Returns:
    - Tuple (sx, sy, sz) of initial spin components (-1 or 1)
    """

    # The fixed spin component is always 1
    sx0 = np.array([1, 1, -1, -1])
    sy0 = np.array([1, -1, 1, -1])
    sz0 = np.ones(4)

    sx_i = sx0 * np.cos(th0) * np.cos(ph0) - sy0 * np.sin(ph0) + sz0 * np.sin(th0) * np.cos(ph0)
    sy_i = sx0 * np.cos(th0) * np.sin(ph0) + sy0 * np.cos(ph0) + sz0 * np.sin(th0) * np.sin(ph0)
    sz_i = -sx0 * np.sin(th0) + sz0 * np.cos(th0)
            
    return sx_i, sy_i, sz_i

@jit(nopython=True)
def sample_a(rng):
    """
    Sample the initial value of 'a' from a Gaussian distribution.
    
    Returns:
    - Complex number sampled from W(a) = exp[-2|a-alpha|^2]
    """
    # Sample real and imaginary parts independently
    a_real = rng.normal(np.real(initial_alpha), 0.5)
    a_imag = rng.normal(np.imag(initial_alpha), 0.5)
    
    return a_real + 1j * a_imag


@jit(nopython=True)
def sample_noise(sample_nspins, rng):
    """
    Sample the noise terms for one time step.
    
    Returns:
    - Tuple of (epsilon_k, epsilon_xup, epsilon_yup, epsilon_zup,
                epsilon_xdown, epsilon_ydown, epsilon_zdown)
    """
    
    # Complex noise for epsilon_k
    epsilon_k_real, epsilon_k_imag = rng.normal(0, np.sqrt(kappa / timestep), size=2)
    epsilon_k = epsilon_k_real + 1j * epsilon_k_imag
    
    return (epsilon_k)

# Define the periodic lambda_t function
@jit(nopython=True)
def g_t(t, modu):
    # Modulo operation to get the coupling within period T
    t_mod = (t + timestep // 2) % (2 * np.pi)
    if t_mod < np.pi:
        return g  # First half of the period
    else:
        if modu == 'True':
            return 0  # Second half of the period
        else:
            return g

@jit(nopython=True)
def equations_of_motion(t: float, y: np.ndarray, noise, sample_nspins, modu, counts):
    """
    Define the differential equations for the system.
    
    Parameters:
    - t: float, current time (not used but required by solver)
    - y: np.ndarray, current state vector [a_real, a_imag, sx, sy, sz]
    - noise: Tuple of noise terms from sample_noise()
    
    Returns:
    - np.ndarray, derivatives of the state vector
    """
    # Unpack state variables
    a = y[0]
    sx, sy, sz = y[1:1+4], y[1+4:1+2*4], y[1+2*4:1+3*4]
        
    # Unpack noise terms
    (epsilon_k) = noise
    
    gt = g_t(t, modu)
    
    # Calculate derivatives
    da_dt = (-1j * w * a - 1j * gt / np.sqrt(sample_nspins) * np.sum (counts * sx)
                - kappa / 2 * a - epsilon_k / 2 )
    
    dsx_dt = (-w0 * sy )
    
    dsy_dt = (w0 * sx - 4 * gt / np.sqrt(sample_nspins) * sz * a.real )
    
    dsz_dt = (4 * gt / np.sqrt(sample_nspins) * sy * a.real )
    
    # Return derivatives as real array
    derivatives = np.zeros(3 * 4 + 1, dtype=np.complex128)
    derivatives[0] = da_dt
    derivatives[1:1+4] = dsx_dt
    derivatives[1+4:1+2*4] = dsy_dt
    derivatives[1+2*4:1+3*4] = dsz_dt

    return derivatives

def run_trajectory(traj_idx):
    """
    Run a single trajectory simulation.
    
    Returns:
    - Dictionary containing time points and state variables
    """
    rng = np.random.default_rng(seed=tasknum*10000+traj_idx)
    print('seed id', tasknum*10000+traj_idx)
    # Initial conditions
    sx0, sy0, sz0 = sample_initial_spin()
    a0 = sample_a(rng)
    y0 = np.concatenate(([a0], sx0, sy0, sz0))
    
    # Time points for output
    t_eval = np.arange(0, t_max, timestep)
    
    # Store results
    results = {
        't': t_eval,
        'a_complex': np.zeros_like(t_eval, dtype=complex),
        'sx': np.zeros((len(t_eval),4), dtype=complex),
        'sy': np.zeros((len(t_eval),4), dtype=complex),
        'sz': np.zeros((len(t_eval),4), dtype=complex)
    }

    # Initial state
    current_y = y0
    results['a_complex'][0] = y0[0]
    results['sx'][0,0:4] = y0[1:1+4]
    results['sy'][0,0:4] = y0[1+4:1+2*4]
    results['sz'][0,0:4] = y0[1+2*4:1+3*4]

    paths = rng.choice([0, 1, 2, 3], size=nspins)
    counts = np.bincount(paths, minlength=4)
    
    # Time stepping
    for i in range(1, len(t_eval)):
        # Sample new noise for this time step
        noise = sample_noise(nspins, rng)
        
        # Time interval for this step
        t_span = (t_eval[i-1], t_eval[i])
        
        # Solve ODE for this step
        sol = solve_ivp(equations_of_motion,
            t_span=t_span,
            y0=current_y,
            t_eval=[t_eval[i]],
            method='RK45',
            args=(noise, nspins, modulated, counts)  # Pass noise as argument
        )
        
        # Update current state
        current_y = sol.y[:, -1]
        
        # Store results
        results['a_complex'][i] = current_y[0]
        results['sx'][i,0:4] = current_y[1:1+4]
        results['sy'][i,0:4] = current_y[1+4:1+2*4]
        results['sz'][i,0:4] = current_y[1+2*4:1+3*4]

    return results, counts

def run_simulation():
    """
    Run multiple trajectories and average the results.
    
    Returns:
    - Dictionary containing time points and averaged state variables
    """
    
    # Initialize storage for averages
    avg_results = None
    
    for traj_ind in range(num_trajectories):
        traj_results, counts = run_trajectory(traj_ind)
        
        # Initialize average results on first trajectory
        if avg_results is None:
            avg_results = {
                't': traj_results['t'],
                'a_complex': np.zeros_like(traj_results['a_complex'], dtype=complex),
                'sx': np.zeros_like(traj_results['sx'], dtype=complex),
                'sy': np.zeros_like(traj_results['sy'], dtype=complex),
                'sz': np.zeros_like(traj_results['sz'], dtype=complex),
                'a_sq_mag': np.zeros_like(traj_results['a_complex'], dtype=complex)
            }
        
        # Accumulate results
        avg_results['a_complex'] += traj_results['a_complex']
        avg_results['sx'] += traj_results['sx'] * counts / nspins
        avg_results['sy'] += traj_results['sy'] * counts / nspins
        avg_results['sz'] += traj_results['sz'] * counts / nspins
        
        # Compute |a|^2 for this trajectory and accumulate
        a_sq_mag = (traj_results['a_complex'].real**2 + traj_results['a_complex'].imag**2)
        avg_results['a_sq_mag'] += a_sq_mag
    
    # Average the results
    for key in ['a_complex', 'sx', 'sy', 'sz', 'a_sq_mag']:
        avg_results[key] /= num_trajectories
    
    return avg_results

def compute_number_average(simulation_results: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute the number average |a|^2 from simulation results.
    
    Parameters:
    - simulation_results: Dictionary returned by run_simulation()
    
    Returns:
    - np.ndarray of |a|^2 averaged over trajectories
    """
    # This is already computed in run_simulation()
    return simulation_results['a_sq_mag'].real - 0.5

def compute_spin_average(simulation_results: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute the number average |a|^2 from simulation results.
    
    Parameters:
    - simulation_results: Dictionary returned by run_simulation()
    
    Returns:
    - np.ndarray of |a|^2 averaged over trajectories
    """
    # This is already computed in run_simulation()
    sxmean = np.sum(simulation_results['sx'].real, axis=1)
    symean = np.sum(simulation_results['sy'].real, axis=1)
    szmean = np.sum(simulation_results['sz'].real, axis=1)
    return np.array([sxmean, symean, szmean])

# Create the parser
parser = argparse.ArgumentParser()

## regime parameters
parser.add_argument('w', type=float, help="w")
parser.add_argument('w0', type=float, help="w0")
parser.add_argument('k', type=float, help="kappa cavity loss rate")
parser.add_argument('type_of_coupling', type=str, help="write per if percentage, lambda if absolute val")
parser.add_argument('lambda_', type=float, help="coupling strength")
parser.add_argument('per', type=float, help="percentage of coupling strength to critical coupling")
parser.add_argument('N', type=int, help="Number of spins N")

## initial conditions
parser.add_argument('a_init', type=float, help="initial value for a, float")
parser.add_argument('sx_init', type=float, help="initial value for sigma_x in second cumulant or jx in mean field")
parser.add_argument('sy_init', type=float, help="initial value for sigma_y in second cumulant or jy in mean field")
parser.add_argument('sz_init', type=float, help="initial value for sigma_z in second cumulant or jz in mean field")

## simulation details
parser.add_argument('tmax', type=float, help="max simulation time")
parser.add_argument('dt', type=float, help="timestep dt")
parser.add_argument('ntraj', type=int, help="number of trajectories, integer")
parser.add_argument('tasknum', type=int, help="task number, integer")
parser.add_argument('modulated', type=str, help="modulated or not, True or False")

# Parse the arguments
args = parser.parse_args()

if args.sx_init != 0.0:
    titlearray=['sx']
elif args.sy_init != 0.0:
    titlearray=['sy']
elif args.sz_init != 0.0:
    titlearray=['sz']

th0array = []
ph0array = []
for ti in range(len(titlearray)):
    if titlearray[ti] == 'sz':
        th0array.append(0)
        ph0array.append(0)
    elif titlearray[ti] == 'sy':
        th0array.append(np.pi/2)
        ph0array.append(np.pi/2)
    elif titlearray[ti] == 'sx':
        th0array.append(np.pi/2)
        ph0array.append(0)

th0 = th0array[0]          # Polar angle
ph0 = ph0array[0]          # Azimuthal angle
initial_alpha = args.a_init + 0.0j  # Complex initial mean of a
kappa = args.k         # Cavity damping rate
gamma_up = 0     # Spin up rate
gamma_down = 0   # Spin down rate
w = args.w             # Cavity frequency
w0 = args.w0            # Spin frequency
g = args.lambda_             # Coupling strength
nspins = args.N             # Number of spins
t_max = args.tmax * 2 * np.pi        # Maximum simulation time
timestep = args.dt * 2 * np.pi          # Time step
num_trajectories = args.ntraj  # Number of trajectories to average
modulated = args.modulated
tasknum = args.tasknum

start_time = time.time()
# Create and run simulator
myresults = run_simulation()
end_time = time.time()
print(f"Simulation took {end_time - start_time:.2f} seconds")

# Compute number average
spin_avg = compute_spin_average(myresults)
number_avg = compute_number_average(myresults)

directory = f'foldername'
os.makedirs(directory, exist_ok=True)
filename = f'i={args.tasknum}.txt'
full_path = os.path.join(directory, filename)
np.savetxt(full_path, np.column_stack((np.arange(0,t_max,timestep),myresults['a_complex'],*spin_avg,number_avg)))