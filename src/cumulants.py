import numpy as np
from numba import jit
import argparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments

## regime parameters
parser.add_argument('ep', type=float, help="epsilon")
parser.add_argument('k', type=float, help="kappa cavity loss rate")
parser.add_argument('type_of_coupling', type=str, help="write per if percentage, lambda if absolute val")
parser.add_argument('λ', type=float, help="coupling strength")
parser.add_argument('per', type=float, help="percentage of coupling strength to critical coupling")
parser.add_argument('N', type=int, help="Number of spins N")

## initial conditions
parser.add_argument('a_init', type=str, help="initial value for a, string, sqrtN possible")
parser.add_argument('sx_init', type=float, help="initial value for sigma_x in second cumulant or jx in mean field")
parser.add_argument('sy_init', type=float, help="initial value for sigma_y in second cumulant or jy in mean field")
parser.add_argument('sz_init', type=float, help="initial value for sigma_z in second cumulant or jz in mean field")

## simulation details
parser.add_argument('method', type=str, help="write fsc for full second cumulant, psc for partial second cumulant, mf for mean field")
parser.add_argument('integrator', type=str, help="write rk for Runge Kutta or bdf")
parser.add_argument('t_max', type=float, help="max simulation time")
parser.add_argument('modulated', type=str, help="if modulated, write True, otherwise False")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
## Define methods
method = args.method
integrator = args.integrator
modulated = args.modulated
N = args.N
sqrtN = np.sqrt(N)

## Define parameters
ε = args.ep
wT = 1.0
w0 = (1+ε)*wT  # Atomic frequency
w  = (1-ε)*wT  # Photonic frequency
k = args.k     # Decay constant
T = 2*np.pi    # Period of the periodic function
t_max = args.t_max * 2 * np.pi # Max time to propagate to
t_eval = 2 * np.pi * np.arange(0, args.t_max + 1) # Points for solve_ivp to evaluate at
print('t_max', t_max)
print('last of t_eval', t_eval[-1])
λ_c= 0.5*np.sqrt((w0/w)*(w**2+k**2/4))
if args.type_of_coupling == 'per':
    λ = args.per*λ_c         # Coupling constant
else:
    λ = args.λ
μ = (λ_c/λ)**(2)

## initial conditions
if args.a_init == "sqrtN":
    a_init = np.sqrt(N)
else:
    a_init = float(args.a_init)
sx_init = args.sx_init
sy_init = args.sy_init
sz_init = args.sz_init

## error tolerance
atol = 10**-12 # absolute tolerance
rtol = 10**-12 # relative tolerance

if method == 'mf':
    initial_state = np.array([a_init, # a
                          sx_init, # sx
                          sy_init, # sy
                          sz_init], # sz
                         dtype=complex)
else:
    initial_state = np.array([a_init, # a
                          sx_init, # sx
                          sy_init, # sy
                          sz_init, # sz
                          a_init*sx_init, # asx
                          a_init*sy_init, # asy
                          a_init*sz_init, # asz
                          sx_init**2, # sxsx
                          sy_init**2, # sysy
                          sz_init**2, # szsz
                          sx_init*sy_init, # sxsy
                          sx_init*sz_init, # sxsz
                          sy_init*sz_init, # sysz
                          a_init**2, # ada
                          a_init**2],# aa
                         dtype=complex)

# Define the periodic lambda_t function
@jit(nopython=True)
def lambda_t(t, T):
    # Modulo operation to get the coupling within period T
    t_mod = t % T
    if t_mod < T / 2:
        return λ  # First half of the period
    else:
        if modulated == 'True':
            return 0  # Second half of the period
        else:
            return λ

# Define the derivatives of mean field
@jit(nopython=True)
def mf_derivatives(t, state):
    a, jx, jy, jz = state

    lambda_val = lambda_t(t, T)

    # Coupled differential equations
    da_dt = -(1j*w + k/2) * a - 2j * lambda_val * jx
    djx_dt = -w0 * jy
    djy_dt = w0 * jx - 4 * lambda_val * a.real * jz
    djz_dt = 4 * lambda_val * a.real * jy

    return np.array([da_dt, djx_dt, djy_dt, djz_dt])


# Define the derivatives of psc (partial second cumulant - without symmetry breaking terms)
@jit(nopython=True)
def psc_derivatives(t, state):
    a, sx, sy, sz, asx, asy, asz, sxsx, sysy, szsz, sxsy, sxsz, sysz, ada, aa = state

    lambda_val = lambda_t(t, T)

    # Coupled differential equations
    da_dt = -(1j*w + k/2) * a - 1j * (lambda_val/sqrtN) * N * sx
    dsx_dt = -w0 * sy
    dsy_dt = w0 * sx - 2 * (lambda_val/sqrtN) * (asz + np.conj(asz))
    dsz_dt = 2 * (lambda_val/sqrtN) * (asy + np.conj(asy))

    # Higher order terms
    dasx_dt = -(1j*w + k/2) * asx - 1j * (lambda_val/sqrtN) * (N - 1) * sxsx - 1j * (lambda_val/sqrtN) - w0 * asy
    dasy_dt = -(1j*w + k/2) * asy - 1j * (lambda_val/sqrtN) * (N - 1) * sxsy - (lambda_val/sqrtN) * sz \
              + w0 * asx - 2 * (lambda_val/sqrtN) * sz * (aa + ada)
    dasz_dt = -(1j*w + k/2) * asz - 1j * (lambda_val/sqrtN) * (N - 1) * sxsz


    dsxsx_dt = - 2 * w0 * sxsy
    dsysy_dt = 2 * w0 * sxsy - 4 * (lambda_val/sqrtN) * sz * (asy + np.conj(asy))
    dszsz_dt = 4 * (lambda_val/sqrtN) * sz * (asy + np.conj(asy))


    dsxsy_dt = -w0 * sysy + w0 * sxsx - 2 * (lambda_val/sqrtN) * sz * (asx + np.conj(asx))
    dsxsz_dt = -w0 * sysz
    dsysz_dt = w0 * sxsz - 4 * (lambda_val/sqrtN) * sz * (asz + np.conj(asz))

    dada_dt = -k * ada - 1j * (lambda_val/sqrtN) * N * (np.conj(asx) - asx)
    daa_dt = -2 * (1j*w + k/2) * aa - 2j * (lambda_val/sqrtN) * N * asx


    return np.array([da_dt, dsx_dt, dsy_dt, dsz_dt, dasx_dt, dasy_dt, dasz_dt, dsxsx_dt, dsysy_dt,
                     dszsz_dt, dsxsy_dt, dsxsz_dt, dsysz_dt, dada_dt, daa_dt])


# Define the derivatives of full second cumulant
@jit(nopython=True)
def fsc_derivatives(t, state):
    a, sx, sy, sz, asx, asy, asz, sxsx, sysy, szsz, sxsy, sxsz, sysz, ada, aa = state

    lambda_val = lambda_t(t, T)

    # Coupled differential equations
    da_dt = -(1j*w + k/2) * a - 1j * (lambda_val/sqrtN) * N * sx
    dsx_dt = -w0 * sy
    dsy_dt = w0 * sx - 2 * (lambda_val/sqrtN) * (asz + np.conj(asz))
    dsz_dt = 2 * (lambda_val/sqrtN) * (asy + np.conj(asy))

    # Higher order terms
    dasx_dt = -(1j*w + k/2) * asx - 1j * (lambda_val/sqrtN) * (N - 1) * sxsx - 1j * (lambda_val/sqrtN) - w0 * asy
    dasy_dt = -(1j*w + k/2) * asy - 1j * (lambda_val/sqrtN) * (N - 1) * sxsy - (lambda_val/sqrtN) * sz \
              + w0 * asx - 2 * (lambda_val/sqrtN) * (2 * a * asz + sz * aa - 2 * (a**2) * sz + a * np.conj(asz)\
              + np.conj(a) * asz + sz * ada - 2 * a * np.conj(a) * sz)
    dasz_dt = -(1j*w + k/2) * asz - 1j * (lambda_val/sqrtN) * (N - 1) * sxsz + (lambda_val/sqrtN) * sy \
              + 2 * (lambda_val/sqrtN) * (2 * a * asy + sy * aa - 2 * a * a * sy + a * np.conj(asy) \
              + np.conj(a) * asy + sy * ada - 2 * a * np.conj(a) * sy)


    dsxsx_dt = -2 * w0 * sxsy
    dsysy_dt = 2 * w0 * sxsy - 4 * (lambda_val/sqrtN) * (sy * asz + a * sysz + sz * asy - 2 * sy * a * sz\
               + sy * np.conj(asz) + np.conj(a) * sysz + sz * np.conj(asy) - 2 * sy * np.conj(a) * sz)
    dszsz_dt = 4 * (lambda_val/sqrtN) * (sz * asy + a * sysz + sy * asz - 2 * sz * a * sy + sz * np.conj(asy)\
               + np.conj(a) * sysz + sy * np.conj(asz) - 2 * sz * np.conj(a) * sy)


    dsxsy_dt = -w0 * sysy + w0 * sxsx - 2 * (lambda_val/sqrtN) * (sx * asz + a * sxsz + sz * asx - 2 * sx * sz * a\
               + sx * np.conj(asz) + np.conj(a) * sxsz + sz * np.conj(asx) - 2 * sx * sz * np.conj(a))
    dsxsz_dt = -w0 * sysz + 2 * (lambda_val/sqrtN) * (sx * asy + a * sxsy + sy * asx - 2 * sx * sy * a + sx * np.conj(asy)\
               + np.conj(a) * sxsy + sy * np.conj(asx) - 2 * sx * sy * np.conj(a))
    dsysz_dt = w0 * sxsz - 2 * (lambda_val/sqrtN) * (a * szsz + 2 * sz * asz - 2 * a * sz * sz + np.conj(a) * szsz\
               + 2 * sz * np.conj(asz) - 2 * np.conj(a) * sz * sz - 2 * sy * asy - a * sysy + 2 * a * sy * sy\
               - 2 * sy * np.conj(asy) - np.conj(a) * sysy + 2 * sy * sy * np.conj(a))

    dada_dt = -k * ada - 1j * (lambda_val/sqrtN) * N * (np.conj(asx) - asx)
    daa_dt = -2 * (1j*w + k/2) * aa - 2j * (lambda_val/sqrtN) * N * asx


    return np.array([da_dt, dsx_dt, dsy_dt, dsz_dt, dasx_dt, dasy_dt, dasz_dt,
                     dsxsx_dt, dsysy_dt, dszsz_dt, dsxsy_dt, dsxsz_dt, dsysz_dt, dada_dt, daa_dt])

# get solution based on method and integrator
def main():
    if integrator == 'rk':
        if method == 'mf':
            solution = solve_ivp(mf_derivatives, (0, t_max), initial_state, method='RK45', t_eval = t_eval, atol=atol, rtol=rtol)
        elif method == 'fsc':
            solution = solve_ivp(fsc_derivatives, (0, t_max), initial_state, method='RK45', t_eval = t_eval, atol=atol, rtol=rtol)
        elif method == 'psc':
            solution = solve_ivp(psc_derivatives, (0, t_max), initial_state, method='RK45', t_eval = t_eval, atol=atol, rtol=rtol)
    elif integrator == 'bdf':
        if method == 'mf':
            solution = solve_ivp(mf_derivatives, (0, t_max), initial_state, method='BDF', t_eval = t_eval, atol=atol, rtol=rtol)
        elif method == 'fsc':
            solution = solve_ivp(fsc_derivatives, (0, t_max), initial_state, method='BDF', t_eval = t_eval, atol=atol, rtol=rtol)
        elif method == 'psc':
            solution = solve_ivp(psc_derivatives, (0, t_max), initial_state, method='BDF', t_eval = t_eval, atol=atol, rtol=rtol)

    np.savetxt(f'filename.txt',np.column_stack((solution.t, solution.y[1])))
    return solution

if __name__ == '__main__':
    main()
