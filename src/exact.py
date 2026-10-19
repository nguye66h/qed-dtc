import qutip
import os
from qutip import *
from qutip.piqs import *
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Create the parser
parser = argparse.ArgumentParser()

# Add arguments

## regime parameters
parser.add_argument('ep', type=float, help="epsilon")
parser.add_argument('k', type=float, help="kappa cavity loss rate")
parser.add_argument('type_of_coupling', type=str, help="write per if percentage, lambda if absolute val")
parser.add_argument('lambda_', type=float, help="coupling strength")
parser.add_argument('per', type=float, help="percentage of coupling strength to critical coupling")
parser.add_argument('N', type=int, help="Number of spins N")

## initial conditions
parser.add_argument('a_init', type=str, help="initial value for a, string, sqrtN possible")
parser.add_argument('sx_init', type=float, help="initial value for sigma_x in second cumulant or jx in mean field")
parser.add_argument('sy_init', type=float, help="initial value for sigma_y in second cumulant or jy in mean field")
parser.add_argument('sz_init', type=float, help="initial value for sigma_z in second cumulant or jz in mean field")

## simulation details
parser.add_argument('tmax', type=float, help="max simulation time")

## Extra details for PIQS
parser.add_argument('nphoton', type=int, help="max photon number")
parser.add_argument('evolving', type=str, help="if prop from init state, write True, otherwise False")
parser.add_argument('dt', type=float, help="timestep dt")
parser.add_argument('modulated', type=str, help="if modulated, write True, otherwise False")
parser.add_argument('ntraj', type=int, help="number of trajectories, integer")
parser.add_argument('i_', type=int, help="index of parallel tasks")

# Parse the arguments
args = parser.parse_args()

# parameters to set
N = args.N
nphot = args.nphoton
ε = args.ep
wT = 1.0
w0 = (1+ε)*wT  # Atomic frequency
w  = (1-ε)*wT  # Photonic frequency

kappa = args.k
gc = 0.5*np.sqrt((w0/w)*(w**2+kappa**2/4))
modulated = args.modulated
if args.type_of_coupling == 'per':
    ratio_g = args.per * gc
else:
    ratio_g = args.lambda_

# processing parameters
nspin = N
nds = N+1
[jx, jy, jz] = jspin(N)
jx, jy, jz = Qobj(jx[:nds,:nds]), Qobj(jy[:nds,:nds]), Qobj(jz[:nds,:nds])
jp = jx + 1j*jy
jm = jx - 1j*jy
g = ratio_g*2/np.sqrt(N)
T = 2*np.pi

#TLS liouvillian
systemhamiltonian = w0 * jz
hsys = tensor(qeye(nphot), systemhamiltonian)

#photonic liouvilian
a = destroy(nphot)
hphot_sub = w * a.dag() * a
hphot = tensor(hphot_sub, qeye(nds))

# Time-dependent lambda(t)
def g_t(t, args):
    if t % T < T / 2:
        return g
    else:
        if modulated == "True":
            return 0
        else:
            return g

# light-matter superoperator
hint = tensor(a + a.dag(), jx)

# total hamiltonian
hamil_tot = [hsys, [hint, g_t], hphot]
c_ops = [tensor(np.sqrt(kappa) * a, qeye(nds))]

# total operators
# number average here is NOT renormalized by N (as can be seen in the form of operator)
jx_tot = tensor(qeye(nphot), jx)
jy_tot = tensor(qeye(nphot), jy)
jz_tot = tensor(qeye(nphot), jz)
nphot_tot = tensor(a.dag()*a, qeye(nds))
a_tot = tensor(a, qeye(nds))

if args.sx_init != 0:
    _, U = np.linalg.eigh(jx)
    indval = round(args.sx_init)
    initspin = "sx"
elif args.sy_init != 0:
    _, U = np.linalg.eigh(jy)
    indval = round(args.sy_init)
    initspin = "sy"
elif args.sz_init != 0:
    _, U = np.linalg.eigh(jz)
    indval = round(args.sz_init)
    initspin = "sz"

if indval == 1:
    myind = -1
elif indval == -1:
    myind = 0

if round(args.ntraj) == 0:

    # Path where you want to save the file
    directory = f'foldername'
    filename = f'me-results.txt'
    full_path = os.path.join(directory, filename)

    os.makedirs(directory, exist_ok=True)

    # initial state
    wf_phot = basis(nphot, round(float(args.a_init)))
    wf_0 = tensor(wf_phot, Qobj(U[:,myind]))

    # Define time evolution parameters
    times = 2*np.pi * np.arange(0,args.tmax+args.dt,args.dt)  # Array of time points to evolve the system

    # Run the time evolution
    result = mesolve(hamil_tot, wf_0, times, c_ops, [a_tot, jx_tot, jy_tot, jz_tot, nphot_tot])

    # find expectation values
    a_t =  result.expect[0]
    jx_t = result.expect[1]
    jy_t = result.expect[2]
    jz_t = result.expect[3]
    nphot_t = result.expect[4]

    # Save the data in text format
    np.savetxt(full_path, np.column_stack((times, a_t, jx_t, jy_t, jz_t, nphot_t)))

else:

    # Path where you want to save the file
    directory = f'foldername'
    filename = f'i={args.i_}.txt'
    full_path = os.path.join(directory, filename)

    os.makedirs(directory, exist_ok=True)

    # initial state
    wf_phot = basis(nphot, round(float(args.a_init)))
    wf_0 = tensor(wf_phot, Qobj(U[:,myind]))

    # Define time evolution parameters
    times = 2*np.pi * np.arange(0,args.tmax+args.dt,args.dt)  # Array of time points to evolve the system

    # Run the time evolution
    result = mcsolve(hamil_tot, wf_0, times, c_ops, [a_tot, jx_tot, jy_tot, jz_tot, nphot_tot], ntraj=args.ntraj)

    # find expectation values
    a_t =  result.expect[0]
    jx_t = result.expect[1]
    jy_t = result.expect[2]
    jz_t = result.expect[3]
    nphot_t = result.expect[4]

    # Save the data in text format
    np.savetxt(full_path, np.column_stack((times, a_t, jx_t, jy_t, jz_t, nphot_t)))
