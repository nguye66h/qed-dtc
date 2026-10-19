import numpy as np
from numpy.linalg import eigvals
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import root

# ----------------------------
# Model & parameters
# ----------------------------
class Params:
    def __init__(self, omega_c=1.0, omega=1.0, kappa=0.05,
                 lam0=1.0, N=100, T=2*np.pi, duty=0.5):
        self.omega_c = float(omega_c)
        self.omega   = float(omega)
        self.kappa   = float(kappa)
        self.lam0    = float(lam0)
        self.N       = int(N)
        self.T       = float(T)
        self.duty    = float(duty)

def square_lambda(t, p: Params):
    # 50/50 by default; use duty in (0,1) for asymmetric drive
    tmod = (t % p.T) / p.T
    return p.lam0 if tmod < p.duty else 0

# Jacobian J = ∂f/∂x (needed for variational eqs / Floquet)
def jacobian(lam, p: Params):
    rtN = np.sqrt(p.N)

    J = np.zeros((4,4), dtype=float)
    sz = -p.omega * (p.kappa**2/4 + p.omega_c**2) / (4 * p.lam0**2 * p.omega_c)
    # d/d[a_r, a_i, s_x, s_y] of each component:
    # da_r/dt
    J[0,0] = -p.kappa/2.0
    J[0,1] = +p.omega_c
    # da_i/dt
    J[1,0] = -p.omega_c
    J[1,1] = -p.kappa/2.0
    J[1,2] = -lam*rtN
    # ds_x/dt
    J[2,3] = -p.omega
    # ds_y/dt
    J[3,2] = +p.omega
    J[3,0] = -(4.0*lam/rtN)*sz
    return J

def monodromy_matrix(p: Params):
    return expm(jacobian(p.lam0, p) * np.pi) @ expm(jacobian(0, p) * np.pi)


def floquet_multipliers(p: Params):
    Phi = monodromy_matrix(p)
    mu = eigvals(Phi)
    return mu, Phi

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    ε_arr = np.arange(-0.1, 0.155, 0.005)
    kappa_arr = np.arange(0, 0.105, 0.005)
    for ε in ε_arr:
        for κ in kappa_arr:
            p = Params(omega_c=1.0-ε, omega=1.0+ε, kappa=κ,
                    lam0=1.0, N=100, T=2*np.pi, duty=0.5)

            # 2) Floquet multipliers at that orbit (optional)
            mu, Phi = floquet_multipliers(p)
            print("epsilon, kappa:", ε, κ)
            print("Floquet multipliers:", mu)
            with open("floquet_mul_heatmap_data_new.txt", "a") as filename:
                np.savetxt(filename, np.column_stack((ε,κ,*mu)))
