import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

# Set matplotlib properties for LaTeX rendering and fonts
rc('text', usetex=True)
rc('font', family='serif', serif='Computer Modern Roman', size=16)
rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')


# Define the directory where the data files are located. Update this if they are elsewhere!
dir_path = './plot_data' 

# Load Exact data
exact = np.loadtxt(os.path.join(dir_path, 'exact.txt'), dtype=complex)

# Load FSC (CE) data
ce_sx = np.loadtxt(os.path.join(dir_path, 'fsc_sx=1.0.txt'), dtype=complex)
ce_sy = np.loadtxt(os.path.join(dir_path, 'fsc_sy=1.0.txt'), dtype=complex)
ce_sz = np.loadtxt(os.path.join(dir_path, 'psc_sz=1.0.txt'), dtype=complex)

# Load TWA data
twa_sx = np.loadtxt(os.path.join(dir_path, 'twa_sx=1.0.txt'), dtype=complex)
twa_sy = np.loadtxt(os.path.join(dir_path, 'twa_sy=1.0.txt'), dtype=complex)
twa_sz = np.loadtxt(os.path.join(dir_path, 'twa_sz=1.0.txt'), dtype=complex)

lw = 2

fig, ax = plt.subplots(figsize=(6, 4.5))

# Plot FSC (CE) data
plt.plot(np.real(ce_sx[:, 0]), np.real(ce_sx[:, 1]), 'b--', lw=lw)
plt.plot(np.real(ce_sy[:, 0]), np.real(ce_sy[:, 1]), 'b-.', lw=lw)
plt.plot(np.real(ce_sz[:, 0]), np.real(ce_sz[:, 1]), 'b', lw=lw, label='FSC')

# Plot TWA data
plt.plot(np.real(twa_sx[:, 0]), np.real(twa_sx[:, 1]), 'g--', lw=lw)
plt.plot(np.real(twa_sy[:, 0]), np.real(twa_sy[:, 1]), 'g-.', lw=lw)
plt.plot(np.real(twa_sz[:, 0]), np.real(twa_sz[:, 1]), 'g', lw=lw, label='TWA')

# Plot Exact data
plt.plot(np.real(exact[:, 0]), np.real(exact[:, 1]), 'r', lw=lw, label='Exact')

# Custom line plots to generate legends with solid colors (ignoring line style)
# plt.plot([0], [-10], 'r', lw=lw, label='Exact')
# plt.plot([0], [-10], 'b', lw=lw, label='FSC')
# plt.plot([0], [-10], 'g', lw=lw, label='TWA')

# Axes settings
plt.xlabel(r'$N$', fontsize=25)
plt.ylabel(r'$\langle \hat{a}^\dagger \hat{a}\rangle/N$', fontsize=25)
plt.xlim([0, 100])
plt.ylim([0.46, 0.58])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

# Legend without frame
plt.legend(frameon=False, fontsize=22)

# Save the figure with a tight layout
plt.savefig('number_ratios.png', dpi=600, bbox_inches='tight')
