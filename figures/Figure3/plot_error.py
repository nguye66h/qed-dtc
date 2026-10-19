import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


diff1 = np.zeros((len(np.arange(2, 31)),4))


for n in np.arange(2,31):

    dat1 = np.loadtxt(f'raw_data/ep=-0.056_exact/N={n}.txt', dtype=complex)
    dat2 = np.loadtxt(f'raw_data/ep=-0.056_twa/N={n}.txt', dtype=complex)
    diff1[n-2,0] = np.mean(np.abs(dat1[0::10,2][:-1]/(n/2) - dat2[00::100,2]))

    dat3 = np.loadtxt(f'raw_data/ep=0.01_exact/N={n}.txt', dtype=complex)
    dat4 = np.loadtxt(f'raw_data/ep=0.01_twa/N={n}.txt', dtype=complex)
    diff1[n-2,1] = np.mean(np.abs(dat3[0::10,2][:-1]/(n/2) - dat4[0::100,2]))

    dat5 = np.loadtxt(f'raw_data/ep=0.051_exact/N={n}.txt', dtype=complex)
    dat6 = np.loadtxt(f'raw_data/ep=0.051_twa/N={n}.txt', dtype=complex)
    diff1[n-2,2] = np.mean(np.abs(dat5[0::10,2][:-1]/(n/2) - dat6[0::100,2]))

    dat7 = np.loadtxt(f'raw_data/ep=0.11_exact/N={n}.txt', dtype=complex)
    dat8 = np.loadtxt(f'raw_data/ep=0.11_twa/N={n}.txt', dtype=complex)
    diff1[n-2,3] = np.mean(np.abs(dat7[0::10,2][:-1]/(n/2) - dat8[0::100,2]))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')

plt.plot(np.arange(2, 31),diff1[:,0], linestyle='-', marker='o', markersize=4, color='red', label=r'$\epsilon=-0.056$')
plt.plot(np.arange(2, 31),diff1[:,1], linestyle='--', marker='s', markersize=4, color='blue', label=r'$\epsilon=0.01$')
plt.plot(np.arange(2, 31),diff1[:,2], linestyle=':', marker='x', markersize=6, color='#30336b', linewidth=3, label=r'$\epsilon=0.051$')
plt.plot(np.arange(2, 31),diff1[:,3], linestyle='-.', marker='^', markersize=4, color='green', label=r'$\epsilon=0.11$')
plt.grid(True, linestyle='--', alpha=0.6) # Add subtle grid lines
plt.xlim(2, 30) # Set x-axis limits
plt.ylim(0, 0.20) # Ensure y-axis starts at 0
plt.axhline(0.05, color='k', linewidth=2.0)

ax = plt.gca()

# Apply MaxNLocator to the y-axis to force integer ticks
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

# Apply MaxNLocator to the x-axis if desired
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))


plt.xlabel(r'$N$', fontsize=30)
plt.ylabel(r'Average $|\Delta \langle \sigma_x \rangle_s|$', fontsize=30)
plt.legend(frameon=False, fontsize=20)
plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=20)
plt.tight_layout()
plt.savefig('error_plot.png', dpi=600, bbox_inches='tight')
plt.savefig('error_plot.pdf', bbox_inches='tight')
plt.savefig('error_plot.eps', bbox_inches='tight')