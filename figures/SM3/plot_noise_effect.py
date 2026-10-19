import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

eparr = [-0.056, 0.01]
narr=np.arange(2, 31)

for ep in eparr:
    y = []
    nplot=[]
    for n in narr:
        try:
            dat2 = np.loadtxt(f'ep={ep}_w_noise/N={n}.txt', dtype=complex)
            dat3 = np.loadtxt(f'ep={ep}_wo_noise/N={n}.txt', dtype=complex)
            y.append(np.mean(np.abs(dat2[:,2]-dat3[:,2])))
            nplot.append(n)
        except:
            continue
    plt.scatter(nplot, y)
    plt.plot(nplot, y, label=f'$\epsilon={ep}$')

plt.plot(narr, 1/narr * (1/4), 'k--', label='1/N Fit')
plt.legend(frameon=False)
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle | \Delta s_x | \rangle$', rotation=0, labelpad=20)
plt.savefig('noise_effect.png', dpi=600, bbox_inches='tight')