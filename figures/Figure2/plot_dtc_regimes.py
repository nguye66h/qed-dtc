import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# PRL Formatting
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', serif='Computer Modern Roman', size=16)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')

def parse_complex(s):
    return complex(s.replace('(', '').replace(')', ''))

def read_data(filepath):
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                # Column 1 is time (real part), Column 3 is magnetization (real part)
                t = parse_complex(parts[0]).real
                mag = parse_complex(parts[2]).real
                data.append((t, mag))
        return np.array(data)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return np.array([])

def get_stroboscopic_data(data, tolerance=1e-2):
    if len(data) == 0:
        return np.array([])
    
    # Filter for t approx 2*n*pi
    # Since T = 2pi, we look for t/2pi being close to an integer
    t = data[:, 0]
    cycles = t / (2 * np.pi)
    
    # Find indices where cycles are close to integers
    indices = []
    for i, c in enumerate(cycles):
        if abs(c - round(c)) < tolerance:
            indices.append(i)
            
    return data[indices]

def main():
    base_dir = '.'
    raw_data_dir = os.path.join(base_dir, 'raw_data')
    output_file = os.path.join(base_dir, 'dtc_regimes_dynamics.png')

    # Regimes Configuration
    regimes = [
        {
            'name': 'Instability',
            'ep': '-0.065',
            'N': 30,
            'ce_available': True, # N=30 missing in CE
            'label': r'\textbf{(a)} Unstable ($\epsilon=-0.065, N=30$)'
        },
        {
            'name': 'Transient',
            'ep': '-0.056',
            'N': 10,
            'ce_available': True,
            'label': r'\textbf{(b)} Transient ($\epsilon=-0.056, N=10$)'
        },
        {
            'name': 'Prethermal',
            'ep': '0.01',
            'N': 20,
            'ce_available': True, # ep=0.01 missing in CE
            'label': r'\textbf{(c)} Pre-thermal ($\epsilon=0.01, N=20$)'
        }
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True) # 2-column width approx 7 inches
    
    # Colors and Styles
    # Colors and Styles
    # Exact matches reference: color='red', linewidth=6

    # CE style - distinct from red/black
    color_ce = '#c8d6e5'
    lw_ce = 1.5
    ls_ce = '--'
    
    # TWA matches reference: color='k', linewidth=1.2
    color_twa = 'green'
    lw_twa = 3.5
    alpha_twa = 0.35

    # Making it semi-transparent so it acts as a background "shade"
    color_exact = 'red' 
    lw_exact = 1.5
    alpha_exact = 1.0

    for i, regime in enumerate(regimes):
        ax = axes[i]
        ep_val = float(regime['ep'])
        n = regime['N']
        
        # Calculate omega = 1 - |epsilon|
        # Note: The user said w = 1 - ep. Assuming ep is small deviation from pi.
        # Usually omega_DTC = 1 - |epsilon| if epsilon is deviation from pi/T.
        # Let's use w = 1 - abs(ep_val) to be safe for both signs, or just 1-ep if that's the specific definition.
        # Given ep can be negative (-0.065), 1 - (-0.065) > 1 which implies frequency > 1?
        # Usually epsilon is defined such that period becomes T / (1-epsilon).
        # Let's assume w = 1 - abs(ep_val) for the frequency scaling.
        w = 1.0 + ep_val
        
        # Paths
        exact_path = os.path.join(raw_data_dir, f'ep={regime["ep"]}_exact', f'N={n}.txt')
        twa_path = os.path.join(raw_data_dir, f'ep={regime["ep"]}_twa', f'N={n}.txt')
        # CE path structure is slightly different: ep=VALUEk=...
        # Assuming k=0.05lambda=1.0 based on directory listing
        ce_path = os.path.join(raw_data_dir, f'ep={regime["ep"]}_fsc', f'N={n}.txt')

        # Read Data
        exact_data = read_data(exact_path)
        twa_data = read_data(twa_path)
        ce_data = read_data(ce_path) if regime['ce_available'] else np.array([])

        # Normalize
        exact_data[:, 1] /= exact_data[0, 1]
        twa_data[:, 1] /= twa_data[0, 1]
        ce_data[:, 1] /= ce_data[0, 1]

        # Get Stroboscopic Data
        exact_strob = get_stroboscopic_data(exact_data)
        twa_strob = get_stroboscopic_data(twa_data)
        ce_strob = get_stroboscopic_data(ce_data)

        # Limit to wt/T approx 100
        # x = w * cycles. We want x <= 100, so cycles <= 100/w
        max_cycles = 95 / w
        
        exact_strob = exact_strob[exact_strob[:, 0] / (2*np.pi) <= max_cycles]
        twa_strob = twa_strob[twa_strob[:, 0] / (2*np.pi) <= max_cycles]
        ce_strob = ce_strob[ce_strob[:, 0] / (2*np.pi) <= max_cycles]

        # Plot
        # X-axis is w * t / pi. 
        # Original code used cycles = t / 2pi.
        # User requested x-axis: wt/T. T is usually pi or 2pi? 
        # In DTC, T is the driving period. Stroboscopic means t = nT.
        # If we plot vs wt/T, and t=nT, then x = w*n.
        # Let's assume T=2pi (based on previous code).
        # So x = w * (t / 2pi).

        # Plot CE
        if regime['ce_available']:
            x = w * (ce_strob[:, 0] / (2*np.pi))
            ax.plot(x, ce_strob[:, 1], color=color_ce, linestyle=ls_ce, label='FSC', linewidth=lw_ce, alpha=1.0)

        # Plot TWA (Thin, dark color, on top)
        x = w * (twa_strob[:, 0] / (2*np.pi))
        ax.plot(x, twa_strob[:, 1], 
                linestyle='-', color=color_twa, label='TWA', linewidth=lw_twa, alpha=alpha_twa)
            
        # Plot Exact (Thick, light color, behind)
        x = w * (exact_strob[:, 0] / (2*np.pi))
        ax.plot(x, exact_strob[:, 1], 
                linestyle='-', color=color_exact, label='Exact', linewidth=lw_exact, alpha=alpha_exact)
        
        
        # Styling
        ax.set_ylabel(r'$\langle \sigma_x (t) \rangle$', fontsize=30)
        # Move label to the bottom to avoid overlap with inset
        ax.text(0.15, 0.08, regime['label'], transform=ax.transAxes, fontsize=20,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Inset for panel (c)
        if i == 2:
            axins = ax.inset_axes([0.65, 0.55, 0.32, 0.4])
            
            if regime['ce_available']:
                x = w * (ce_strob[:, 0] / (2*np.pi))
                mask = (x >= 60) & (x <= 70)
                axins.plot(x[mask], ce_strob[mask, 1], color=color_ce, linestyle=ls_ce, linewidth=lw_ce, alpha=1.0)
            
            x = w * (twa_strob[:, 0] / (2*np.pi))
            mask = (x >= 60) & (x <= 70)
            axins.plot(x[mask], twa_strob[mask, 1], linestyle='-', color=color_twa, linewidth=lw_twa, alpha=alpha_twa)
            
            x = w * (exact_strob[:, 0] / (2*np.pi))
            mask = (x >= 60) & (x <= 70)
            axins.plot(x[mask], exact_strob[mask, 1], linestyle='-', color=color_exact, linewidth=lw_exact, alpha=alpha_exact)
            
            axins.set_xlim(60, 70)
            
            # Find min and max y in this range to set ylim
            y_min = float('inf')
            y_max = float('-inf')
            for ds in [exact_strob, twa_strob, ce_strob]:
                x = w * (ds[:, 0] / (2*np.pi))
                mask = (x >= 60) & (x <= 70)
                if np.any(mask):
                    y_min = min(y_min, np.min(ds[mask, 1]))
                    y_max = max(y_max, np.max(ds[mask, 1]))
                        
            if y_min != float('inf') and y_max != float('-inf'):
                margin = (y_max - y_min) * 0.15
                if margin == 0:
                    margin = 0.1
                axins.set_ylim(y_min - margin, y_max + margin)
                
            axins.tick_params(axis='both', which='major', labelsize=12)
            ax.indicate_inset_zoom(axins, edgecolor="black")
        
    # Legend (Horizontal, above figure)
    # Collect handles and labels from the first subplot (assuming it has all types or we manually define them)
    # To be safe, we can manually create proxy artists or just use the handles from the first plot if we know they exist.
    # Given the code, 'Instability' (i=0) has ce_available=True, so it should have all 3.
    handles, labels = axes[0].get_legend_handles_labels()
    # Sort or reorder if needed? The order of plotting is CE, Exact, TWA.
    # If we want a specific order, we can reorder.
    # Current order in plot: FSC, Exact, TWA.
    
    # Place legend above the first subplot
    axes[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), 
                   ncol=3, frameon=False, fontsize=16)

    axes[-1].set_xlabel(r'$\omega t / T$', fontsize=30)
    axes[-1].set_xlim(0, 100)
    axes[-1].set_xticks([0, 20, 40, 60, 80, 100])
    
    # Match y-axis minimum of (a) to (b) (as requested by user)
    # We assume (b) and (c) have the "perfect" minimum.
    bottom_b, _ = axes[1].get_ylim()
    axes[0].set_ylim(bottom=bottom_b)
    
    # Remove vertical space between subplots
    plt.subplots_adjust(hspace=0)
    
    plt.savefig('dtc_regimes_dynamics_lowres.png', dpi=600, bbox_inches='tight')
    #plt.savefig('dtc_regimes_dynamics.eps', bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    main()
