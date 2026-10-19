import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib import rc
from scipy.optimize import curve_fit
import os
import glob
import re
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

# --- Global Configuration ---
rc('text', usetex=True)
rc('font', family='serif', serif='Computer Modern Roman', size=32)
rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')
plt.rcParams['hatch.linewidth'] = 2.0

BASE_DATA_DIR = './' # Assuming scripts are running from code/

# --- Helper Functions ---

def single_exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def read_and_process_signal(file_path, n_val):
    if not os.path.exists(file_path):
        return None, None

    try:
        dat = np.loadtxt(file_path, dtype=complex)
    except Exception:
        return None, None

    if dat.shape[0] == 0: return None, None
    norm_val = dat[0, 2].real
    if norm_val == 0: return None, None

    # Sampling Logic
    if n_val < 20:
        # Exact
        start_index = 100
        if dat.shape[0] > 60 and dat[60, 2].real < 0:
            start_index = 110
        step = 20
    else:
        # TWA
        start_index = 1000
        if dat.shape[0] > 600 and dat[600, 2].real < 0:
            start_index = 1100
        step = 200

    # if dat[start_index, 2].real / norm_val < 0.05:
    #     return None, None

    if dat.shape[0] <= start_index:
        return None, None

    time_axis = dat[start_index::step, 0].real
    signal_data = dat[start_index::step, 2].real / norm_val

    if len(time_axis) == 0:
        return None, None

    # Enforce Positive Points Only
    positive_mask = signal_data > 0
    if not np.all(positive_mask):
        first_non_pos = np.argmax(~positive_mask)
        if first_non_pos == 0:
            return None, None
        time_axis = time_axis[:first_non_pos]
        signal_data = signal_data[:first_non_pos]

    if len(signal_data) < 3:
        return None, None

    return signal_data, time_axis

def fit_decay(signal, t):
    t_normalized = t - t[0]
    initial_amplitude_guess = signal[0]
    tau_duration = t_normalized[-1] - t_normalized[0]
    initial_tau_guess = tau_duration / 2 if tau_duration > 0 else 1.0
    p0 = [max(1e-6, initial_amplitude_guess), max(1e-6, initial_tau_guess)]

    try:
        params, _ = curve_fit(
            single_exp_decay, t_normalized, signal, p0=p0, bounds=(0, np.inf),
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000
        )
        return params[0], params[1]
    except:
        return np.nan, np.nan

def get_mean_amplitude(file_path, is_csv=False, n_val=None):
    if not os.path.exists(file_path):
        return None

    try:
        if is_csv:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=complex)
            if data.size == 0: return None
            jx = data[:, 2]
            sx = 2 * np.abs(jx)
            if sx[0] != 0:
                sx = sx / sx[0]
            mean_amp = np.mean(sx[-1000:])
            return mean_amp

        else:
            data = np.loadtxt(file_path, dtype=complex)
            if data.size == 0: return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            jx = data[:, 2]
            sx = 2 * np.abs(jx)
            if sx[0] != 0:
                sx = sx / sx[0]
            
            if n_val is None: return None
            
            if n_val < 20:
                step = 20
                half_period = 10
                base_offset = 60
            else:
                step = 200
                half_period = 100
                base_offset = 600
                
            total_points = len(sx)
            required_points = 100 * step
            
            # Calculate start_idx aligned with base_offset + m * step
            raw_start_idx = total_points - required_points
            if raw_start_idx < base_offset:
                 start_idx = base_offset
            else:
                 m = (raw_start_idx - base_offset) // step
                 start_idx = base_offset + m * step
                 
            if start_idx >= total_points:
                return None
                
            jx_real = data[:, 2].real
            if jx_real[0] != 0:
                jx_real_norm = jx_real / np.abs(jx_real[0])
            else:
                jx_real_norm = jx_real
                
            sampled_jx = jx_real_norm[start_idx::step]
            
            if np.mean(sampled_jx) < 0:
                start_idx += half_period
                sampled_jx = jx_real_norm[start_idx::step]
                
            mean_val = np.mean(sampled_jx)
            return np.abs(mean_val)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Generators ---

def generate_floquet_heatmap(ax):
    print("Generating Floquet Heatmap...")
    data_file = os.path.join(BASE_DATA_DIR, 'floquet_mul_heatmap_data.txt')
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    data = np.loadtxt(data_file, dtype=complex)
    dat = np.zeros((data.shape[0],3))
    dat[:,0] = data[:,0].real
    dat[:,0] = np.round(dat[:,0],4)
    dat[:,1] = data[:,1].real
    dat[:,1] = np.round(dat[:,1],4)
    for i in range(0, data.shape[0]):
        dat[i,2] = data[i,np.argmax(np.abs(data[i,2:6]))+2].real

    df = pd.DataFrame(dat, columns=['epsilon', 'kappa', 'max_mul'])
    heatmap_data = df.pivot(index='kappa', columns='epsilon', values='max_mul')

    colors = ["#c23616", "#ffffff", "#0fbcf9"] # Dark Red -> White -> Medium Blue
    asym_cmap = mcolors.LinearSegmentedColormap.from_list("AsymRdBu", colors, N=256)

    sns.heatmap(heatmap_data, 
                ax=ax,
                annot=False, 
                cmap=asym_cmap,
                center=-1,
                vmin=-1-0.12,
                vmax=-1+0.12,
                xticklabels=10,
                yticklabels=4,
                linewidths=0,
                rasterized=True,
                cbar=False 
    )
    
    # Add colorbar manually to match style
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    # We need a mappable
    mappable = ax.collections[0]
    cbar = plt.colorbar(mappable, cax=cax)
    
    ax.set_xticklabels(['{:.2f}'.format(float(t)) for t in np.array([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15])])
    ax.set_yticklabels(['{:.2f}'.format(float(t)) for t in np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1])], rotation=0)
    
    ax.set_xlabel(r'$\epsilon$', fontsize=70)
    ax.set_ylabel(r'$\kappa$', fontsize=70, rotation='horizontal', va='center', labelpad=20)
    
    # Rectangle patch
    x_start = 8
    y_start = 10
    width = 35
    height = 1
    rect = patches.Rectangle(
        (x_start, y_start), width, height,
        edgecolor='red', facecolor='none', lw=2 # Increased lw for visibility
    )
    ax.grid(False)
    ax.add_patch(rect)
    ax.invert_yaxis()

def generate_rate_heatmap(ax):
    print("Generating Rate Heatmap...")
    EP_VALUES = np.round(np.arange(-0.06, 0.12, 0.01), 2)
    data_points = []
    
    # Try to load saved data first to save time? 
    # User said "copy the content...". I'll re-run processing to be safe/compliant.
    
    for ep_val_float in EP_VALUES:
        ep_str = f"{ep_val_float:.2f}"
        dir = os.path.join(BASE_DATA_DIR, f'plot_data/ep={ep_str}')
        files = glob.glob(os.path.join(dir, 'N=*.txt'))
        files = list(set(files))
        
        for f_path in files:
            match = re.search(r'N=(\d+)', os.path.basename(f_path))
            if not match: continue
            n_val = int(match.group(1))
            if n_val == 1: continue
            
            signal, t = read_and_process_signal(f_path, n_val)
            if signal is None: continue
            
            A, tau = fit_decay(signal, t)
            if not np.isnan(tau) and tau > 0:
                inv_lifetime = (1.0 / tau) / (1 + ep_val_float)
                data_points.append((n_val, ep_val_float, inv_lifetime))

    dtype = [('n', int), ('ep', float), ('val', float)]
    data = np.array(data_points, dtype=dtype)
    n_values = np.unique(data['n'])
    ep_values = EP_VALUES # Use full range to show missing data
    heatmap_grid = np.full((len(ep_values), len(n_values)), np.nan)

    for n_idx, n in enumerate(n_values):
        for ep_idx, ep in enumerate(ep_values):
            mask = (data['n'] == n) & (np.isclose(data['ep'], ep))
            vals = data['val'][mask]
            if len(vals) > 0:
                if len(vals) > 1:
                    print(f"Multiple values found for N={n}, ep={ep}: {vals} in rate heatmap")
                heatmap_grid[ep_idx, n_idx] = np.mean(vals)

    cmap = copy.copy(plt.cm.magma)
    cmap.set_bad(color='white')

    ep_step = ep_values[1] - ep_values[0]
    ep_edges = np.concatenate([ep_values - ep_step/2, [ep_values[-1] + ep_step/2]])
    
    log_n = np.log(n_values)
    log_midpoints = (log_n[:-1] + log_n[1:]) / 2
    d_log_start = log_midpoints[0] - log_n[0]
    log_start = log_n[0] - d_log_start
    d_log_end = log_n[-1] - log_midpoints[-1]
    log_end = log_n[-1] + d_log_end
    log_edges = np.concatenate([[log_start], log_midpoints, [log_end]])
    n_edges = np.exp(log_edges)

    X, Y = np.meshgrid(ep_edges, n_edges, indexing='ij')

    im = ax.pcolormesh(X, Y, heatmap_grid, cmap=cmap, norm=LogNorm(vmin=10**-6, vmax=10**-1))

    hatch_mask = np.where(np.isnan(heatmap_grid), 1, np.nan)

    ax.pcolor(X, Y, hatch_mask, hatch='xx', edgecolors='k', linewidth=0, cmap=ListedColormap(['none']), zorder=10)

    ax.set_yscale('log')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax, ticks=[10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])
    cbar.outline.set_visible(True)

    y_line = np.sqrt(17 * 23)
    ax.axhline(y=y_line, color='white', linestyle='-', linewidth=2)

    ax.set_xlabel(r'$\epsilon$', fontsize=70)
    ax.set_ylabel(r'$N$', fontsize=70, rotation='horizontal', va='center', labelpad=20)

def generate_amplitude_heatmap(ax):
    print("Generating Amplitude Heatmap...")
    PLOT_DATA_DIR = os.path.join(BASE_DATA_DIR, 'plot_data')
    RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, 'raw_data')
    EP_VALUES = np.round(np.arange(-0.04, 0.05, 0.01), 2)
    
    data_points = []
    
    for ep_val in EP_VALUES:
        ep_str = f"{ep_val:.2f}"
        mf_file = os.path.join(PLOT_DATA_DIR, f'ep={ep_str}_mf_10-9.csv')
        mf_amp = get_mean_amplitude(mf_file, is_csv=True)
        
        if mf_amp is None or mf_amp == 0: continue
            
        ep_dir = os.path.join(PLOT_DATA_DIR, f'ep={ep_str}')
        raw_ep_dir = os.path.join(RAW_DATA_DIR, f'ep={ep_str}_twa')
        
        files = []
        # N <= 1000 from plot_data
        if os.path.exists(ep_dir):
            plot_files = glob.glob(os.path.join(ep_dir, 'N=*.txt'))
            for f in plot_files:
                match = re.search(r'N=(\d+)', os.path.basename(f))
                if match and int(match.group(1)) <= 1000 and int(match.group(1)) >= 10:
                    files.append(f)
                    
        # N > 1000 from raw_data
        if os.path.exists(raw_ep_dir):
            raw_files = glob.glob(os.path.join(raw_ep_dir, 'N=*.txt'))
            for f in raw_files:
                match = re.search(r'N=(\d+)', os.path.basename(f))
                if match and int(match.group(1)) > 1000:
                    files.append(f)
        
        processed_ns = set()
        for f_path in files:
            match = re.search(r'N=(\d+)', os.path.basename(f_path))
            if not match: continue
            n_val = int(match.group(1))
            if n_val == 1: continue
            if n_val in processed_ns: continue
            processed_ns.add(n_val)
            
            sim_amp = get_mean_amplitude(f_path, is_csv=False, n_val=n_val)
                
            ratio = sim_amp / mf_amp
            data_points.append((n_val, ep_val, ratio))


    dtype = [('n', int), ('ep', float), ('val', float)]
    data = np.array(data_points, dtype=dtype)
    n_values = np.unique(data['n'])
    ep_values = EP_VALUES # Use full range to show missing data
    
    # Swapped axes: N on x, Ep on y
    heatmap_grid = np.full((len(n_values), len(ep_values)), np.nan)

    for n_idx, n in enumerate(n_values):
        for ep_idx, ep in enumerate(ep_values):
            mask = (data['n'] == n) & (np.isclose(data['ep'], ep))
            vals = data['val'][mask]
            if len(vals) > 0:
                if len(vals) > 1:
                    print(f"Multiple values found for N={n}, ep={ep}: {vals} in amplitude heatmap")
                heatmap_grid[n_idx, ep_idx] = np.mean(vals)

    cmap = copy.copy(plt.cm.viridis_r)
    cmap.set_bad(color='white')

    ep_step = ep_values[1] - ep_values[0]
    ep_edges = np.concatenate([ep_values - ep_step/2, [ep_values[-1] + ep_step/2]])
    
    log_n = np.log(n_values)
    log_midpoints = (log_n[:-1] + log_n[1:]) / 2
    d_log_start = log_midpoints[0] - log_n[0]
    log_start = log_n[0] - d_log_start
    d_log_end = log_n[-1] - log_midpoints[-1]
    log_end = log_n[-1] + d_log_end
    log_edges = np.concatenate([[log_start], log_midpoints, [log_end]])
    n_edges = np.exp(log_edges)

    X, Y = np.meshgrid(n_edges, ep_edges, indexing='ij')

    im = ax.pcolormesh(X, Y, heatmap_grid, cmap=cmap, vmin=0, vmax=1.0)
    
    hatch_mask = np.where(np.isnan(heatmap_grid), 1, np.nan)

    ax.pcolor(X, Y, hatch_mask, hatch='xx', edgecolors='k', linewidth=0, cmap=ListedColormap(['none']), zorder=10)


    ax.set_xscale('log')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.outline.set_visible(True)
    
    x_line = np.sqrt(17 * 23)
    ax.axvline(x=x_line, color='red', linestyle='-', linewidth=2)

    ax.set_xlabel(r'$N$', fontsize=70)
    ax.set_ylabel(r'$\epsilon$', fontsize=70, rotation='horizontal', va='center', labelpad=20)

# --- Main Execution ---

def main():
    # Create figure with 3 subplots
    # Removed constrained_layout=True to allow manual spacing
    fig, axes = plt.subplots(1, 3, figsize=(40, 8))
    plt.subplots_adjust(wspace=0.5) # Add more space between subplots
    
    # Generate Heatmaps
    generate_floquet_heatmap(axes[0])
    generate_rate_heatmap(axes[1])
    generate_amplitude_heatmap(axes[2])
    
    # Add labels (a), (b), (c)
    labels = [r'$\textbf{(a)}$', r'$\textbf{(b)}$', r'$\textbf{(c)}$']
    for i, ax in enumerate(axes):
        # Force ylabel position to align panel labels
        # Standard position is roughly x=-0.25 for these font sizes
        ax.yaxis.set_label_coords(-0.18, 0.5)
        
        # Position panel label at same x, top y
        ax.text(-0.15, 1.02, labels[i], transform=ax.transAxes, 
                fontsize=60, fontweight='bold', va='bottom', ha='right')
        
        # Ensure tick label sizes are consistent
        ax.tick_params(axis='both', which='major', labelsize=32)

    output_file = os.path.join(BASE_DATA_DIR, 'combined_heatmap_figure.png')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', transparent=False)
    print(f"Combined figure saved to {output_file}")

if __name__ == "__main__":
    main()
