import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

# Configuration
BASE_DIR = Path(r'e:\PaperV2\图件pro')
DATA_DIR = BASE_DIR / 'data' / 'raw'
OUTPUT_DIR = BASE_DIR / 'selected_dam_plots'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Data
df_main = pd.read_csv(r'e:\PaperV2\Other\1109备份最终数据.csv')
df_climate = pd.read_csv(r'e:\PaperV2\Other\dam_data_with_climate.csv')

# Merge Climate Data
# Try merging on DAM_NAME first, then ID/GDW_ID if needed
# df_main has GDW_ID, df_climate has ID
df_merged = pd.merge(df_main, df_climate[['ID', 'Cli_Zone', 'CliTypeCHN']], left_on='GDW_ID', right_on='ID', how='left')

# Country to Continent Mapping
country_to_continent = {
    'Brazil': 'South America', 'United States': 'North America', 'China': 'Asia', 
    'Algeria': 'Africa', 'Argentina': 'South America', 'Afghanistan': 'Asia', 
    'Australia': 'Oceania', 'Austria': 'Europe', 'Azerbaijan': 'Asia', 
    'Belgium': 'Europe', 'Bolivia': 'South America', 'Brunei': 'Asia', 
    'Bulgaria': 'Europe', 'Canada': 'North America', 'Chile': 'South America', 
    'Colombia': 'South America', 'Croatia': 'Europe', 'Cuba': 'North America', 
    'Cyprus': 'Europe', 'Czechia': 'Europe', 'Dominican Republic': 'North America', 
    'Ecuador': 'South America', 'Ethiopia': 'Africa', 'France': 'Europe', 
    'Iran': 'Asia', 'Mexico': 'North America', 'Philippines': 'Asia'
}

df_merged['Continent'] = df_merged['COUNTRY'].map(country_to_continent)

# Filter Groups
# Group A: Seepage FOS (F19_max) > Normal FOS (F8_max)
# Group B: Seepage FOS (F19_max) < Normal FOS (F8_max)
group_a = df_merged[df_merged['F19_max'] > df_merged['F8_max']]
group_b = df_merged[df_merged['F19_max'] < df_merged['F8_max']]

print(f"Group A (Seepage > Normal): {len(group_a)} dams")
print(f"Group B (Seepage < Normal): {len(group_b)} dams")

# Selection Logic
selected_dams = []

# We need 1 from Group B
# Try to pick one with good data and distinct continent/climate if possible
# Let's just pick the first one for now, or random
if not group_b.empty:
    dam_b = group_b.iloc[0]
    selected_dams.append(dam_b)
    print(f"Selected from Group B: {dam_b['DAM_NAME']} ({dam_b['Continent']}, {dam_b['Cli_Zone']})")
else:
    print("Warning: No dams found for Group B!")

# We need 4 from Group A
# Try to maximize diversity in Continent and Climate Zone
# Exclude the continent/climate of the dam selected from Group B if possible
used_continents = {d['Continent'] for d in selected_dams}
used_climates = {d['Cli_Zone'] for d in selected_dams}

candidates_a = group_a.copy()

# Simple greedy selection for diversity
for _ in range(4):
    if candidates_a.empty:
        break
    
    # Score candidates: +1 for new continent, +1 for new climate
    scores = []
    for idx, row in candidates_a.iterrows():
        score = 0
        if row['Continent'] not in used_continents:
            score += 2 # Prioritize continent
        if row['Cli_Zone'] not in used_climates:
            score += 1
        scores.append((score, idx))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    best_idx = scores[0][1]
    best_dam = candidates_a.loc[best_idx]
    
    selected_dams.append(best_dam)
    used_continents.add(best_dam['Continent'])
    used_climates.add(best_dam['Cli_Zone'])
    
    # Remove selected
    candidates_a = candidates_a.drop(best_idx)

# Manual fix for Dry Falls (ID 106) if selected
for d in selected_dams:
    if d['GDW_ID'] == 106 and pd.isna(d['Cli_Zone']):
        d['Cli_Zone'] = 'D' # Verified via raster lookup
        d['Continent'] = 'North America' # Ensure continent is set

print("Selected Dams:")
for d in selected_dams:
    print(f"- {d['DAM_NAME']} (ID: {d['GDW_ID']}): {d['Continent']}, {d['Cli_Zone']}, Normal FOS={d['F8_max']}, Seepage FOS={d['F19_max']}")

# Plotting Parameters
# Each figure shows one parameter for all 5 dams
# Layout: 5 rows x 2 columns (Normal left, Seepage right)
parameter_groups = [
    ('F6', 'F17', 'Pressure Head', 'm', 'pressure_head'),
    ('F7', 'F18', 'Displacement', 'm', 'displacement'),
    ('F9', 'F20', 'Resistivity', 'Ωm', 'resistivity'),
    ('F10', 'F21', 'SP Potential', 'mV', 'sp_potential'),
    (None, 'F22', 'Seepage Electric Field Potential', 'mV', 'electric_field')
]

def read_csv_data(dam_name, f_code):
    if f_code is None:
        return None
    
    filename = f"{dam_name}_{f_code}.csv"
    filepath = DATA_DIR / f_code / filename
    
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return None
        
    try:
        data = pd.read_csv(filepath, header=None, names=['x', 'y', 'value'])
        return data
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

# Custom colormap: fresh style with red-yellow-blue-green
from matplotlib.colors import LinearSegmentedColormap
colors = ['#2E86AB', '#06A77D', '#D5C67A', '#F77F00', '#D62828']  # Blue-Green-Yellow-Orange-Red
custom_cmap = LinearSegmentedColormap.from_list('fresh', colors)

# Generate one figure per parameter
for param_idx, (f_code_left, f_code_right, param_name, unit, file_suffix) in enumerate(parameter_groups):
    print(f"\nGenerating figure for {param_name}...")
    
    # Determine if this is a two-column or one-column layout
    has_both_columns = f_code_left is not None and f_code_right is not None
    has_only_right = f_code_left is None and f_code_right is not None
    
    # Collect all data to find global vmin/vmax for unified colorbar
    # Also collect x/y ranges for consistent axis limits
    all_values = []
    all_x_ranges = []
    all_y_ranges = []
    temp_data = []
    for dam in selected_dams:
        dam_name = dam['DAM_NAME']
        if f_code_left:
            data = read_csv_data(dam_name, f_code_left)
            if data is not None and len(data) > 0:
                all_values.extend(data['value'].values)
                all_x_ranges.extend([data['x'].min(), data['x'].max()])
                all_y_ranges.extend([data['y'].min(), data['y'].max()])
                temp_data.append(data)
        if f_code_right:
            data = read_csv_data(dam_name, f_code_right)
            if data is not None and len(data) > 0:
                all_values.extend(data['value'].values)
                all_x_ranges.extend([data['x'].min(), data['x'].max()])
                all_y_ranges.extend([data['y'].min(), data['y'].max()])
    
    if len(all_values) > 0:
        vmin, vmax = np.percentile(all_values, [2, 98])  # Use percentile to avoid outliers
    else:
        vmin, vmax = 0, 1
    
    # Calculate global x/y limits for this parameter across all dams
    if len(all_x_ranges) > 0:
        global_x_min, global_x_max = min(all_x_ranges), max(all_x_ranges)
        global_y_min, global_y_max = min(all_y_ranges), max(all_y_ranges)
    else:
        global_x_min, global_x_max = 0, 1
        global_y_min, global_y_max = 0, 1
    
    if has_both_columns:
        fig, axes = plt.subplots(5, 2, figsize=(10, 11))
    else:
        fig, axes = plt.subplots(5, 1, figsize=(5, 11))
        axes = axes.reshape(-1, 1)  # Make it 2D for consistent indexing
    
    # Plot each dam
    for dam_idx, dam in enumerate(selected_dams):
        dam_name = dam['DAM_NAME']
        continent = dam['Continent']
        climate = dam['Cli_Zone']
        
        # Label for this row
        row_label = chr(ord('a') + dam_idx)
        
        # Left column - Normal state
        if f_code_left is not None:
            ax_left = axes[dam_idx, 0] if has_both_columns else None
            if ax_left is not None:
                data_norm = read_csv_data(dam_name, f_code_left)
                if data_norm is not None and len(data_norm) > 0:
                    try:
                        # Use scatter plot - shows exact data points without interpolation
                        sc = ax_left.scatter(data_norm['x'], data_norm['y'], c=data_norm['value'], 
                                           s=1, cmap=custom_cmap, vmin=vmin, vmax=vmax, 
                                           marker='s', edgecolors='none', rasterized=True)
                        ax_left.set_title(f'{dam_name}', fontsize=8, pad=2)
                        ax_left.set_aspect('equal')
                        ax_left.set_xlim(global_x_min, global_x_max)
                        ax_left.set_ylim(global_y_min, global_y_max)
                        ax_left.tick_params(labelsize=6)
                        ax_left.set_xlabel('X (m)', fontsize=6)
                        ax_left.set_ylabel('Y (m)', fontsize=6)
                        # Add label (a), (b), etc. on the left top corner
                        ax_left.text(0.02, 0.98, f'({row_label})', transform=ax_left.transAxes, 
                                   fontsize=10, fontweight='bold', va='top', ha='left')
                    except Exception as e:
                        ax_left.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                                   ha='center', va='center', transform=ax_left.transAxes, fontsize=7)
                        ax_left.set_title(f'{dam_name}', fontsize=8, pad=2)
                        ax_left.text(0.02, 0.98, f'({row_label})', transform=ax_left.transAxes, 
                                   fontsize=10, fontweight='bold', va='top', ha='left')
                else:
                    ax_left.text(0.5, 0.5, 'No data', 
                               ha='center', va='center', transform=ax_left.transAxes, fontsize=7)
                    ax_left.set_title(f'{dam_name}', fontsize=8, pad=2)
                    ax_left.text(0.02, 0.98, f'({row_label})', transform=ax_left.transAxes, 
                                   fontsize=10, fontweight='bold', va='top', ha='left')
        
        # Right column - Seepage state
        if f_code_right is not None:
            ax_right = axes[dam_idx, 1] if has_both_columns else axes[dam_idx, 0]
            data_seep = read_csv_data(dam_name, f_code_right)
            if data_seep is not None and len(data_seep) > 0:
                try:
                    # Use scatter plot - shows exact data points without interpolation
                    sc = ax_right.scatter(data_seep['x'], data_seep['y'], c=data_seep['value'], 
                                        s=1, cmap=custom_cmap, vmin=vmin, vmax=vmax, 
                                        marker='s', edgecolors='none', rasterized=True)
                    
                    ax_right.set_title(f'{dam_name}', fontsize=8, pad=2)
                    ax_right.set_aspect('equal')
                    ax_right.set_xlim(global_x_min, global_x_max)
                    ax_right.set_ylim(global_y_min, global_y_max)
                    ax_right.tick_params(labelsize=6)
                    ax_right.set_xlabel('X (m)', fontsize=6)
                    ax_right.set_ylabel('Y (m)', fontsize=6)
                    # Add label (b), (d), etc. on the left top corner
                    label_char = chr(ord(row_label) + (1 if has_both_columns else 0))
                    ax_right.text(0.02, 0.98, f'({label_char})', transform=ax_right.transAxes, 
                                fontsize=10, fontweight='bold', va='top', ha='left')
                except Exception as e:
                    ax_right.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                               ha='center', va='center', transform=ax_right.transAxes, fontsize=7)
                    ax_right.set_title(f'{dam_name}', fontsize=8, pad=2)
                    label_char = chr(ord(row_label) + 1)
                    ax_right.text(0.02, 0.98, f'({label_char})', transform=ax_right.transAxes, 
                                fontsize=10, fontweight='bold', va='top', ha='left')
            else:
                ax_right.text(0.5, 0.5, 'No data', 
                           ha='center', va='center', transform=ax_right.transAxes, fontsize=7)
                ax_right.set_title(f'{dam_name}', fontsize=8, pad=2)
                label_char = chr(ord(row_label) + 1)
                ax_right.text(0.02, 0.98, f'({label_char})', transform=ax_right.transAxes, 
                            fontsize=10, fontweight='bold', va='top', ha='left')
    
    # Add single unified colorbar
    if has_both_columns:
        fig.subplots_adjust(right=0.90, hspace=0.05, wspace=0.05, left=0.02, top=0.99, bottom=0.01)
        cbar_ax = fig.add_axes((0.92, 0.1, 0.015, 0.8))
    else:
        fig.subplots_adjust(right=0.86, hspace=0.05, left=0.02, top=0.99, bottom=0.01)
        cbar_ax = fig.add_axes((0.88, 0.1, 0.025, 0.8))
    
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(unit, rotation=270, labelpad=20, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    output_path = OUTPUT_DIR / f"{file_suffix}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

print("\nAll figures generated successfully!")
