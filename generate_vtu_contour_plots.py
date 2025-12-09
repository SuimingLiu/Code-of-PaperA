"""
Generate contour plots from VTU files for selected dams using PyVista
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
from pathlib import Path
import pyvista as pv

# Configuration
BASE_DIR = Path(r'e:\PaperV2\图件pro')
VTU_DIR = BASE_DIR / 'data' / 'vtu'
OUTPUT_DIR = BASE_DIR / 'selected_dam_plots_vtu'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Data
df_main = pd.read_csv(r'e:\PaperV2\Other\1109备份最终数据.csv')
df_climate = pd.read_csv(r'e:\PaperV2\Other\dam_data_with_climate.csv')

# Merge Climate Data
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
group_a = df_merged[df_merged['F19_max'] > df_merged['F8_max']]
group_b = df_merged[df_merged['F19_max'] < df_merged['F8_max']]

print(f"Group A (Seepage > Normal): {len(group_a)} dams")
print(f"Group B (Seepage < Normal): {len(group_b)} dams")

# Selection Logic
selected_dams = []

if not group_b.empty:
    dam_b = group_b.iloc[0]
    selected_dams.append(dam_b)
    print(f"Selected from Group B: {dam_b['DAM_NAME']} ({dam_b['Continent']}, {dam_b['Cli_Zone']})")

used_continents = {d['Continent'] for d in selected_dams}
used_climates = {d['Cli_Zone'] for d in selected_dams}

candidates_a = group_a.copy()

for _ in range(4):
    if candidates_a.empty:
        break
    
    scores = []
    for idx, row in candidates_a.iterrows():
        score = 0
        if row['Continent'] not in used_continents:
            score += 2
        if row['Cli_Zone'] not in used_climates:
            score += 1
        scores.append((score, idx))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    best_idx = scores[0][1]
    best_dam = candidates_a.loc[best_idx]
    
    selected_dams.append(best_dam)
    used_continents.add(best_dam['Continent'])
    used_climates.add(best_dam['Cli_Zone'])
    candidates_a = candidates_a.drop(best_idx)

# Manual fix for Dry Falls
for d in selected_dams:
    if d['GDW_ID'] == 106 and pd.isna(d['Cli_Zone']):
        d['Cli_Zone'] = 'D'
        d['Continent'] = 'North America'

print("Selected Dams:")
for d in selected_dams:
    print(f"- {d['DAM_NAME']} (ID: {d['GDW_ID']}): {d['Continent']}, {d['Cli_Zone']}, Normal FOS={d['F8_max']}, Seepage FOS={d['F19_max']}")

# Plotting Parameters
parameter_groups = [
    ('F6', 'F17', 'Pressure Head', 'm', 'pressure_head'),
    ('F7', 'F18', 'Displacement', 'm', 'displacement'),
    ('F9', 'F20', 'Resistivity', 'Ωm', 'resistivity'),
    ('F10', 'F21', 'SP Potential', 'mV', 'sp_potential'),
    (None, 'F22', 'Seepage Electric Field Potential', 'mV', 'electric_field')
]

def read_vtu_data(dam_name, f_code):
    """Read VTU file and extract data"""
    if f_code is None:
        return None
    
    vtu_path = VTU_DIR / f_code / f"{dam_name}.vtu"
    
    if not vtu_path.exists():
        print(f"  VTU file not found: {vtu_path}")
        return None
    
    try:
        mesh = pv.read(str(vtu_path))
        
        # Extract point coordinates
        points = mesh.points
        x = points[:, 0]
        y = points[:, 1]
        
        # Extract scalar values (field name is f_code)
        if f_code in mesh.point_data:
            values = mesh.point_data[f_code]
        else:
            print(f"  Field {f_code} not found in {vtu_path}")
            return None
        
        return {'x': x, 'y': y, 'value': values, 'mesh': mesh}
    
    except Exception as e:
        print(f"  Error reading {vtu_path}: {e}")
        return None

def plot_vtu_contour(ax, data, custom_cmap, vmin, vmax):
    """Plot contour using PyVista mesh data with matplotlib tricontourf"""
    try:
        mesh = data['mesh']
        
        # Get 2D coordinates (x, y from the mesh points)
        points = mesh.points
        x = points[:, 0]
        y = points[:, 1]
        
        # Get scalar field values
        f_code = list(mesh.point_data.keys())[0]
        values = mesh.point_data[f_code]
        
        # Extract triangles from PyVista UnstructuredGrid
        # The cells array format is: [n_vertices, v1, v2, v3, n_vertices, v4, v5, v6, ...]
        # For triangles, n_vertices = 3
        from matplotlib.tri import Triangulation
        
        cells = mesh.cells
        # Reshape to extract triangles (assuming all cells are triangles with 3 vertices)
        # Format: [3, v1, v2, v3, 3, v4, v5, v6, ...]
        cells_reshaped = cells.reshape(-1, 4)  # Each triangle: [3, v1, v2, v3]
        triangles = cells_reshaped[:, 1:]  # Extract vertex indices [v1, v2, v3]
        
        # Create matplotlib Triangulation
        triang = Triangulation(x, y, triangles)
        
        # Get bounds
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Plot filled contour
        contourf = ax.tricontourf(triang, values, levels=15, cmap=custom_cmap, 
                                  vmin=vmin, vmax=vmax)
        
        # Add contour lines
        ax.tricontour(triang, values, levels=15, colors='black', 
                     linewidths=0.5, alpha=0.6)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        return True
        
    except Exception as e:
        print(f"  Error plotting contour: {e}")
        import traceback
        traceback.print_exc()
        return False

# Custom colormap
colors = ['#2E86AB', '#06A77D', '#D5C67A', '#F77F00', '#D62828']
custom_cmap = LinearSegmentedColormap.from_list('fresh', colors)

# Generate one figure per parameter
for param_idx, (f_code_left, f_code_right, param_name, unit, file_suffix) in enumerate(parameter_groups):
    print(f"\nGenerating figure for {param_name}...")
    
    has_both_columns = f_code_left is not None and f_code_right is not None
    has_only_right = f_code_left is None and f_code_right is not None
    
    # Collect all data to find global vmin/vmax and x/y ranges
    all_values = []
    all_x_ranges = []
    all_y_ranges = []
    dam_data_cache = {}  # Cache data for each dam to avoid re-reading
    
    for dam in selected_dams:
        dam_name = dam['DAM_NAME']
        if f_code_left:
            data = read_vtu_data(dam_name, f_code_left)
            if data is not None:
                all_values.extend(data['value'])
                all_x_ranges.extend([data['x'].min(), data['x'].max()])
                all_y_ranges.extend([data['y'].min(), data['y'].max()])
                dam_data_cache[(dam_name, f_code_left)] = data
        if f_code_right:
            data = read_vtu_data(dam_name, f_code_right)
            if data is not None:
                all_values.extend(data['value'])
                all_x_ranges.extend([data['x'].min(), data['x'].max()])
                all_y_ranges.extend([data['y'].min(), data['y'].max()])
                dam_data_cache[(dam_name, f_code_right)] = data
    
    if len(all_values) > 0:
        # Global vmin/vmax for unified colorbar
        global_vmin, global_vmax = np.percentile(all_values, [2, 98])
    else:
        global_vmin, global_vmax = 0, 1
    
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
        axes = axes.reshape(-1, 1)
    
    # Plot each dam
    for dam_idx, dam in enumerate(selected_dams):
        dam_name = dam['DAM_NAME']
        row_label = chr(ord('a') + dam_idx)
        
        # Left column - Normal state
        if f_code_left is not None:
            ax_left = axes[dam_idx, 0] if has_both_columns else None
            if ax_left is not None:
                data_norm = dam_data_cache.get((dam_name, f_code_left))
                if data_norm is not None:
                    # Calculate individual dam's vmin/vmax
                    dam_vmin = np.percentile(data_norm['value'], 2)
                    dam_vmax = np.percentile(data_norm['value'], 98)
                    
                    # Use PyVista plotting
                    success = plot_vtu_contour(ax_left, data_norm, custom_cmap, dam_vmin, dam_vmax)
                    
                    if success:
                        ax_left.set_title(f'{dam_name}', fontsize=8, pad=2)
                        ax_left.set_aspect('equal')
                        ax_left.tick_params(labelsize=6)
                        ax_left.set_xlabel('X (m)', fontsize=6)
                        ax_left.set_ylabel('Y (m)', fontsize=6)
                        ax_left.text(0.02, 0.98, f'({row_label})', transform=ax_left.transAxes,
                                   fontsize=10, fontweight='bold', va='top', ha='left')
                    else:
                        ax_left.text(0.5, 0.5, 'Plotting error',
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
            data_seep = dam_data_cache.get((dam_name, f_code_right))
            if data_seep is not None:
                # Calculate individual dam's vmin/vmax
                dam_vmin = np.percentile(data_seep['value'], 2)
                dam_vmax = np.percentile(data_seep['value'], 98)
                
                # Use PyVista plotting
                success = plot_vtu_contour(ax_right, data_seep, custom_cmap, dam_vmin, dam_vmax)
                
                if success:
                    ax_right.set_title(f'{dam_name}', fontsize=8, pad=2)
                    ax_right.set_aspect('equal')
                    ax_right.tick_params(labelsize=6)
                    ax_right.set_xlabel('X (m)', fontsize=6)
                    ax_right.set_ylabel('Y (m)', fontsize=6)
                    label_char = chr(ord(row_label) + (1 if has_both_columns else 0))
                    ax_right.text(0.02, 0.98, f'({label_char})', transform=ax_right.transAxes,
                                fontsize=10, fontweight='bold', va='top', ha='left')
                else:
                    ax_right.text(0.5, 0.5, 'Plotting error',
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
    
    # Add unified colorbar
    if has_both_columns:
        fig.subplots_adjust(right=0.90, hspace=0.05, wspace=0.05, left=0.02, top=0.99, bottom=0.01)
        cbar_ax = fig.add_axes((0.92, 0.1, 0.015, 0.8))
    else:
        fig.subplots_adjust(right=0.86, hspace=0.05, left=0.02, top=0.99, bottom=0.01)
        cbar_ax = fig.add_axes((0.88, 0.1, 0.025, 0.8))
    
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(unit, rotation=270, labelpad=20, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    output_path = OUTPUT_DIR / f"{file_suffix}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

print("\nAll VTU contour figures generated successfully!")
