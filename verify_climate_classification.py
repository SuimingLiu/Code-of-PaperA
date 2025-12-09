"""
Verify Climate Classification by Visualization

This script creates visualizations to verify that the climate zone
classification was performed correctly:
1. World map with climate zones from raster
2. Dam locations colored by assigned climate zone
3. Overlay verification plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from pathlib import Path


# Climate zone colors - consistent with Koppen classification
CLIMATE_COLORS = {
    'A': '#e74c3c',   # Tropical - Red
    'B': '#f39c12',   # Arid - Orange
    'C': '#27ae60',   # Temperate - Green
    'D': '#3498db',   # Continental - Blue
    'E': '#9b59b6',   # Polar - Purple
}

CLIMATE_NAMES = {
    'A': 'Tropical',
    'B': 'Arid', 
    'C': 'Temperate',
    'D': 'Continental',
    'E': 'Polar'
}

# Mapping from GRIDCODE to main climate zone
GRIDCODE_TO_ZONE = {
    1: 'A', 2: 'A', 3: 'A',           # Tropical
    4: 'B', 5: 'B', 6: 'B', 7: 'B',   # Arid
    8: 'C', 9: 'C', 11: 'C', 12: 'C', 13: 'C', 14: 'C', 15: 'C', 16: 'C',  # Temperate
    17: 'D', 18: 'D', 19: 'D', 20: 'D', 21: 'D', 22: 'D', 23: 'D', 24: 'D',
    25: 'D', 26: 'D', 27: 'D', 28: 'D',  # Continental
    29: 'E', 30: 'E', 31: 'E', 32: 'E',  # Polar
}


def create_zone_raster(raster_data):
    """Convert GRIDCODE raster to main climate zone raster"""
    zone_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    zone_raster = np.zeros_like(raster_data, dtype=np.uint8)
    
    for gridcode, zone in GRIDCODE_TO_ZONE.items():
        zone_raster[raster_data == gridcode] = zone_map[zone]
    
    return zone_raster


def plot_verification(p3_csv, raster_path, output_dir):
    """Create verification plots"""
    
    # Read dam data
    print("Loading dam data...")
    df = pd.read_csv(p3_csv, encoding='utf-8-sig')
    print(f"Loaded {len(df)} dams")
    
    # Read raster data
    print("Loading climate raster...")
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        bounds = src.bounds
        transform = src.transform
    
    # Convert to zone raster
    zone_raster = create_zone_raster(raster_data)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # =========================================================================
    # Plot 1: World climate zones map
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Create custom colormap for climate zones
    colors = ['white', '#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    
    # Plot raster
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im1 = ax1.imshow(zone_raster, extent=extent, cmap=cmap, norm=norm, 
                     aspect='auto', origin='upper')
    
    ax1.set_title('World Climate Zones (Koppen Classification)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=CLIMATE_COLORS[z], label=f'{z}: {CLIMATE_NAMES[z]}')
                      for z in ['A', 'B', 'C', 'D', 'E']]
    ax1.legend(handles=legend_patches, loc='lower left', fontsize=10)
    
    # =========================================================================
    # Plot 2: Dam locations by climate zone
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Plot raster background (lighter)
    zone_raster_alpha = np.ma.masked_where(zone_raster == 0, zone_raster)
    ax2.imshow(zone_raster_alpha, extent=extent, cmap=cmap, norm=norm,
               aspect='auto', origin='upper', alpha=0.3)
    
    # Plot dam points
    for zone in ['A', 'B', 'C', 'D', 'E']:
        mask = df['Climate_Zone'] == zone
        if mask.sum() > 0:
            ax2.scatter(df.loc[mask, 'LONG_RIV'], df.loc[mask, 'LAT_RIV'],
                       c=CLIMATE_COLORS[zone], s=50, alpha=0.8,
                       label=f'{zone}: {CLIMATE_NAMES[zone]} (n={mask.sum()})',
                       edgecolors='black', linewidth=0.5)
    
    ax2.set_title('Dam Locations Colored by Assigned Climate Zone', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 3: Zoomed regions with high dam density
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Focus on regions with most dams (China, Europe, Americas)
    # Plot raster background
    ax3.imshow(zone_raster_alpha, extent=extent, cmap=cmap, norm=norm,
               aspect='auto', origin='upper', alpha=0.4)
    
    # Plot all dam points with larger markers
    for zone in ['A', 'B', 'C', 'D', 'E']:
        mask = df['Climate_Zone'] == zone
        if mask.sum() > 0:
            ax3.scatter(df.loc[mask, 'LONG_RIV'], df.loc[mask, 'LAT_RIV'],
                       c=CLIMATE_COLORS[zone], s=80, alpha=0.9,
                       label=f'{zone}: {CLIMATE_NAMES[zone]}',
                       edgecolors='white', linewidth=1)
    
    # Zoom to main dam regions
    ax3.set_xlim(-130, 150)
    ax3.set_ylim(-40, 60)
    ax3.set_title('Dam Distribution (Zoomed View)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 4: Statistical summary
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Bar chart of dam counts by climate zone
    zone_counts = df['Climate_Zone'].value_counts().sort_index()
    zones = zone_counts.index.tolist()
    counts = zone_counts.values
    colors_bar = [CLIMATE_COLORS[z] for z in zones]
    
    bars = ax4.bar(zones, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.annotate(f'{count}\n({count/len(df)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add zone names below
    ax4.set_xticks(range(len(zones)))
    ax4.set_xticklabels([f'{z}\n{CLIMATE_NAMES[z]}' for z in zones], fontsize=11)
    ax4.set_ylabel('Number of Dams', fontsize=12)
    ax4.set_title('Dam Distribution by Climate Zone', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, max(counts) * 1.2)
    
    # Add total count
    ax4.text(0.98, 0.95, f'Total: {len(df)} dams', transform=ax4.transAxes,
             fontsize=12, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'climate_classification_verification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVerification plot saved to: {output_path}")
    
    plt.show()
    
    # =========================================================================
    # Create additional detailed verification plot
    # =========================================================================
    create_regional_verification(df, zone_raster, extent, output_dir)
    
    return df


def create_regional_verification(df, zone_raster, extent, output_dir):
    """Create regional verification plots for major dam regions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define regions of interest
    regions = [
        {'name': 'China', 'xlim': (70, 140), 'ylim': (15, 55)},
        {'name': 'Europe', 'xlim': (-15, 45), 'ylim': (35, 60)},
        {'name': 'North America', 'xlim': (-130, -60), 'ylim': (25, 55)},
        {'name': 'South America', 'xlim': (-80, -30), 'ylim': (-40, 10)},
        {'name': 'Africa & Middle East', 'xlim': (-20, 60), 'ylim': (-35, 40)},
        {'name': 'Southeast Asia & Australia', 'xlim': (90, 160), 'ylim': (-40, 30)},
    ]
    
    # Create colormap
    colors = ['white', '#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    zone_raster_masked = np.ma.masked_where(zone_raster == 0, zone_raster)
    
    for ax, region in zip(axes.flatten(), regions):
        # Plot raster background
        ax.imshow(zone_raster_masked, extent=extent, cmap=cmap, norm=norm,
                  aspect='auto', origin='upper', alpha=0.5)
        
        # Filter dams in this region
        mask_region = ((df['LONG_RIV'] >= region['xlim'][0]) & 
                       (df['LONG_RIV'] <= region['xlim'][1]) &
                       (df['LAT_RIV'] >= region['ylim'][0]) & 
                       (df['LAT_RIV'] <= region['ylim'][1]))
        df_region = df[mask_region]
        
        # Plot dam points
        for zone in ['A', 'B', 'C', 'D', 'E']:
            zone_mask = df_region['Climate_Zone'] == zone
            if zone_mask.sum() > 0:
                ax.scatter(df_region.loc[zone_mask, 'LONG_RIV'], 
                          df_region.loc[zone_mask, 'LAT_RIV'],
                          c=CLIMATE_COLORS[zone], s=100, alpha=0.9,
                          label=f'{zone} (n={zone_mask.sum()})',
                          edgecolors='black', linewidth=1)
        
        ax.set_xlim(region['xlim'])
        ax.set_ylim(region['ylim'])
        ax.set_title(f"{region['name']} (n={len(df_region)} dams)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Regional Climate Classification Verification', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'climate_regional_verification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Regional verification plot saved to: {output_path}")
    
    plt.show()


def print_sample_verification(df):
    """Print sample dams for manual verification"""
    print("\n" + "="*80)
    print("Sample Dams for Manual Verification")
    print("="*80)
    
    for zone in ['A', 'B', 'C', 'D', 'E']:
        mask = df['Climate_Zone'] == zone
        if mask.sum() > 0:
            sample = df[mask].head(3)
            print(f"\n{zone} - {CLIMATE_NAMES[zone]}:")
            for _, row in sample.iterrows():
                print(f"  {row['Dam Name']:30} | Lon: {row['LONG_RIV']:10.4f} | Lat: {row['LAT_RIV']:10.4f} | "
                      f"Code: {row['Climate_Code']} | Country: {row['COUNTRY']}")


def main():
    """Main function"""
    project_root = Path(__file__).parent
    data_dir = project_root / 'data'
    climate_dir = project_root / 'results' / 'climate'
    results_dir = climate_dir
    
    # Create output directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    p3_csv = data_dir / 'P3.csv'
    raster_path = climate_dir / 'koppen_climate.tif'
    
    if not p3_csv.exists():
        print(f"Error: P3.csv not found at {p3_csv}")
        return
    
    if not raster_path.exists():
        print(f"Error: Raster not found at {raster_path}")
        return
    
    # Create verification plots
    df = plot_verification(p3_csv, raster_path, results_dir)
    
    # Print sample verification
    print_sample_verification(df)
    
    print("\n" + "="*80)
    print("Verification complete!")
    print("="*80)


if __name__ == '__main__':
    main()
