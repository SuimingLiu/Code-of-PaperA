
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Wedge, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patheffects as path_effects

# Configuration
DATA_FILE = r"e:\PaperV2\图件pro\data\P3.csv"
OUTPUT_FILE = r"e:\PaperV2\图件pro\results\climate\global_stability_stacked_pie.png"

# Colors - Using Poor/Fair/Good/Excellent labels to match map legend
COLORS = {
    'Excellent': '#3498db',       # Blue - FOS >= 2.0
    'Good': '#abebc6',            # Light Green - 1.5 <= FOS < 2.0
    'Fair': '#f9e79f',            # Light Yellow - 1.2 <= FOS < 1.5
    'Poor': '#e74c3c'             # Red - FOS < 1.2
}

def classify_stability(fos):
    """Classify FOS into 4 categories based on config.py thresholds"""
    if pd.isna(fos):
        return None
    if fos >= 2.0:
        return 'Excellent'
    elif fos >= 1.5:
        return 'Good'
    elif fos >= 1.2:
        return 'Fair'
    else:
        return 'Poor'

def draw_3d_pie(ax, counts, z_level, label, colors, show_text=True):
    total = counts.sum()
    sizes = counts.values
    labels = counts.index
    
    # Starting angle (90 degrees to match standard pie charts)
    start_angle = 90
    
    for i, size in enumerate(sizes):
        if size == 0:
            continue
            
        # Calculate angle for this slice
        angle = (size / total) * 360
        
        # Create a wedge
        theta1 = start_angle
        theta2 = start_angle + angle
        
        # Style: Transparent with black edge, larger radius
        wedge = Wedge((0, 0), 1.2, theta1, theta2, 
                      facecolor=colors.get(labels[i], '#95a5a6'), 
                      edgecolor='black', linewidth=0.5,
                      width=0.5, alpha=0.8) # Wider ring
        
        # Add to 3D axes
        ax.add_patch(wedge)
        art3d.pathpatch_2d_to_3d(wedge, z=z_level, zdir='z')
        
        # Add text label (percentage) - placed outside the ring
        if show_text:
            mid_angle_rad = np.radians(theta1 + angle / 2)
            
            # Radius for text - further outside the ring to avoid overlap
            r_text = 1.6
            x = r_text * np.cos(mid_angle_rad)
            y = r_text * np.sin(mid_angle_rad)
            
            percentage = f"{size/total*100:.0f}%"
            
            # Only show label if slice is big enough (> 8%)
            if size/total > 0.08:
                txt = ax.text(x, y, z_level, percentage, ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='black', zorder=10)

        start_angle += angle

    # Add label for the whole disk in center (just FOS or FOV)
    if show_text:
        short_label = label.replace('\n', ' ').strip()
        ax.text(0, 0, z_level, short_label, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black', zorder=10)

def main():
    # Load data
    try:
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        print(f"Loaded {len(df)} records.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Columns to analyze
    # Order: Bottom to Top visually
    # Bottom (z=0): Predicted FOV
    # Middle (z=1.5): Seepage FOS
    # Top (z=3.0): Normal FOS
    scenarios = [
        ('Predicted FOV', 'FOV'),
        ('Seepage FOS', 'Seepage FOS Maximum'),
        ('Normal FOS', 'Normal FOS Maximum')
    ]

    # Prepare data
    plot_data = []
    categories = ['Excellent', 'Good', 'Fair', 'Poor']
    
    for label, col in scenarios:
        if col not in df.columns:
            print(f"Warning: Column {col} not found.")
            continue
        
        classified = df[col].apply(classify_stability)
        counts = classified.value_counts()
        
        # Ensure all categories exist for consistent coloring (even if 0)
        for cat in categories:
            if cat not in counts:
                counts[cat] = 0
                
        # Reorder
        counts = counts[categories]
        plot_data.append((label, counts))
        
        # Print statistics for this scenario
        print(f"\n{label} Stability Distribution:")
        print(f"  Total samples: {counts.sum()}")
        for cat in categories:
            count = counts[cat]
            percentage = count / counts.sum() * 100
            print(f"  {cat:20s}: {count:4d} ({percentage:5.1f}%)")

    # Create 3D figure
    # Convert cm to inches: 8.47cm height, 7.06cm width
    # 1 inch = 2.54 cm
    fig_width_inch = 7.06 / 2.54  # ~2.78 inches
    fig_height_inch = 8.47 / 2.54  # ~3.33 inches
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    ax = fig.add_subplot(111, projection='3d')
    
    # Z-levels for stacking
    # Bottom: Predicted FOV (z=0)
    # Middle: Seepage FOS (z=1.5)
    # Top: Normal FOS (z=3.0)
    z_levels = [0, 1.5, 3.0]
    
    for i, (label, counts) in enumerate(plot_data):
        z = z_levels[i]
        draw_3d_pie(ax, counts, z, label, COLORS, show_text=True)

    # Set limits - wider for larger rings
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-0.5, 4.0)
    
    # Remove axes panes and lines
    ax.set_axis_off()
    
    # Set view angle
    ax.view_init(elev=30, azim=-60)
    
    # Add Legend with 2x2 format, font size 12 to match other figures
    legend_elements = [Rectangle((0,0),1,1, facecolor=COLORS[cat], edgecolor='black', label=cat) for cat in categories]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=12, frameon=False)
    
    # Title removed as requested
    # plt.suptitle('Global Dam Stability Distribution (3D Stacked)', fontsize=16, y=0.92)

    # Save version with text and legend
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Figure saved to {output_path}")
    plt.close(fig)
    
    # ========== Create version without text and legend ==========
    fig2 = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    for i, (label, counts) in enumerate(plot_data):
        z = z_levels[i]
        draw_3d_pie(ax2, counts, z, label, COLORS, show_text=False)
    
    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_zlim(-0.5, 4.0)
    ax2.set_axis_off()
    ax2.view_init(elev=30, azim=-60)
    
    # Save version without text and legend
    output_path_notext = output_path.parent / 'global_stability_stacked_pie_notext.png'
    plt.savefig(output_path_notext, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Figure (no text) saved to {output_path_notext}")
    plt.close(fig2)

if __name__ == "__main__":
    main()
