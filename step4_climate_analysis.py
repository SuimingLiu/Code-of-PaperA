"""
Step 4: Climate Impact Analysis on Dam Safety

Functionality:
1. Evaluate impact of different climate zones on dam safety based on Koppen classification
2. Classify FOV values by stability thresholds
3. Compute descriptive statistics for each climate zone
4. Execute Kruskal-Wallis non-parametric test
5. Generate boxplots, violin plots, stability stacked bar charts and dashboards

Koppen Climate Classification:
- A: Tropical
- B: Arid
- C: Temperate
- D: Continental
- E: Polar
"""

import os
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import configuration
from config import (
    PROJECT_ROOT, P3_FILE, CLIMATE_DIR, DAM_INFO_FILE,
    KOPPEN_CLIMATE_ZONES, FOV_THRESHOLDS, COLORS, PLOT_CONFIG,
    STABILITY_PALETTE,
    classify_stability, print_section_header, print_subsection, ensure_directories
)

warnings.filterwarnings('ignore')

# Set plot parameters
plt.rcParams.update(PLOT_CONFIG)
sns.set_theme(style='whitegrid')


class ClimateAnalyzer:
    """Climate impact analyzer"""
    
    def __init__(self, data_file=None):
        """
        Initialize
        
        Args:
            data_file: Input data file path (P3.csv - manually prepared)
        """
        self.data_file = Path(data_file) if data_file else P3_FILE
        self.output_dir = CLIMATE_DIR
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load data"""
        if not self.data_file.exists():
            print(f"Error: Data file not found: {self.data_file}")
            print(f"Please manually create P3.csv based on P2.csv before running Step 4.")
            return False
        
        try:
            self.data = pd.read_csv(self.data_file, encoding='utf-8-sig')
            print(f"Data loaded: {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def find_column(self, keywords):
        """Find column by keywords"""
        for col in self.data.columns:
            if all(kw.lower() in col.lower() for kw in keywords):
                return col
        return None
    
    def assign_climate_zones(self):
        """
        Assign climate zones - use existing Climate_Zone column from P3.csv
        or fall back to latitude-based assignment
        """
        print_subsection("Assigning Climate Zones")
        
        # Check if Climate_Zone column already exists (from add_climate_zone.py)
        if 'Climate_Zone' in self.data.columns:
            # Use existing climate zone data from GeoTIFF
            print("  Using existing Climate_Zone column from P3.csv (GeoTIFF-based)")
            
            # Count climate zones
            zone_counts = self.data['Climate_Zone'].value_counts()
            print("\n  Climate zone distribution:")
            for zone, count in zone_counts.items():
                zone_name = KOPPEN_CLIMATE_ZONES.get(zone, {}).get('name', zone)
                print(f"    {zone} ({zone_name}): {count} dams")
            return
        
        # Fallback: Find latitude and longitude columns for simplified assignment
        print("  Warning: Climate_Zone column not found, using latitude-based assignment")
        lat_col = None
        lon_col = None
        
        for col in self.data.columns:
            col_lower = col.lower()
            if 'lat' in col_lower:
                lat_col = col
            elif 'lon' in col_lower or 'long' in col_lower:
                lon_col = col
        
        if not lat_col:
            print("  Warning: Latitude column not found")
            # Try to simulate climate zone assignment
            self._simulate_climate_zones()
            return
        
        print(f"  Using latitude column: {lat_col}")
        
        # Simplified climate zone assignment (based on latitude)
        def assign_zone(lat):
            if pd.isna(lat):
                return 'Unknown'
            lat = abs(lat)
            if lat <= 23.5:
                return 'A'  # Tropical
            elif lat <= 35:
                return 'B'  # Arid/Subtropical
            elif lat <= 50:
                return 'C'  # Temperate
            elif lat <= 66.5:
                return 'D'  # Continental
            else:
                return 'E'  # Polar
        
        self.data['Climate_Zone'] = self.data[lat_col].apply(assign_zone)
        
        # Count climate zones
        zone_counts = self.data['Climate_Zone'].value_counts()
        print("\n  Climate zone distribution:")
        for zone, count in zone_counts.items():
            zone_name = KOPPEN_CLIMATE_ZONES.get(zone, {}).get('name', zone)
            print(f"    {zone} ({zone_name}): {count} dams")
    
    def _simulate_climate_zones(self):
        """Simulate climate zones based on country when no lat/lon data"""
        print("  Simulating climate zones based on country...")
        
        country_col = None
        for col in self.data.columns:
            if 'country' in col.lower():
                country_col = col
                break
        
        if country_col:
            # Simple country-climate mapping
            country_climate = {
                'Brazil': 'A', 'Colombia': 'A', 'Ecuador': 'A',
                'Australia': 'B', 'Algeria': 'B', 'Afghanistan': 'B',
                'China': 'C', 'France': 'C', 'United States': 'C',
                'Canada': 'D', 'Czech Republic': 'D', 'Bulgaria': 'D',
            }
            
            def get_zone(country):
                if pd.isna(country):
                    return 'C'  # Default temperate
                for c, z in country_climate.items():
                    if c.lower() in str(country).lower():
                        return z
                return 'C'
            
            self.data['Climate_Zone'] = self.data[country_col].apply(get_zone)
        else:
            # Random assignment
            np.random.seed(42)
            zones = ['A', 'B', 'C', 'D']
            weights = [0.15, 0.2, 0.4, 0.25]
            self.data['Climate_Zone'] = np.random.choice(zones, size=len(self.data), p=weights)
        
        zone_counts = self.data['Climate_Zone'].value_counts()
        print("\n  Climate zone distribution (simulated):")
        for zone, count in zone_counts.items():
            zone_name = KOPPEN_CLIMATE_ZONES.get(zone, {}).get('name', zone)
            print(f"    {zone} ({zone_name}): {count} dams")
    
    def classify_stability(self):
        """Classify stability for all three FOS scenarios"""
        print_subsection("Classifying Stability for Three FOS Scenarios")
        
        # Find all FOS columns
        normal_fos_col = self.find_column(['Normal', 'FOS', 'Maximum'])
        seepage_fos_col = self.find_column(['Seepage', 'FOS', 'Maximum'])
        
        # FOV column (Factor of Vulnerability - predicted safety considering seepage degradation)
        fov_col = None
        for col in self.data.columns:
            if col.upper() == 'FOV' or 'FOV' in col.upper():
                fov_col = col
                break
        
        # Store column names for later use
        self.fos_columns = {}
        
        if normal_fos_col:
            self.fos_columns['Normal FOS'] = normal_fos_col
            self.data['Stability_Normal'] = self.data[normal_fos_col].apply(classify_stability)
            print(f"  Normal FOS: {normal_fos_col}")
        
        if seepage_fos_col:
            self.fos_columns['Seepage FOS'] = seepage_fos_col
            self.data['Stability_Seepage'] = self.data[seepage_fos_col].apply(classify_stability)
            print(f"  Seepage FOS (with drainage): {seepage_fos_col}")
        
        if fov_col:
            self.fos_columns['FOV (Predicted)'] = fov_col
            self.data['Stability_FOV'] = self.data[fov_col].apply(classify_stability)
            print(f"  FOV (Factor of Vulnerability): {fov_col}")
        else:
            print(f"  FOV column not found in data")
        
        # Print statistics for each scenario
        print("\n  Stability distribution by scenario:")
        for scenario, col in self.fos_columns.items():
            # Extract first word for stability column naming
            first_word = scenario.split()[0]
            # Handle FOV specially
            if 'FOV' in scenario:
                stability_col = 'Stability_FOV'
            else:
                stability_col = f"Stability_{first_word}"
            
            if stability_col in self.data.columns:
                counts = self.data[stability_col].value_counts()
                print(f"\n  {scenario}:")
                for cat, count in counts.items():
                    percent = count / len(self.data) * 100
                    print(f"    {cat}: {count} ({percent:.1f}%)")
        
        # Keep backward compatibility - use Seepage FOS as default
        if seepage_fos_col:
            self.fos_col = seepage_fos_col
            self.data['Stability'] = self.data['Stability_Seepage']
        elif normal_fos_col:
            self.fos_col = normal_fos_col
            self.data['Stability'] = self.data['Stability_Normal']
    
    def compute_statistics(self):
        """Compute statistics for each climate zone"""
        print_subsection("Computing Statistics by Climate Zone")
        
        if 'Climate_Zone' not in self.data.columns or not hasattr(self, 'fos_col'):
            print("  Warning: Required columns not available")
            return
        
        stats_records = []
        
        for zone in sorted(self.data['Climate_Zone'].unique()):
            if zone == 'Unknown':
                continue
            
            zone_data = self.data[self.data['Climate_Zone'] == zone][self.fos_col].dropna()
            
            if len(zone_data) == 0:
                continue
            
            zone_name = KOPPEN_CLIMATE_ZONES.get(zone, {}).get('name', zone)
            
            stats_records.append({
                'Climate_Zone': zone,
                'Zone_Name': zone_name,
                'Count': len(zone_data),
                'Mean': zone_data.mean(),
                'Std': zone_data.std(),
                'Min': zone_data.min(),
                'Q25': zone_data.quantile(0.25),
                'Median': zone_data.median(),
                'Q75': zone_data.quantile(0.75),
                'Max': zone_data.max()
            })
            
            print(f"  {zone} ({zone_name}): n={len(zone_data)}, mean={zone_data.mean():.3f}, median={zone_data.median():.3f}")
        
        self.climate_stats = pd.DataFrame(stats_records)
        return self.climate_stats
    
    def kruskal_wallis_test(self):
        """Execute Kruskal-Wallis non-parametric test"""
        print_subsection("Kruskal-Wallis Test")
        
        if 'Climate_Zone' not in self.data.columns or not hasattr(self, 'fos_col'):
            return None
        
        # Prepare group data
        groups = []
        zone_names = []
        
        for zone in sorted(self.data['Climate_Zone'].unique()):
            if zone == 'Unknown':
                continue
            zone_data = self.data[self.data['Climate_Zone'] == zone][self.fos_col].dropna()
            if len(zone_data) >= 3:
                groups.append(zone_data.values)
                zone_names.append(zone)
        
        if len(groups) < 2:
            print("  Warning: Not enough groups for test")
            return None
        
        # Execute test
        h_stat, p_value = stats.kruskal(*groups)
        
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  p-value: {p_value:.4e}")
        print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        self.results['kruskal_wallis'] = {
            'h_statistic': h_stat,
            'p_value': p_value,
            'groups': zone_names,
            'significant': p_value < 0.05
        }
        
        return h_stat, p_value
    
    def create_boxplot(self):
        """Create boxplot"""
        if 'Climate_Zone' not in self.data.columns or not hasattr(self, 'fos_col'):
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        df_plot = self.data[self.data['Climate_Zone'] != 'Unknown'].copy()
        
        # Create labels
        zone_labels = {z: f"{z}\n({KOPPEN_CLIMATE_ZONES.get(z, {}).get('name', z)})" 
                      for z in df_plot['Climate_Zone'].unique()}
        df_plot['Zone_Label'] = df_plot['Climate_Zone'].map(zone_labels)
        
        # Get climate zone colors
        colors = [KOPPEN_CLIMATE_ZONES.get(z, {}).get('color', '#95a5a6') 
                 for z in sorted(df_plot['Climate_Zone'].unique())]
        
        sns.boxplot(
            data=df_plot,
            x='Zone_Label',
            y=self.fos_col,
            palette=colors,
            ax=ax
        )
        
        # Add stability threshold lines
        ax.axhline(y=FOV_THRESHOLDS['stable'], color='green', linestyle='--', 
                  label=f"Stable (>{FOV_THRESHOLDS['stable']})")
        ax.axhline(y=FOV_THRESHOLDS['marginal'], color='orange', linestyle='--',
                  label=f"At Risk (<{FOV_THRESHOLDS['marginal']})")
        
        ax.set_xlabel('Climate Zone', fontsize=12)
        ax.set_ylabel('Dam Stability Factor (FOS)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'climate_fos_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("  Boxplot saved")
    
    def create_violin_plot(self):
        """Create violin plots for all three FOS scenarios"""
        if 'Climate_Zone' not in self.data.columns or not hasattr(self, 'fos_columns'):
            return
        
        print("  Creating violin plots for three FOS scenarios...")
        
        # Create three violin plots
        for scenario, fos_col in self.fos_columns.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            df_plot = self.data[self.data['Climate_Zone'] != 'Unknown'].copy()
            
            # Define explicit order A-E
            desired_order = ['A', 'B', 'C', 'D', 'E']
            available_zones = set(df_plot['Climate_Zone'].unique())
            plot_order = [z for z in desired_order if z in available_zones]
            
            zone_labels = {z: f"{z}\n({KOPPEN_CLIMATE_ZONES.get(z, {}).get('name', z)})" 
                          for z in plot_order}
            df_plot['Zone_Label'] = df_plot['Climate_Zone'].map(zone_labels)
            
            # Get labels in the correct order
            order_labels = [zone_labels[z] for z in plot_order]
            
            colors = [KOPPEN_CLIMATE_ZONES.get(z, {}).get('color', '#95a5a6') 
                     for z in plot_order]
            
            sns.violinplot(
                data=df_plot,
                x='Zone_Label',
                y=fos_col,
                order=order_labels,
                palette=colors,
                ax=ax
            )
            
            # Determine label term (FOS or FOV)
            term = 'FOV' if 'FOV' in scenario else 'FOS'
            
            # Add FOS stability threshold lines
            # Good (Stable) -> Green
            ax.axhline(y=FOV_THRESHOLDS['stable'], color=COLORS['stable'], linestyle='--', linewidth=2,
                      label=f"Stable ({term} ≥ {FOV_THRESHOLDS['stable']})")
            # Poor (At Risk) -> Red
            ax.axhline(y=FOV_THRESHOLDS['marginal'], color=COLORS['at_risk'], linestyle='--', linewidth=2,
                      label=f"Marginal ({term} = {FOV_THRESHOLDS['marginal']})")
            
            ax.set_xlabel('Climate Zone', fontsize=24, fontweight='bold')
            ax.set_ylabel(f'{scenario}', fontsize=24, fontweight='bold')
            ax.set_ylim(0, 4)  # Set unified y-axis range
            ax.tick_params(axis='both', which='major', labelsize=20)
            # ax.set_title(f'Climate Impact on Dam Stability - {scenario}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=18)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Save figure
            scenario_slug = scenario.lower().replace(' ', '_').replace('(', '').replace(')', '')
            fig.tight_layout()
            fig.savefig(self.output_dir / f'climate_violin_{scenario_slug}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    Violin plot saved: climate_violin_{scenario_slug}.png")
    
    def create_stability_pie_charts(self):
        """Create stability pie charts for all three FOS scenarios - stacked vertically"""
        if 'Climate_Zone' not in self.data.columns:
            return
        
        print("  Creating stability pie charts for three FOS scenarios...")
        
        # Map scenario names to stability column names
        scenarios = []
        for scenario_name in self.fos_columns.keys():
            stability_col = f"Stability_{scenario_name.split()[0]}"
            if stability_col in self.data.columns:
                scenarios.append((scenario_name, stability_col))
        
        if not scenarios:
            print("    Warning: No stability columns found")
            return
        
        df_plot = self.data[self.data['Climate_Zone'] != 'Unknown'].copy()
        zones = sorted(df_plot['Climate_Zone'].unique())
        n_zones = len(zones)
        
        # Stability colors using FOS color scheme
        from config import STABILITY_PALETTE, FIGURE_SIZES
        stability_colors = {
            'At Risk': STABILITY_PALETTE[0],        # Red
            'Marginally Stable': STABILITY_PALETTE[1],  # Yellow
            'Stable': STABILITY_PALETTE[2],         # Green
            'Unknown': COLORS['neutral']            # Gray
        }
        stability_order = ['Stable', 'Marginally Stable', 'At Risk']
        
        # Create one figure for each scenario
        for scenario_name, stability_col in scenarios:
            # Create vertical stacked layout: n_zones rows × 1 column
            fig_height = FIGURE_SIZES['climate_pie_stacked'][1]
            fig, axes = plt.subplots(n_zones, 1, figsize=(8, fig_height))
            
            # Ensure axes is always a list
            if n_zones == 1:
                axes = [axes]
            
            for idx, zone in enumerate(zones):
                ax = axes[idx]
                zone_data = df_plot[df_plot['Climate_Zone'] == zone]
                
                # Count stability categories
                counts = zone_data[stability_col].value_counts()
                
                # Prepare data in order
                labels = []
                sizes = []
                colors = []
                for cat in stability_order:
                    if cat in counts.index:
                        labels.append(cat)
                        sizes.append(counts[cat])
                        colors.append(stability_colors[cat])
                
                if sizes:
                    # Create pie chart with donut style
                    wedges, texts, autotexts = ax.pie(
                        sizes, labels=None, colors=colors,
                        autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                        startangle=90, pctdistance=0.75,
                        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
                    )
                    
                    # Style autopct text
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                        autotext.set_color('white')
                    
                    # Add center text with zone info
                    zone_name = KOPPEN_CLIMATE_ZONES.get(zone, {}).get('name', zone)
                    total = sum(sizes)
                    ax.text(0, 0, f'{zone}\n{zone_name}\n(n={total})', 
                           ha='center', va='center', fontsize=10.5, fontweight='bold')
                    
                    # Add title for this subplot
                    # ax.set_title(f'Climate Zone {zone} - {zone_name}', 
                    #            fontsize=12, fontweight='bold', pad=10)
                
                ax.set_aspect('equal')
            
            # Add common legend at the bottom
            legend_elements = [plt.Circle((0, 0), 1, facecolor=stability_colors[cat], 
                                           edgecolor='white', label=cat)
                              for cat in stability_order]
            fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                      fontsize=10.5, bbox_to_anchor=(0.5, 0.02), frameon=True)
            
            # Add overall title
            # fig.suptitle(f'Dam Stability Distribution by Climate Zone - {scenario_name}', 
            #             fontsize=14, fontweight='bold', y=0.995)
            fig.tight_layout(rect=[0, 0.05, 1, 0.99])
            
            # Save figure
            scenario_slug = scenario_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            fig.savefig(self.output_dir / f'climate_pie_{scenario_slug}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    Pie chart saved: climate_pie_{scenario_slug}.png")
    
    def save_results(self):
        """Save results"""
        # Save statistics
        if hasattr(self, 'climate_stats'):
            self.climate_stats.to_csv(self.output_dir / 'climate_statistics.csv', 
                                     index=False, encoding='utf-8-sig')
        
        # Save test results
        if self.results:
            with open(self.output_dir / 'climate_analysis_results.json', 'w', encoding='utf-8') as f:
                # Convert numpy types to Python native types
                def convert_value(v):
                    if isinstance(v, (np.floating, np.integer)):
                        return float(v)
                    elif isinstance(v, np.bool_):
                        return bool(v)
                    elif isinstance(v, np.ndarray):
                        return v.tolist()
                    return v
                
                results_serializable = {}
                for k, v in self.results.items():
                    if isinstance(v, dict):
                        results_serializable[k] = {
                            kk: convert_value(vv)
                            for kk, vv in v.items()
                        }
                    else:
                        results_serializable[k] = convert_value(v)
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def run(self):
        """Execute full climate impact analysis"""
        print_section_header("Step 4: Climate Impact Analysis")
        
        # Ensure directories exist
        ensure_directories()
        
        # Load data
        if not self.load_data():
            return None
        
        # Assign climate zones
        self.assign_climate_zones()
        
        # Classify stability
        self.classify_stability()
        
        # Compute statistics
        self.compute_statistics()
        
        # Execute test
        self.kruskal_wallis_test()
        
        # Create visualizations
        print_subsection("Creating Visualizations")
        # self.create_boxplot()  # Removed as per user request
        self.create_violin_plot()
        self.create_stability_pie_charts()
        
        # Save results
        self.save_results()
        
        return self.results


def main():
    """Main function"""
    os.chdir(PROJECT_ROOT)
    
    analyzer = ClimateAnalyzer()
    results = analyzer.run()
    
    print("\n" + "=" * 70)
    print("Step 4 completed!")
    print("All analysis steps finished.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
