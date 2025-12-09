
"""
Step 2: FOS Analysis and Climate Impact Visualization

Functionality:
1. Parameter correlation analysis:
   - Normal Self-Potential vs Normal FOS
   - Seepage Self-Potential vs Seepage FOS
   - Coupling Electric Field vs Seepage FOS
2. FOS comparison analysis:
   - Subplot a: Seepage FOS vs Normal FOS scatter plot
   - Subplot b: FOS change distribution and normality test
   - Subplot c: Climate zone FOS violin plot
   - Subplot d: 3D waterfall plot of FOS distribution by climate zone
3. Mark FOS instability thresholds
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
from mpl_toolkits.mplot3d import Axes3D

# Import configuration
from config import (
    PROJECT_ROOT, DATA_DIR, P1_FILE, CORRELATION_DIR, MODELS_DIR,
    KOPPEN_CLIMATE_ZONES, FOS_THRESHOLDS, COLORS, PLOT_CONFIG,
    FIGURE_SIZES,
    print_section_header, print_subsection, ensure_directories
)

warnings.filterwarnings('ignore')

# Set plot parameters
plt.rcParams.update(PLOT_CONFIG)
sns.set_theme(style='whitegrid')


class FOSAnalyzer:
    """FOS Analyzer for correlation and climate analysis"""
    
    def __init__(self, data_file=None):
        """
        Initialize the analyzer
        
        Args:
            data_file: Input data file path, defaults to P1.csv
        """
        self.data_file = Path(data_file) if data_file else P1_FILE
        self.output_dir = CORRELATION_DIR
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load data"""
        if not self.data_file.exists():
            print(f"Error: Data file not found: {self.data_file}")
            return False
        
        try:
            self.data = pd.read_csv(self.data_file, encoding='utf-8-sig')
            print(f"Data loaded: {len(self.data)} records, {len(self.data.columns)} columns")
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
        """Assign climate zones based on latitude"""
        lat_col = None
        for col in ['LAT_RIV', 'LAT', 'Latitude', 'latitude']:
            if col in self.data.columns:
                lat_col = col
                break
        
        if lat_col is None:
            print("  Warning: Latitude column not found")
            return False
        
        def get_climate_zone(lat):
            if pd.isna(lat):
                return None
            abs_lat = abs(lat)
            if abs_lat <= 23.5:
                return 'A'  # Tropical
            elif abs_lat <= 35:
                return 'B'  # Arid
            elif abs_lat <= 55:
                return 'C'  # Temperate
            elif abs_lat <= 66.5:
                return 'D'  # Continental
            else:
                return 'E'  # Polar
        
        self.data['Climate_Zone'] = self.data[lat_col].apply(get_climate_zone)
        return True
    
    def filter_outliers_by_residuals(self, x, y, threshold=3.0):
        """
        Filter outliers based on residuals from linear regression
        
        Args:
            x: x-axis data (numpy array)
            y: y-axis data (numpy array)
            threshold: number of standard deviations to use as cutoff (default: 3.0)
            
        Returns:
            x_filtered, y_filtered: filtered arrays with outliers removed
        """
        if len(x) < 3:
            return x, y
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Predicted values
        y_pred = slope * x + intercept
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Calculate residual standard deviation
        residual_std = np.std(residuals)
        
        # Filter: keep points within threshold * std
        mask = np.abs(residuals) <= (threshold * residual_std)
        
        return x[mask], y[mask]
    
    def create_parameter_correlation_figure(self):
        """创建参数与FOS的相关性分析图 - 分为两个图"""
        print_subsection("Creating Parameter Correlation Figures")
        
        normal_fos = ['Normal', 'FOS', 'Maximum']
        seepage_fos = ['Seepage', 'FOS', 'Maximum']

        # ===== Figure 2a: 4个重点子图 (a, b, c, d) =====
        pairs_key = [
            (normal_fos, ['Normal', 'Resistivity', 'Maximum'], '(a)', 'Normal FOS', 'Normal Res Max (Ω·m)'),
            (normal_fos, ['Normal', 'Resistivity', 'Minimum'], '(b)', 'Normal FOS', 'Normal Res Min (Ω·m)'),
            (seepage_fos, ['Seepage', 'Resistivity', 'Maximum'], '(c)', 'Seepage FOS', 'Seepage Res Max (Ω·m)'),
            (seepage_fos, ['Seepage', 'Resistivity', 'Minimum'], '(d)', 'Seepage FOS', 'Seepage Res Min (Ω·m)'),
        ]

        fig1, axes1 = plt.subplots(2, 2, figsize=FIGURE_SIZES['quad'])
        axes1 = axes1.flatten()
        
        # Import Line2D for custom legend
        from matplotlib.lines import Line2D
        
        for i, (x_kws, y_kws, label, xlabel, ylabel) in enumerate(pairs_key):
            ax = axes1[i]
            x_col = self.find_column(x_kws)
            y_col = self.find_column(y_kws)
            
            if not x_col or not y_col:
                ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
                continue
                
            df_plot = self.data[[x_col, y_col]].dropna()
            x = df_plot[x_col]
            y = df_plot[y_col]
            
            # Use the same color as COLORS['primary'] for scatter
            scatter_color = COLORS['primary']
            ax.scatter(x, y, alpha=0.6, c=scatter_color, edgecolor='white', s=50)
            
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line = slope * x + intercept
                ax.plot(x, line, 'r-', linewidth=2)
                
                # Format p-value: if < 1e-6, show as "p < 0.001"
                if p_value < 1e-6:
                    p_str = 'p < 0.001'
                else:
                    p_str = f'p = {p_value:.4f}'
                
                # Create custom legend with data color matching scatter points (similar to fig2b style)
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label=f'Data (n={len(x)})',
                          markerfacecolor=scatter_color, markersize=8, 
                          markeredgecolor='white', markeredgewidth=0.5, alpha=0.6),
                    Line2D([0], [0], color='r', lw=2, label='Fit Line'),
                    Line2D([0], [0], linestyle='None', label=f'R = {r_value:.4f}'),
                    Line2D([0], [0], linestyle='None', label=f'{p_str}')
                ]
                
                # Place legend at top-right inside the plot (matching fig2b placement)
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9.5, 
                         framealpha=0.95, edgecolor='gray', fancybox=True)
            
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
            ax.grid(True, alpha=0.3)
            
            # Disable offset on y-axis to remove "1e-7-7.9934e-1" style labels
            ax.ticklabel_format(useOffset=False, axis='y')
            
        plt.tight_layout()
        fig1.savefig(self.output_dir / 'fig2a_key_correlations.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("  ✓ Key correlations saved: fig2a_key_correlations.png")

        # ===== Figure 2b: 其余6个子图 (3列2行) =====
        pairs_other = [
            (normal_fos, ['Normal', 'Self-Potential', 'Maximum'], '(a)', 'Normal FOS', 'Normal SP Max (mV)'),
            (normal_fos, ['Normal', 'Self-Potential', 'Minimum'], '(b)', 'Normal FOS', 'Normal SP Min (mV)'),
            (seepage_fos, ['Coupling', 'Electric', 'Minimum'], '(e)', 'Seepage FOS', 'Coupling Field Min (V)'),
            (seepage_fos, ['Seepage', 'Self-Potential', 'Maximum'], '(f)', 'Seepage FOS', 'Seepage SP Max (mV)'),
            (seepage_fos, ['Seepage', 'Self-Potential', 'Minimum'], '(g)', 'Seepage FOS', 'Seepage SP Min (mV)'),
            (seepage_fos, ['Coupling', 'Electric', 'Maximum'], '(j)', 'Seepage FOS', 'Coupling Field Max (V)'),
        ]


        # 2列3行排布
        fig2, axes2 = plt.subplots(3, 2, figsize=(12, 14))
        axes2 = axes2.flatten()

        # 为每个参数分配不同颜色 (Normal SP, Seepage SP, Coupling)
        # Normal SP: Blue (#1f77b4)
        # Seepage SP: Orange (#ff7f0e)
        # Coupling: Green (#2ca02c)
        
        def get_param_color(y_keywords):
            y_str = ' '.join(y_keywords).lower()
            if 'normal' in y_str and 'self-potential' in y_str:
                return '#1f77b4' # Blue
            elif 'seepage' in y_str and 'self-potential' in y_str:
                return '#ff7f0e' # Orange
            elif 'coupling' in y_str:
                return '#2ca02c' # Green
            else:
                return '#7f7f7f' # Gray (fallback)

        for i, (x_kws, y_kws, label, xlabel, ylabel) in enumerate(pairs_other):
            ax = axes2[i]
            x_col = self.find_column(x_kws)
            y_col = self.find_column(y_kws)

            if not x_col or not y_col:
                ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
                continue

            df_plot = self.data[[x_col, y_col]].dropna()
            x = df_plot[x_col].values
            y = df_plot[y_col].values
            
            # Filter outliers using residual-based method
            original_count = len(x)
            x_filtered, y_filtered = self.filter_outliers_by_residuals(x, y, threshold=3.0)
            filtered_count = original_count - len(x_filtered)

            color = get_param_color(y_kws)
            ax.scatter(x_filtered, y_filtered, alpha=0.7, c=color, edgecolor='white', s=50)

            if len(x_filtered) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_filtered, y_filtered)
                line = slope * x_filtered + intercept
                ax.plot(x_filtered, line, color='black', linestyle='-', linewidth=2)

                # Format p-value: if < 1e-6, show as "p < 0.001"
                if p_value < 1e-6:
                    p_str = 'p < 0.001'
                else:
                    p_str = f'p = {p_value:.4f}'
                stats_text = f'R = {r_value:.4f}\n{p_str}'
                if filtered_count > 0:
                    stats_text += f'\n(filtered: {filtered_count})'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10.5,
                        verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=10.5, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=10.5, fontweight='bold')
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
            ax.grid(True, alpha=0.3)

        # 隐藏多余子图
        for j in range(len(pairs_other), len(axes2)):
            axes2[j].axis('off')

        plt.tight_layout()
        fig2.savefig(self.output_dir / 'fig2b_other_correlations.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("  ✓ Other correlations saved: fig2b_other_correlations.png")

    def create_combined_figure(self):
        """Create combined analysis figure (Fig 2c in paper)"""
        print_subsection("Creating Combined Figure")
        
        normal_fos_col = self.find_column(['Normal', 'FOS', 'Maximum'])
        seepage_fos_col = self.find_column(['Seepage', 'FOS', 'Maximum'])
        
        if not normal_fos_col or not seepage_fos_col:
            print("  Warning: FOS columns not found")
            return None
        
        # 准备数据
        df_valid = self.data[[normal_fos_col, seepage_fos_col, 'Climate_Zone']].dropna()
        df_valid = df_valid.copy()
        # Calculate change (Seepage - Normal)
        df_valid['FOS_Change'] = df_valid[seepage_fos_col] - df_valid[normal_fos_col]
        
        # 创建2x2子图 - 统一图件大小
        fig = plt.figure(figsize=(14, 12))
        
        # ==================== 子图a: FOS对比散点图 ====================
        ax1 = fig.add_subplot(2, 2, 1)
        
        # 根据稳定性着色
        colors_scatter = []
        for _, row in df_valid.iterrows():
            fos = row[seepage_fos_col]
            if fos < FOS_THRESHOLDS['at_risk']:
                colors_scatter.append(COLORS['at_risk'])
            elif fos <= FOS_THRESHOLDS['stable']:
                colors_scatter.append(COLORS['marginal'])
            else:
                colors_scatter.append(COLORS['stable'])
        
        ax1.scatter(df_valid[normal_fos_col], df_valid[seepage_fos_col], 
                   c=colors_scatter, alpha=0.6, s=50, edgecolor='white', linewidth=0.5)
        
        # 1:1参考线
        lims = [
            min(df_valid[normal_fos_col].min(), df_valid[seepage_fos_col].min()) - 0.1,
            max(df_valid[normal_fos_col].max(), df_valid[seepage_fos_col].max()) + 0.1
        ]
        ax1.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, label='1:1 Line')
        
        # 不稳定性阈值线
        ax1.axhline(y=FOS_THRESHOLDS['at_risk'], color=COLORS['at_risk'], 
                   linestyle=':', linewidth=2, label=f'At Risk (FOS<{FOS_THRESHOLDS["at_risk"]})')
        ax1.axhline(y=FOS_THRESHOLDS['stable'], color=COLORS['marginal'], 
                   linestyle=':', linewidth=2, label=f'Marginal (FOS<{FOS_THRESHOLDS["stable"]})')
        
        ax1.set_xlabel('Normal FOS', fontsize=10.5, fontweight='bold')
        ax1.set_ylabel('Seepage FOS', fontsize=10.5, fontweight='bold')
        ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
        ax1.legend(loc='upper right', fontsize=10.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        
        # 统计信息
        total = len(df_valid)
        seepage_higher = (df_valid[seepage_fos_col] > df_valid[normal_fos_col]).sum()
        ax1.text(0.05, 0.95, f'n={total}\nSeepage>Normal: {seepage_higher} ({seepage_higher/total*100:.1f}%)',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ==================== 子图b: FOS变化分布与正态性检验 (改进) ====================
        ax2 = fig.add_subplot(2, 2, 2)
        
        # 显式转换为numpy数组以避免lint错误
        fos_change = np.array(df_valid['FOS_Change'].values, dtype=float)
        
        # 直方图
        n, bins, patches = ax2.hist(fos_change, bins=50, density=True, 
                                    color=COLORS['primary'], edgecolor='white', alpha=0.7, label='Actual Distribution')
        
        # 正态分布拟合曲线 (以实际均值和标准差拟合)
        mu, std = stats.norm.fit(fos_change)
        x_norm = np.linspace(float(fos_change.min()), float(fos_change.max()), 200)
        y_norm = stats.norm.pdf(x_norm, mu, std)
        ax2.plot(x_norm, y_norm, 'r-', linewidth=2.5, label=f'Normal Fit (μ={mu:.3f}, σ={std:.3f})')
        
        # 理论正态分布 (中心为0)
        y_norm_zero = stats.norm.pdf(x_norm, 0, std)
        ax2.plot(x_norm, y_norm_zero, 'g--', linewidth=2, alpha=0.7, label=f'Theoretical (μ=0, σ={std:.3f})')
        
        # Shapiro-Wilk正态性检验 (样本量限制为5000)
        sample_size = min(len(fos_change), 5000)
        _, shapiro_p = stats.shapiro(np.random.choice(fos_change, sample_size, replace=False))
        
        # K-S检验 (针对实际拟合的正态分布)
        _, ks_p = stats.kstest(fos_change, 'norm', args=(mu, std))
        
        # 参考线 - 移除红色阈值线，只保留零线和均值线
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Zero Change')
        ax2.axvline(x=mu, color='blue', linestyle=':', linewidth=2, label=f'Mean: {mu:.3f}')
        
        ax2.set_xlabel('FOS Change (Seepage - Normal)', fontsize=10.5, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=10.5, fontweight='bold')
        ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
        ax2.legend(loc='upper right', fontsize=10.5)
        ax2.grid(True, alpha=0.3)
        
        # 正态性检验结果
        normality_text = f'Shapiro p={shapiro_p:.4f}\nK-S p={ks_p:.4f}'
        if shapiro_p < 0.05:
            normality_text += '\nNon-normal'
        else:
            normality_text += '\nNormal'
        ax2.text(0.05, 0.85, normality_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # ==================== 子图c: 模型预测散点图 (原生绘制) ====================
        ax3 = fig.add_subplot(2, 2, 3)
        
        scatter_data_path = MODELS_DIR / 'model_scatter_data.json'
        
        if scatter_data_path.exists():
            try:
                with open(scatter_data_path, 'r', encoding='utf-8') as f:
                    scatter_data = json.load(f)
                
                model_colors = {
                    'Model1': COLORS.get('model1', '#3498db'),
                    'Model2': COLORS.get('model2', '#27ae60'),
                    'Model3': COLORS.get('model3', '#f39c12'),
                    'Model4': COLORS.get('model4', '#e74c3c'),
                    'Model5': COLORS.get('model5', '#9b59b6'),
                }
                
                all_y = []
                for model_name, data in scatter_data.items():
                    y_test = np.array(data['y_test'])
                    y_pred = np.array(data['y_pred'])
                    r2 = data['r2']
                    
                    all_y.extend(y_test)
                    all_y.extend(y_pred)
                    
                    is_priority = model_name in ['Model4', 'Model5']
                    size = 50 if is_priority else 30
                    edge = 'black' if is_priority else 'white'
                    edge_width = 1.0 if is_priority else 0.5
                    alpha = 0.7 if is_priority else 0.5
                    
                    ax3.scatter(y_test, y_pred, alpha=alpha, s=size, 
                               c=model_colors.get(model_name, '#666666'),
                               label=f"{model_name} (R²={r2:.3f})",
                               edgecolors=edge, linewidths=edge_width)
                
                # 1:1 line and thresholds
                if all_y:
                    all_y = np.array(all_y)
                    min_val = float(all_y.min()) - 0.1
                    max_val = float(all_y.max()) + 0.1
                    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
                    
                    ax3.axhline(y=FOS_THRESHOLDS['stable'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
                    ax3.axvline(x=FOS_THRESHOLDS['stable'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
                    ax3.axhline(y=FOS_THRESHOLDS['marginal'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                    ax3.axvline(x=FOS_THRESHOLDS['marginal'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                    
                    ax3.set_xlim(min_val, max_val)
                    ax3.set_ylim(min_val, max_val)
                
                ax3.set_xlabel('Actual FOS', fontsize=10.5, fontweight='bold')
                ax3.set_ylabel('Predicted FOS', fontsize=10.5, fontweight='bold')
                ax3.legend(loc='upper right', fontsize=10.5, framealpha=0.9)
                ax3.grid(True, alpha=0.3)
                ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
                
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error plotting data:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.axis('off')
        else:
            # 如果数据不存在，显示提示信息
            ax3.text(0.5, 0.5, 'Model scatter data\nnot yet generated.\nRun step3 first.', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10.5)
            ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
            ax3.axis('off')
        
        # ==================== 子图d: 气候区分布平滑曲线 (互换坐标轴) ====================
        
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 为每个气候区创建平滑的分布曲线
        from scipy.interpolate import make_interp_spline
        from scipy.ndimage import gaussian_filter1d
        
        # 准备数据 - 需要climate相关变量
        climate_order = ['A', 'B', 'C', 'D']
        available_zones = [z for z in climate_order if z in df_valid['Climate_Zone'].unique()]
        df_climate = df_valid[df_valid['Climate_Zone'].isin(available_zones)]
        
        # 创建更密集的bins用于平滑
        bins = np.linspace(df_valid[seepage_fos_col].min(), df_valid[seepage_fos_col].max(), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # 为每个气候区绘制平滑曲线
        for zone in available_zones:
            zone_data = df_climate[df_climate['Climate_Zone']==zone][seepage_fos_col].values
            if len(zone_data) > 5:  # 至少需要一些数据点
                hist, _ = np.histogram(zone_data, bins=bins)
                
                # 使用高斯滤波器平滑曲线
                hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
                
                # 互换 x 和 y 坐标 - 原来是 (bin_centers, hist)，现在变为 (hist, bin_centers)
                zone_color = KOPPEN_CLIMATE_ZONES[zone]['color']
                zone_label = f"{zone} ({KOPPEN_CLIMATE_ZONES[zone]['name']})"
                ax4.plot(hist_smooth, bin_centers, linewidth=2.5, 
                        color=zone_color, label=zone_label, alpha=0.8)
                # 可选：填充曲线下方区域
                ax4.fill_betweenx(bin_centers, 0, hist_smooth, 
                                 color=zone_color, alpha=0.2)
        
        # 添加不稳定性阈值线 (y轴)
        ax4.axhline(y=FOS_THRESHOLDS['at_risk'], color=COLORS['at_risk'], 
                   linestyle='--', linewidth=2, alpha=0.9, label=f'At Risk (<{FOS_THRESHOLDS["at_risk"]})')
        ax4.axhline(y=FOS_THRESHOLDS['stable'], color=COLORS['stable'], 
                   linestyle='--', linewidth=2, alpha=0.9, label=f'Stable (≥{FOS_THRESHOLDS["stable"]})')
        
        # 互换后的标签
        ax4.set_ylabel('Seepage FOS', fontsize=10.5, fontweight='bold')  # y轴现在是FOS
        ax4.set_xlabel('Count', fontsize=10.5, fontweight='bold')  # x轴现在是计数
        ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')
        ax4.legend(loc='upper right', fontsize=10.5, framealpha=0.9)
        ax4.grid(True, alpha=0.3, axis='both')

        
        # 调整布局
        fig.tight_layout()
        fig.savefig(self.output_dir / 'fov_analysis_combined.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("  ✓ Combined figure saved: fov_analysis_combined.png")
        
        # 保存统计结果
        self.results = {
            'total_dams': total,
            'seepage_higher_count': int(seepage_higher),
            'seepage_higher_pct': seepage_higher/total*100,
            'fos_change_mean': mu,
            'fos_change_std': std,
            'shapiro_p': shapiro_p,
            'ks_p': ks_p,
            'is_normal': shapiro_p >= 0.05
        }
        
        return self.results
    
    def save_results(self):
        """保存分析结果"""
        if not self.results:
            return
        
        # 转换numpy类型为Python原生类型
        results_serializable = {}
        for k, v in self.results.items():
            if isinstance(v, (np.floating, np.integer)):
                results_serializable[k] = float(v)
            elif isinstance(v, np.bool_):
                results_serializable[k] = bool(v)
            else:
                results_serializable[k] = v
        
        with open(self.output_dir / 'fos_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def run(self, skip_combined=False):
        """Execute full analysis
        
        Args:
            skip_combined: If True, skip creating fov_analysis_combined.png
        """
        print_section_header("Step 2: FOS Analysis & Climate Visualization")
        
        # 确保目录存在
        ensure_directories()
        
        # 加载数据
        if not self.load_data():
            return None
        
        # 分配气候区
        self.assign_climate_zones()
        
        # 创建参数相关性图表
        self.create_parameter_correlation_figure()
        
        # 创建组合图表（如果不跳过）
        results = None
        if not skip_combined:
            results = self.create_combined_figure()
            # 保存结果
            self.save_results()
        
        print_subsection("Summary")
        if results:
            print(f"  Total dams analyzed: {results['total_dams']}")
            print(f"  Seepage FOS > Normal FOS: {results['seepage_higher_count']} ({results['seepage_higher_pct']:.1f}%)")
            print(f"  FOS change: μ={results['fos_change_mean']:.4f}, σ={results['fos_change_std']:.4f}")
            print(f"  Normal distribution: {'Yes' if results['is_normal'] else 'No'}")
        
        return results


def main():
    """Main function"""
    os.chdir(PROJECT_ROOT)
    
    analyzer = FOSAnalyzer()
    results = analyzer.run()
    
    print("\n" + "=" * 70)
    print("Step 2 completed!")
    print("Next: Run step3_model_comparison.py")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
