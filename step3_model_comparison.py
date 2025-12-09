"""
Step 3: Model Comparison for Dam Safety Prediction

Functionality:
1. Four-model comparison framework tests prediction capability of different electrical signal combinations for FOV
2. Each model tests six machine learning algorithms with multi-threading support
3. Uses 80/20 train-test split
4. Computes R2, RMSE, MAE metrics
5. Auto-selects best algorithm and generates performance comparison plots
6. Saves best model predictions to P2.csv

Four model configurations:
- Model 1: Normal Self-Potential -> Normal FOV (baseline)
- Model 2: Seepage Self-Potential -> Seepage FOV
- Model 3: Coupling Electric Field -> Seepage FOV
- Model 4: Seepage Self-Potential + Coupling Electric Field -> Seepage FOV
"""

import os
import json
import warnings
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import configuration
from config import (
    PROJECT_ROOT, P1_FILE, P2_FILE, MODELS_DIR,
    MODEL_CONFIGS, ML_ALGORITHMS, COLORS, PLOT_CONFIG,
    print_section_header, print_subsection, ensure_directories
)

warnings.filterwarnings('ignore')

# Set plot parameters
plt.rcParams.update(PLOT_CONFIG)
sns.set_theme(style='whitegrid')


class ModelComparison:
    """Model comparison analyzer with multi-threading support"""
    
    def __init__(self, data_file=None, num_threads=4):
        """
        Initialize
        
        Args:
            data_file: Input data file path
            num_threads: Number of threads for parallel processing
        """
        self.data_file = Path(data_file) if data_file else P1_FILE
        self.output_dir = MODELS_DIR
        self.data = None
        self.results = {}
        self.test_size = 0.2
        self.random_state = 42
        self.num_threads = num_threads
        self.lock = threading.Lock()
        
    def load_data(self):
        """Load data and compute derived target variables"""
        if not self.data_file.exists():
            print(f"Error: Data file not found: {self.data_file}")
            return False
        
        try:
            self.data = pd.read_csv(self.data_file, encoding='utf-8-sig')
            print(f"Data loaded: {len(self.data)} records")
            
            # Compute derived target variables for safety degradation analysis
            self._compute_safety_metrics()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _compute_safety_metrics(self):
        """Compute safety degradation metrics"""
        from config import calculate_sdi
        
        print_subsection("Computing Safety Degradation Metrics")
        
        normal_fos_col = self.find_column(['Normal', 'FOS', 'Maximum'])
        seepage_fos_col = self.find_column(['Seepage', 'FOS', 'Maximum'])
        
        if not normal_fos_col or not seepage_fos_col:
            print("  Warning: FOS columns not found")
            return
        
        # 1. FOS Degradation Rate (SDI)
        # 正值表示渗漏时FOS上升（需要排水通道维持），实际安全性下降
        self.data['FOS_Degradation_Rate'] = self.data.apply(
            lambda row: calculate_sdi(row[normal_fos_col], row[seepage_fos_col]),
            axis=1
        )
        
        # 2. Actual Safety Factor (考虑渗漏导致的真实安全性下降)
        # 假设：如果没有排水通道，渗漏会导致FOS下降相同比例
        # Actual_Safety_Factor = Normal_FOS * (1 - |SDI|)
        # 这反映了渗漏时的真实安全性（不依赖排水通道）
        self.data['Actual_Safety_Factor'] = self.data.apply(
            lambda row: row[normal_fos_col] * (1 - abs(row['FOS_Degradation_Rate'])) 
            if pd.notna(row['FOS_Degradation_Rate']) else np.nan,
            axis=1
        )
        
        # Print statistics
        sdi_stats = self.data['FOS_Degradation_Rate'].describe()
        asf_stats = self.data['Actual_Safety_Factor'].describe()
        
        print(f"\n  FOS Degradation Rate (SDI) statistics:")
        print(f"    Mean: {sdi_stats['mean']:.4f}")
        print(f"    Median: {sdi_stats['50%']:.4f}")
        print(f"    Range: [{sdi_stats['min']:.4f}, {sdi_stats['max']:.4f}]")
        
        print(f"\n  Actual Safety Factor statistics:")
        print(f"    Mean: {asf_stats['mean']:.4f}")
        print(f"    Median: {asf_stats['50%']:.4f}")
        print(f"    Range: [{asf_stats['min']:.4f}, {asf_stats['max']:.4f}]")
        
        # Compare with Normal and Seepage FOS
        normal_mean = self.data[normal_fos_col].mean()
        seepage_mean = self.data[seepage_fos_col].mean()
        actual_mean = asf_stats['mean']
        
        print(f"\n  FOS Comparison:")
        print(f"    Normal FOS mean:   {normal_mean:.4f}")
        print(f"    Seepage FOS mean:  {seepage_mean:.4f} (with drainage)")
        print(f"    Actual SF mean:    {actual_mean:.4f} (true safety with seepage)")
        print(f"    Safety degradation: {(normal_mean - actual_mean) / normal_mean * 100:.1f}%")
    
    def find_column(self, keywords):
        """Find column by keywords"""
        for col in self.data.columns:
            if all(kw.lower() in col.lower() for kw in keywords):
                return col
        return None
    
    def get_model_columns(self, model_config):
        """
        Get feature columns and target column for a model
        
        Args:
            model_config: Model configuration dict
        
        Returns:
            (feature_columns, target_column) or None
        """
        # 查找目标列
        target_keywords = model_config['target'].split()
        target_col = self.find_column(target_keywords)
        
        if not target_col:
            return None, None
        
        # 查找特征列
        feature_cols = []
        for feature in model_config['features']:
            feature_keywords = feature.split()
            col = self.find_column(feature_keywords)
            if col:
                feature_cols.append(col)
        
        if not feature_cols:
            return None, None
        
        return feature_cols, target_col
    
    def get_algorithms(self):
        """Get all machine learning algorithm instances"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
    
    def evaluate_model(self, model_name, model_config):
        """
        Evaluate a single model configuration
        
        Args:
            model_name: Model name
            model_config: Model configuration
        
        Returns:
            Result dict
        """
        print(f"\n  Evaluating {model_name}: {model_config['description']}")
        
        # 获取列
        feature_cols, target_col = self.get_model_columns(model_config)
        
        if not feature_cols or not target_col:
            print(f"    ⚠ Required columns not found")
            return None
        
        print(f"    Features: {feature_cols}")
        print(f"    Target: {target_col}")
        
        # 准备数据
        df_valid = self.data[feature_cols + [target_col]].dropna()
        
        if len(df_valid) < 10:
            print(f"    ⚠ Insufficient data: {len(df_valid)} samples")
            return None
        
        X = df_valid[feature_cols].values
        y = df_valid[target_col].values
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"    Data: {len(df_valid)} samples (Train: {len(X_train)}, Test: {len(X_test)})")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 测试所有算法
        algorithms = self.get_algorithms()
        algo_results = {}
        best_predictions = None
        best_r2 = -np.inf
        best_algo_name = None
        
        for algo_name, algo in algorithms.items():
            try:
                # 训练
                algo.fit(X_train_scaled, y_train)
                
                # 预测
                y_pred = algo.predict(X_test_scaled)
                
                # 计算指标
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                algo_results[algo_name] = {
                    'R2': r2,
                    'RMSE': rmse,
                    'MAE': mae
                }
                
                # Save best algorithm predictions
                if r2 > best_r2:
                    best_r2 = r2
                    best_algo_name = algo_name
                    best_predictions = {
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                
            except Exception as e:
                print(f"    Warning: {algo_name} failed: {e}")
                continue
        
        if not algo_results:
            return None
        
        print(f"    Best algorithm: {best_algo_name} (R² = {best_r2:.4f})")
        
        return {
            'model_name': model_name,
            'description': model_config['description'],
            'features': feature_cols,
            'target': target_col,
            'n_samples': len(df_valid),
            'algorithms': algo_results,
            'best_algorithm': best_algo_name,
            'best_r2': best_r2,
            'predictions': best_predictions  # Save predictions for plotting
        }
    
    def run_comparison(self):
        """Execute model comparison using unified test set with multi-threading"""
        print_subsection("Model Comparison (Unified Test Set)")
        
        # First determine unified test set indices (based on Seepage FOS, as Model2-4 all predict it)
        seepage_fos_col = self.find_column(['Seepage', 'FOS', 'Maximum'])
        if not seepage_fos_col:
            print("  Warning: Cannot find Seepage FOS column for unified split")
            return
        
        # Get all valid sample indices
        valid_indices = self.data[seepage_fos_col].dropna().index.tolist()
        n_samples = len(valid_indices)
        
        # Unified train/test split
        np.random.seed(self.random_state)
        shuffled = np.random.permutation(valid_indices)
        n_test = int(n_samples * self.test_size)
        self.test_indices = set(shuffled[:n_test])
        self.train_indices = set(shuffled[n_test:])
        
        print(f"  Unified split: {len(self.train_indices)} train, {len(self.test_indices)} test samples")
        
        for model_name, model_config in MODEL_CONFIGS.items():
            result = self.evaluate_model_unified(model_name, model_config)
            if result:
                self.results[model_name] = result
    
    def evaluate_model_unified(self, model_name, model_config):
        """
        Evaluate a single model configuration using unified test set
        
        Args:
            model_name: Model name
            model_config: Model configuration
        
        Returns:
            Result dict
        """
        print(f"\n  Evaluating {model_name}: {model_config['description']}")
        
        # Get columns
        feature_cols, target_col = self.get_model_columns(model_config)
        
        if not feature_cols or not target_col:
            print(f"    Warning: Required columns not found")
            return None
        
        print(f"    Features: {feature_cols}")
        print(f"    Target: {target_col}")
        
        # Prepare data (only keep rows where model columns are not null)
        df_model = self.data[feature_cols + [target_col]].dropna()
        model_indices = set(df_model.index.tolist())
        
        # Intersection: model valid samples AND unified train/test indices
        train_idx = list(model_indices & self.train_indices)
        test_idx = list(model_indices & self.test_indices)
        
        if len(train_idx) < 10 or len(test_idx) < 5:
            print(f"    Warning: Insufficient data after intersection: train={len(train_idx)}, test={len(test_idx)}")
            return None
        
        X_train = self.data.loc[train_idx, feature_cols].values
        y_train = self.data.loc[train_idx, target_col].values
        X_test = self.data.loc[test_idx, feature_cols].values
        y_test = self.data.loc[test_idx, target_col].values
        
        print(f"    Data: Train={len(train_idx)}, Test={len(test_idx)}")
        
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test all algorithms with thread safety
        algorithms = self.get_algorithms()
        algo_results = {}
        best_predictions = None
        best_r2 = -np.inf
        best_algo_name = None
        
        for algo_name, algo in algorithms.items():
            try:
                algo.fit(X_train_scaled, y_train)
                y_pred = algo.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                algo_results[algo_name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_algo_name = algo_name
                    best_predictions = {
                        'y_test': y_test.copy(),
                        'y_pred': y_pred.copy(),
                        'test_indices': test_idx  # Save indices for alignment
                    }
            except Exception as e:
                print(f"    Warning: {algo_name} failed: {e}")
                continue
        
        if not algo_results:
            return None
        
        print(f"    Best algorithm: {best_algo_name} (R² = {best_r2:.4f})")
        
        return {
            'model_name': model_name,
            'description': model_config['description'],
            'features': feature_cols,
            'target': target_col,
            'n_samples': len(train_idx) + len(test_idx),
            'algorithms': algo_results,
            'best_algorithm': best_algo_name,
            'best_r2': best_r2,
            'predictions': best_predictions
        }
    
    def create_performance_heatmap(self):
        """Create performance comparison heatmap"""
        if not self.results:
            return
        
        print_subsection("Creating Performance Heatmap")
        
        # Prepare data
        models = list(self.results.keys())
        algorithms = list(self.get_algorithms().keys())
        
        # R2 matrix
        r2_matrix = []
        for model in models:
            row = []
            for algo in algorithms:
                if algo in self.results[model]['algorithms']:
                    row.append(self.results[model]['algorithms'][algo]['R2'])
                else:
                    row.append(np.nan)
            r2_matrix.append(row)
        
        r2_df = pd.DataFrame(r2_matrix, index=models, columns=algorithms)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(
            r2_df,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={'label': 'R² Score'},
            ax=ax
        )
        
        # ax.set_title('Model Performance Comparison (R² Score)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Model Configuration', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("  Heatmap saved")
    
    def create_bar_chart(self):
        """Create best performance bar chart"""
        if not self.results:
            return
        
        print_subsection("Creating Bar Chart")
        
        # Prepare data
        models = []
        r2_scores = []
        best_algos = []
        
        for model_name, result in self.results.items():
            models.append(model_name)
            r2_scores.append(result['best_r2'])
            best_algos.append(result['best_algorithm'])
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['stable'], COLORS['marginal']]
        bars = ax.bar(models, r2_scores, color=colors[:len(models)], edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, r2, algo in zip(bars, r2_scores, best_algos):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{r2:.3f}\n({algo[:10]}...)',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model Configuration', fontsize=12)
        ax.set_ylabel('Best R² Score', fontsize=12)
        # ax.set_title('Best Performance by Model Configuration', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'model_comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("  Bar chart saved")
    
    def create_scatter_plots(self):
        """Create scatter plot comparison for all models with FOS thresholds - supports 5 models"""
        if not self.results:
            return
        
        print_subsection("Creating Model Comparison Scatter Plot")
        
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        model_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        
        n_models = len(self.results)
        # Create grid: 2 columns, enough rows for all models
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Use appropriate figure size based on number of models
        from config import FIGURE_SIZES, FOS_THRESHOLDS
        if n_models <= 4:
            figsize = FIGURE_SIZES['model_scatter']
        else:
            figsize = FIGURE_SIZES['model_scatter_5']
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Find global best R2 for highlighting
        all_r2 = [r.get('best_r2', -float('inf')) for r in self.results.values() if r.get('best_r2') is not None]
        best_r2 = max(all_r2) if all_r2 else -float('inf')

        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            if 'predictions' not in result or result['predictions'] is None:
                ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_labels[idx]} {model_name}', fontsize=12, fontweight='bold')
                continue
            
            y_test = result['predictions']['y_test']
            y_pred = result['predictions']['y_pred']
            r2 = result['best_r2']
            rmse = result['algorithms'][result['best_algorithm']]['RMSE']
            mae = result['algorithms'][result['best_algorithm']]['MAE']
            algo = result['best_algorithm']
            
            # Visual indicator: best model gets larger markers and thicker edge
            is_best = (r2 == best_r2)
            marker_size = 70 if is_best else 50
            edge_width = 2.0 if is_best else 0.5
            edge_color = 'gold' if is_best else 'white'
            
            # Scatter plot
            color_idx = idx % len(colors_list)
            ax.scatter(y_test, y_pred, alpha=0.7, s=marker_size, c=colors_list[color_idx], 
                      edgecolors=edge_color, linewidths=edge_width)
            
            # 1:1 reference line
            min_val = min(y_test.min(), y_pred.min()) - 0.1
            max_val = max(y_test.max(), y_pred.max()) + 0.1
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.6, label='1:1 Line')
            
            # Add FOS stability threshold lines (only for FOS-related targets)
            if 'FOS' in result['target'] or 'Safety' in result['target']:
                ax.axhline(y=FOS_THRESHOLDS['stable'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axvline(x=FOS_THRESHOLDS['stable'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axhline(y=FOS_THRESHOLDS['marginal'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axvline(x=FOS_THRESHOLDS['marginal'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            
            # Fit line
            slope, intercept, r, p, se = scipy_stats.linregress(y_test, y_pred)
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r-', lw=2, alpha=0.8, label=f'Fit (slope={slope:.2f})')
            
            # Labels and title
            target_label = result['target'].replace('_', ' ')
            # Update x-axis label to be clearer and more descriptive
            if 'FOS' in target_label or 'Safety' in target_label:
                x_label = 'Measured Factor of Safety (FOS)'
                y_label = 'Predicted Factor of Safety (FOS)'
            elif 'Normal' in target_label:
                x_label = f'Measured {target_label}'
                y_label = f'Predicted {target_label}'
            elif 'Seepage' in target_label:
                x_label = f'Measured {target_label}'
                y_label = f'Predicted {target_label}'
            else:
                x_label = f'Measured {target_label}'
                y_label = f'Predicted {target_label}'
            
            ax.set_xlabel(x_label, fontsize=10.5, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=10.5, fontweight='bold')
            
            # Add model label with best indicator inside the plot (top-left)
            title_text = f'{model_labels[idx]} {model_name}'
            if is_best:
                title_text += ' ★'  # Star for best model
            # Place title/caption at top-left inside the plot
            ax.text(0.03, 0.97, title_text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
            
            # Combine legend and stats: Add stats to a custom legend entry
            # User request: "legend and r,p values please draw in one legend"
            
            # Create a custom legend handle for the stats
            from matplotlib.lines import Line2D
            
            # Format p-value
            if p < 0.001:
                p_str = 'p < 0.001'
            else:
                p_str = f'p = {p:.3f}'
            
            # Create legend elements with all info in one legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f'Data (n={len(y_test)})',
                      markerfacecolor=colors_list[color_idx], markersize=8, 
                      markeredgecolor=edge_color, markeredgewidth=edge_width),
                Line2D([0], [0], color='k', lw=2, linestyle='--', label='1:1 Line'),
                Line2D([0], [0], color='r', lw=2, label=f'Fit Line'),
                Line2D([0], [0], linestyle='None', label=f'R² = {r2:.3f}, {p_str}'),
                Line2D([0], [0], linestyle='None', label=f'RMSE = {rmse:.3f}'),
                Line2D([0], [0], linestyle='None', label=f'MAE = {mae:.3f}')
            ]
            
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        # fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(self.output_dir / 'model_scatter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("  Model comparison scatter plot saved")

    def create_combined_model_scatter(self):
        """Create combined scatter plot with all models, highlighting Model4, with FOS thresholds"""
        if not self.results:
            return

        print_subsection("Creating Combined Model Scatter")
        
        from config import COLORS, FOS_THRESHOLDS
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {
            'Model1': COLORS.get('model1', '#3498db'),  # Blue
            'Model2': COLORS.get('model2', '#27ae60'),  # Green
            'Model3': COLORS.get('model3', '#f39c12'),  # Yellow/Orange
            'Model4': COLORS.get('model4', '#e74c3c'),  # Red
            'Model5': COLORS.get('model5', '#9b59b6'),  # Purple
        }

        for model_name, result in self.results.items():
            if 'predictions' not in result or result['predictions'] is None:
                continue
            y_test = np.array(result['predictions']['y_test'])
            y_pred = np.array(result['predictions']['y_pred'])

            # Highlight best model (Model4 or Model5 priority)
            is_priority = model_name in ['Model4', 'Model5']
            marker = 'o'
            size = 70 if is_priority else 40
            edge = 'black' if is_priority else 'white'
            edge_width = 1.5 if is_priority else 0.5

            ax.scatter(y_test, y_pred, alpha=0.7, s=size, c=colors.get(model_name, '#666666'),
                       label=f"{model_name} (R²={result['best_r2']:.3f})", 
                       edgecolors=edge, linewidths=edge_width)

        # 1:1 reference line
        all_y = []
        for r in self.results.values():
            if 'predictions' in r and r['predictions']:
                all_y.append(r['predictions']['y_test'])
                all_y.append(r['predictions']['y_pred'])
        if all_y:
            all_y = np.concatenate([np.array(a) for a in all_y])
            min_val = float(all_y.min()) - 0.1
            max_val = float(all_y.max()) + 0.1
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
            
            # Add FOS stability threshold lines
            ax.axhline(y=FOS_THRESHOLDS['stable'], color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f"Stable (>{FOS_THRESHOLDS['stable']})")
            ax.axvline(x=FOS_THRESHOLDS['stable'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axhline(y=FOS_THRESHOLDS['marginal'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f"At Risk (<{FOS_THRESHOLDS['marginal']})")
            ax.axvline(x=FOS_THRESHOLDS['marginal'], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

        ax.set_xlabel('Actual FOS', fontsize=10.5, fontweight='bold')
        ax.set_ylabel('Predicted FOS', fontsize=10.5, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10.5, framealpha=0.95)

        out_file = self.output_dir / 'model_comparison_combined.png'
        fig.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save plotting data for Step 2 combined figure
        scatter_data = {}
        for model_name, result in self.results.items():
            if 'predictions' in result and result['predictions']:
                scatter_data[model_name] = {
                    'y_test': result['predictions']['y_test'].tolist() if hasattr(result['predictions']['y_test'], 'tolist') else list(result['predictions']['y_test']),
                    'y_pred': result['predictions']['y_pred'].tolist() if hasattr(result['predictions']['y_pred'], 'tolist') else list(result['predictions']['y_pred']),
                    'r2': float(result['best_r2'])
                }
        
        with open(self.output_dir / 'model_scatter_data.json', 'w', encoding='utf-8') as f:
            json.dump(scatter_data, f)
        print('  Scatter plot data saved to model_scatter_data.json')

        print('  Combined model scatter saved')

    def save_results(self):
        """Save results to JSON and CSV"""
        if not self.results:
            return
        
        # Prepare serializable results (remove ndarray)
        results_for_json = {}
        for model_name, result in self.results.items():
            results_for_json[model_name] = {
                'model_name': result['model_name'],
                'description': result['description'],
                'features': result['features'],
                'target': result['target'],
                'n_samples': result['n_samples'],
                'algorithms': result['algorithms'],
                'best_algorithm': result['best_algorithm'],
                'best_r2': result['best_r2']
                # Exclude 'predictions' as it contains ndarray
            }
        
        # Save detailed results as JSON
        with open(self.output_dir / 'model_comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
        # Save summary as CSV
        summary_records = []
        for model_name, result in self.results.items():
            for algo_name, metrics in result['algorithms'].items():
                summary_records.append({
                    'Model': model_name,
                    'Algorithm': algo_name,
                    'R2': metrics['R2'],
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE']
                })
        
        df_summary = pd.DataFrame(summary_records)
        df_summary.to_csv(self.output_dir / 'model_comparison_summary.csv', 
                         index=False, encoding='utf-8-sig')
        
        print(f"\nResults saved to {self.output_dir}")
    
    def save_fov_predictions(self):
        """
        Use best degradation model to predict FOV (Factor of Vulnerability) for all samples
        Save predictions to P2.csv
        """
        print_subsection("Saving FOV Predictions to P2.csv")
        
        # Find best degradation model (prefer Model5, then Model4)
        best_model_name = None
        for model_name in ['Model5', 'Model4', 'Model3', 'Model2']:
            if model_name in self.results:
                best_model_name = model_name
                break
        
        if not best_model_name:
            print("  Warning: No suitable model found for prediction")
            return None
        
        best_result = self.results[best_model_name]
        feature_cols = best_result['features']
        target_col = best_result['target']
        best_algo_name = best_result['best_algorithm']
        prediction_type = MODEL_CONFIGS[best_model_name].get('prediction_type', 'degradation')
        
        print(f"  Using {best_model_name} ({prediction_type}) with {best_algo_name}")
        
        # Prepare full dataset for prediction
        df_valid = self.data[feature_cols].dropna()
        valid_indices = df_valid.index.tolist()
        
        # Re-train best model using training set
        train_idx = list(set(valid_indices) & self.train_indices)
        X_train = self.data.loc[train_idx, feature_cols].values
        y_train = self.data.loc[train_idx, target_col].values
        
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train best algorithm
        algorithms = self.get_algorithms()
        best_algo = algorithms[best_algo_name]
        best_algo.fit(X_train_scaled, y_train)
        
        # Predict for all valid samples
        X_all = self.data.loc[valid_indices, feature_cols].values
        X_all_scaled = scaler.transform(X_all)
        predictions = best_algo.predict(X_all_scaled)
        
        # Copy original data
        df_output = self.data.copy()
        
        # Compute final safety factor based on prediction type
        normal_fos_col = self.find_column(['Normal', 'FOS', 'Maximum'])
        
        if prediction_type == 'degradation':
            # Predicted degradation rate -> compute FOV (Factor of Vulnerability)
            # FOV = Normal_FOS * (1 - |predicted_degradation_rate|)
            df_output['Predicted_Degradation_Rate'] = np.nan
            df_output['FOV'] = np.nan
            
            for idx, pred in zip(valid_indices, predictions):
                df_output.loc[idx, 'Predicted_Degradation_Rate'] = pred
                normal_fos = df_output.loc[idx, normal_fos_col]
                if pd.notna(normal_fos):
                    df_output.loc[idx, 'FOV'] = normal_fos * (1 - abs(pred))
        
        elif prediction_type == 'actual_safety':
            # Direct prediction of FOV (Factor of Vulnerability)
            df_output['FOV'] = np.nan
            for idx, pred in zip(valid_indices, predictions):
                df_output.loc[idx, 'FOV'] = pred
        
        else:  # direct FOS prediction
            df_output['FOV'] = np.nan
            for idx, pred in zip(valid_indices, predictions):
                df_output.loc[idx, 'FOV'] = pred
        
        # Save to P2.csv
        df_output.to_csv(P2_FILE, index=False, encoding='utf-8-sig')
        
        fov_values = df_output['FOV'].dropna()
        print(f"  FOV (Factor of Vulnerability) predictions saved to {P2_FILE}")
        print(f"    Total samples: {len(fov_values)}")
        print(f"    FOV range: [{fov_values.min():.4f}, {fov_values.max():.4f}]")
        print(f"    FOV mean: {fov_values.mean():.4f}")
        
        # Compare with actual safety factor (ground truth)
        if 'Actual_Safety_Factor' in df_output.columns:
            actual_sf = df_output.loc[valid_indices, 'Actual_Safety_Factor']
            correlation = actual_sf.corr(fov_values)
            print(f"    Correlation with ground truth: {correlation:.4f}")
        
        return P2_FILE
    
    def run(self):
        """Execute full model comparison analysis"""
        print_section_header("Step 3: Model Comparison")
        
        # Ensure directories exist
        ensure_directories()
        
        # Load data
        if not self.load_data():
            return None
        
        # Execute comparison
        self.run_comparison()
        
        # Create visualizations - keep only model_scatter_comparison
        print_subsection("Creating Visualizations")
        self.create_scatter_plots()
        # Combined plot needed for embedding in fov_analysis_combined.png (will be deleted after use)
        self.create_combined_model_scatter()
        
        # Save results
        self.save_results()
        
        # Save FOV predictions to P2.csv
        self.save_fov_predictions()
        
        # Print summary
        print_subsection("Summary")
        print(f"  Models evaluated: {len(self.results)}")
        
        if self.results:
            best_model = max(self.results.keys(), key=lambda k: self.results[k]['best_r2'])
            best_r2 = self.results[best_model]['best_r2']
            best_algo = self.results[best_model]['best_algorithm']
            print(f"  Best overall: {best_model} with {best_algo} (R² = {best_r2:.4f})")
        
        return self.results


def main():
    """Main function"""
    os.chdir(PROJECT_ROOT)
    
    comparison = ModelComparison()
    results = comparison.run()
    
    print("\n" + "=" * 70)
    print("Step 3 completed!")
    print("Next: Run step4_climate_analysis.py")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
