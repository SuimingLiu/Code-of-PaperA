"""
Dam Safety Analysis - Configuration File
堤坝安全分析 - 配置文件

This file contains all configuration parameters for the dam safety analysis pipeline.
本文件包含堤坝安全分析流水线的所有配置参数。
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# Directory Configuration / 目录配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Output directories / 输出目录
CORRELATION_DIR = RESULTS_DIR / 'correlations'
MODELS_DIR = RESULTS_DIR / 'models'
CLIMATE_DIR = RESULTS_DIR / 'climate'

# Data files / 数据文件
P1_FILE = DATA_DIR / 'P1.csv'  # Step1 output: processed dam parameters
P2_FILE = DATA_DIR / 'P2.csv'  # Step3 output: with model predictions (FOV)
P3_FILE = DATA_DIR / 'P3.csv'  # Step4 input: with climate zones
DAM_INFO_FILE = PROJECT_ROOT / '1029NG.csv'  # Dam basic info

# Climate data directory and raster file / 气候数据目录和栅格文件
CLIMATE_DATA_DIR = DATA_DIR / 'climate'
CLIMATE_RASTER_FILE = CLIMATE_DATA_DIR / 'world_koppen1.tif'

# ============================================================================
# F Parameter Configuration / F参数配置
# ============================================================================

# F-code to full parameter name mapping (based on 1109备份数据说明.csv)
# F6: 正常压力水头分布csv - Normal Head CSV
# F7: 正常位移分布图csv - Normal Displacement CSV  
# F8: 正常FOS值csv - Normal FOS Value CSV
# F9: 正常电阻率分布图csv - Normal Resistivity CSV
# F10: 正常堤坝自然电位csv - Normal SP Potential CSV
# F17: 渗漏压力水头分布csv - Seepage Head CSV
# F18: 渗漏位移分布图csv - Seepage Displacement CSV
# F19: 渗漏FOS值csv - Seepage FOS Value CSV
# F20: 渗漏电阻率分布图csv - Seepage Resistivity CSV
# F21: 渗漏堤坝自然电位csv - Seepage SP Potential CSV
# F22: 渗漏堤坝渗漏电场电位csv - Seepage Electric Field Potential CSV
F_PARAMETER_CONFIG = {
    # Three-column data (x, y, value)
    'F6':  {'format': 'three_col', 'name': 'Normal Head', 'name_en': 'Normal Head', 'unit': 'm', 'category': 'Head'},
    'F7':  {'format': 'three_col', 'name': 'Normal Displacement', 'name_en': 'Normal Displacement', 'unit': 'm', 'category': 'Displacement'},
    'F9':  {'format': 'three_col', 'name': 'Normal Resistivity', 'name_en': 'Normal Resistivity', 'unit': 'Ω·m', 'category': 'Resistivity'},
    'F10': {'format': 'three_col', 'name': 'Normal Self-Potential', 'name_en': 'Normal Self-Potential', 'unit': 'mV', 'category': 'Self-Potential'},
    'F17': {'format': 'three_col', 'name': 'Seepage Head', 'name_en': 'Seepage Head', 'unit': 'm', 'category': 'Head'},
    'F18': {'format': 'three_col', 'name': 'Seepage Displacement', 'name_en': 'Seepage Displacement', 'unit': 'm', 'category': 'Displacement'},
    'F20': {'format': 'three_col', 'name': 'Seepage Resistivity', 'name_en': 'Seepage Resistivity', 'unit': 'Ω·m', 'category': 'Resistivity'},
    'F21': {'format': 'three_col', 'name': 'Seepage Self-Potential', 'name_en': 'Seepage Self-Potential', 'unit': 'mV', 'category': 'Self-Potential'},
    'F22': {'format': 'three_col', 'name': 'Coupling Electric Field', 'name_en': 'Coupling Electric Field', 'unit': 'V', 'category': 'Electric Field'},
    # Two-column FOS data (FOS, displacement)
    'F8':  {'format': 'two_col', 'name': 'Normal FOS', 'name_en': 'Normal FOS', 'unit': '-', 'category': 'FOS'},
    'F19': {'format': 'two_col', 'name': 'Seepage FOS', 'name_en': 'Seepage FOS', 'unit': '-', 'category': 'FOS'},
}

# Full parameter names for output columns (updated per 1109备份数据说明.csv)
PARAMETER_NAMES = {
    'F6_max': 'Normal Head Maximum',
    'F6_min': 'Normal Head Minimum',
    'F7_max': 'Normal Displacement Maximum',
    'F7_min': 'Normal Displacement Minimum',
    'F9_max': 'Normal Resistivity Maximum',
    'F9_min': 'Normal Resistivity Minimum',
    'F10_max': 'Normal Self-Potential Maximum',
    'F10_min': 'Normal Self-Potential Minimum',
    'F17_max': 'Seepage Head Maximum',
    'F17_min': 'Seepage Head Minimum',
    'F18_max': 'Seepage Displacement Maximum',
    'F18_min': 'Seepage Displacement Minimum',
    'F20_max': 'Seepage Resistivity Maximum',
    'F20_min': 'Seepage Resistivity Minimum',
    'F21_max': 'Seepage Self-Potential Maximum',
    'F21_min': 'Seepage Self-Potential Minimum',
    'F22_max': 'Coupling Electric Field Maximum',
    'F22_min': 'Coupling Electric Field Minimum',
    'F8_max': 'Normal FOS Maximum',
    'F19_max': 'Seepage FOS Maximum',
}

def get_full_parameter_name(f_code, stat):
    """Get full parameter name from F-code and statistic type"""
    key = f"{f_code}_{stat}"
    return PARAMETER_NAMES.get(key, key)

# ============================================================================
# Model Configuration / 模型配置
# ============================================================================

MODEL_CONFIGS = {
    'Model1': {
        'description': 'Normal SP → Normal FOS (Baseline)',
        'features': ['Normal Self-Potential Minimum', 'Normal Self-Potential Maximum'],
        'target': 'Normal FOS Maximum',
        'prediction_type': 'direct'  # 直接预测FOS
    },
    'Model2': {
        'description': 'Seepage SP → FOS Degradation Rate',
        'features': ['Seepage Self-Potential Minimum', 'Seepage Self-Potential Maximum'],
        'target': 'FOS_Degradation_Rate',  # 预测安全系数下降率
        'prediction_type': 'degradation'
    },
    'Model3': {
        'description': 'Coupling Field → FOS Degradation Rate',
        'features': ['Coupling Electric Field Minimum', 'Coupling Electric Field Maximum'],
        'target': 'FOS_Degradation_Rate',
        'prediction_type': 'degradation'
    },
    'Model4': {
        'description': 'SP + Coupling → FOS Degradation Rate',
        'features': [
            'Seepage Self-Potential Minimum', 
            'Seepage Self-Potential Maximum',
            'Coupling Electric Field Minimum', 
            'Coupling Electric Field Maximum'
        ],
        'target': 'FOS_Degradation_Rate',
        'prediction_type': 'degradation'
    },
    'Model5': {
        'description': 'SP + Coupling → FOV (Factor of Vulnerability)',
        'features': [
            'Normal FOS Maximum',  # 包含正常FOS作为基准
            'Seepage Self-Potential Minimum', 
            'Seepage Self-Potential Maximum',
            'Coupling Electric Field Minimum', 
            'Coupling Electric Field Maximum'
        ],
        'target': 'Actual_Safety_Factor',  # 真实安全系数（Ground Truth用于训练）
        'prediction_type': 'actual_safety'  # 预测结果保存为FOV
    }
}

# Machine learning algorithms / 机器学习算法
ML_ALGORITHMS = [
    'Linear Regression',
    'Ridge Regression',
    'Lasso Regression',
    'Random Forest',
    'Gradient Boosting',
    'SVR'
]

# Safety Degradation Index (SDI) Calculation
# SDI反映渗漏导致的安全性下降，数值越高表示安全性下降越严重
def calculate_sdi(normal_fos, seepage_fos):
    """
    Calculate Safety Degradation Index (SDI)
    
    概念：渗漏时FOS上升是因为排水通道释放了应力，但这实际代表堤坝安全性下降
    SDI = (Seepage FOS - Normal FOS) / Normal FOS
    
    解释：
    - SDI > 0: 渗漏时FOS上升，表示安全性下降（需要排水才能维持）
    - SDI < 0: 渗漏时FOS下降，表示渗漏直接导致失稳
    - SDI ≈ 0: 渗漏影响较小
    
    实际应用中，我们预测的目标是：
    - 如果没有排水通道，渗漏会导致FOS下降多少
    - Predicted Safety Factor = Normal FOS - |SDI| * Normal FOS
    """
    if pd.isna(normal_fos) or pd.isna(seepage_fos) or normal_fos == 0:
        return np.nan
    return (seepage_fos - normal_fos) / normal_fos

# ============================================================================
# Threshold Configuration / 阈值配置
# ============================================================================

# FOS thresholds for stability classification
FOS_THRESHOLDS = {
    'stable': 1.5,      # FOS > 1.5: Stable
    'marginal': 1.2,    # 1.2 <= FOS <= 1.5: Marginally Stable
    'at_risk': 1.2      # FOS < 1.2: At Risk
}

# FOV (Factor of Vulnerability) thresholds
# FOV represents predicted safety factor considering seepage degradation
FOV_THRESHOLDS = {
    'stable': 1.5,      # FOV ≥ 1.5: Stable
    'marginal': 1.2,    # 1.2 ≤ FOV < 1.5: Marginally Stable  
    'at_risk': 1.2      # FOV < 1.2: At Risk
}

def classify_stability(fov_value):
    """Classify stability based on FOV (Factor of Vulnerability) value"""
    if fov_value is None or (isinstance(fov_value, float) and fov_value != fov_value):  # NaN check
        return 'Unknown'
    if fov_value >= FOV_THRESHOLDS['stable']:
        return 'Stable'
    elif fov_value >= FOV_THRESHOLDS['marginal']:
        return 'Marginally Stable'
    else:
        return 'At Risk'

# ============================================================================
# Köppen Climate Classification / 柯本气候分类
# ============================================================================

KOPPEN_CLIMATE_ZONES = {
    'A': {'name': 'Tropical', 'name_cn': '热带', 'color': '#4A90E2'},      # Blue
    'B': {'name': 'Arid', 'name_cn': '干旱带', 'color': '#F5B041'},          # Sand/Orange
    'C': {'name': 'Temperate', 'name_cn': '温带', 'color': '#58D68D'},     # Green
    'D': {'name': 'Continental', 'name_cn': '大陆性', 'color': '#EC7063'},   # Pink/Red
    'E': {'name': 'Polar', 'name_cn': '极地', 'color': '#CACFD2'}           # Grey
}

# Climate zone mapping from GRIDCODE
CLIMATE_GRIDCODE_MAPPING = {
    1:  ('Af',  'A', 'Af', '热带雨林气候'),
    2:  ('Am',  'A', 'Am', '热带季风气候'),
    3:  ('Aw',  'A', 'Aw', '热带疏林气候'),
    4:  ('Bwh', 'B', 'Bw', '沙漠气候'),
    5:  ('Bwk', 'B', 'Bw', '沙漠气候'),
    6:  ('Bsh', 'B', 'Bs', '草原气候'),
    7:  ('Bsk', 'B', 'Bs', '草原气候'),
    8:  ('Csa', 'C', 'Cs', '夏干温暖气候'),
    9:  ('Csb', 'C', 'Cs', '夏干温暖气候'),
    11: ('Cwa', 'C', 'Cw', '冬干温暖气候'),
    12: ('Cwb', 'C', 'Cw', '冬干温暖气候'),
    13: ('Cwc', 'C', 'Cw', '冬干温暖气候'),
    14: ('Cfa', 'C', 'Cf', '常湿温暖气候'),
    15: ('Cfb', 'C', 'Cf', '常湿温暖气候'),
    16: ('Cfc', 'C', 'Cf', '常湿温暖气候'),
    17: ('Dsa', 'D', 'Ds', '亚寒带大陆性气候'),
    18: ('Dsb', 'D', 'Ds', '亚寒带大陆性气候'),
    19: ('Dsc', 'D', 'Ds', '亚寒带大陆性气候'),
    20: ('Dsd', 'D', 'Ds', '亚寒带大陆性气候'),
    21: ('Dwa', 'D', 'Dw', '亚寒带季风性气候'),
    22: ('Dwb', 'D', 'Dw', '亚寒带季风性气候'),
    23: ('Dwc', 'D', 'Dw', '亚寒带季风性气候'),
    24: ('Dwd', 'D', 'Dw', '亚寒带季风性气候'),
    25: ('Dfa', 'D', 'Df', '常湿冷温气候'),
    26: ('Dfb', 'D', 'Df', '常湿冷温气候'),
    27: ('Dfc', 'D', 'Df', '常湿冷温气候'),
    28: ('Dfd', 'D', 'Df', '常湿冷温气候'),
    29: ('ET',  'E', 'ET', '冰原气候'),
    30: ('EF',  'E', 'EF', '苔原气候'),
    31: ('ETH', 'E', 'ET', '冰原气候'),
    32: ('EFH', 'E', 'EF', '苔原气候'),
}

# ============================================================================
# Color Configuration / 颜色配置
# ============================================================================

# FOS-based stability colors (Red-Yellow-Green-Blue system)
# 红色(风险) -> 橙黄色(边缘) -> 绿色(稳定) -> 蓝色(非常稳定)
FOS_COLORS = {
    'at_risk': '#e74c3c',       # Red - FOS < 1.2 (At Risk)
    'marginal': '#f39c12',      # Yellow/Orange - 1.2 ≤ FOS < 1.5 (Marginally Stable)
    'stable': '#27ae60',        # Green - 1.5 ≤ FOS < 2.0 (Stable)
    'very_stable': '#3498db',   # Blue - FOS ≥ 2.0 (Very Stable)
}

COLORS = {
    # Primary colors (use FOS-based colors for consistency)
    'primary': '#3498db',       # Blue (Very Stable)
    'secondary': '#27ae60',     # Green (Stable)
    'tertiary': '#f39c12',      # Yellow (Marginal)
    'quaternary': '#e74c3c',    # Red (At Risk)
    
    # Stability colors (aligned with FOS thresholds)
    'stable': FOS_COLORS['stable'],          # Green - FOS ≥ 1.5
    'marginal': FOS_COLORS['marginal'],      # Yellow - 1.2 ≤ FOS < 1.5
    'at_risk': FOS_COLORS['at_risk'],        # Red - FOS < 1.2
    'very_stable': FOS_COLORS['very_stable'], # Blue - FOS ≥ 2.0
    'neutral': '#95a5a6',                    # Gray
    
    # Additional colors for models
    'model1': '#3498db',  # Blue
    'model2': '#27ae60',  # Green
    'model3': '#f39c12',  # Yellow/Orange
    'model4': '#e74c3c',  # Red
    'model5': '#9b59b6',  # Purple
}

# Stability classification color palette
STABILITY_PALETTE = [
    FOS_COLORS['at_risk'],      # At Risk - Red
    FOS_COLORS['marginal'],     # Marginally Stable - Yellow
    FOS_COLORS['stable'],       # Stable - Green
]

# ============================================================================
# Plot Configuration / 绑图配置
# ============================================================================

PLOT_CONFIG = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10.5,  # Unified base font size (Wu Hao approx 10.5pt)
    'axes.titlesize': 12, # Slightly larger than base
    'axes.labelsize': 10.5,  # Unified label size
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlepad': 10,  # Padding above title to prevent overlap
    'axes.labelpad': 8,   # Padding for axis labels
    'xtick.major.pad': 5, # Padding for x-axis tick labels
    'ytick.major.pad': 5, # Padding for y-axis tick labels
}

# Figure size presets (in inches)
FIGURE_SIZES = {
    'single': (8, 6),
    'double_wide': (14, 6),
    'double_tall': (8, 12),
    'quad': (14, 12),
    'correlation_2x5': (26, 10),
    'correlation_3x2': (16, 10),
    'fov_combined': (16, 14),
    'model_scatter': (14, 12),
    'model_scatter_5': (14, 18),  # For 5 models (3 rows × 2 cols)
    'climate_pie_stacked': (8, 20),  # For stacked pie charts
}

# ============================================================================
# Utility Functions / 工具函数
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        RESULTS_DIR,
        CORRELATION_DIR,
        MODELS_DIR,
        CLIMATE_DIR,
        CLIMATE_DATA_DIR
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


def print_section_header(title, char='=', width=70):
    """Print a formatted section header"""
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_subsection(title, char='-', width=50):
    """Print a formatted subsection header"""
    print()
    print(f"{char * 3} {title} {char * 3}")


def setup_plot_style():
    """Apply the plot configuration globally"""
    plt.rcParams.update(PLOT_CONFIG)


def get_stability_color(stability):
    """Get color for stability category"""
    color_map = {
        'Stable': COLORS['stable'],
        'Marginally Stable': COLORS['marginal'],
        'At Risk': COLORS['at_risk'],
        'Unknown': COLORS['neutral']
    }
    return color_map.get(stability, COLORS['neutral'])


def get_climate_color(zone):
    """Get color for climate zone"""
    return KOPPEN_CLIMATE_ZONES.get(zone, {}).get('color', COLORS['neutral'])


# ============================================================================
# Module Initialization / 模块初始化
# ============================================================================

# Ensure directories exist when config is imported
ensure_directories()

# Apply plot style
setup_plot_style()
