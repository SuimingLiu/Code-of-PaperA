"""
Step 1: Dam Data Processing Pipeline

Functionality:
1. Read CSV data from F-code subfolders under the raw folder
2. Use multi-threading for parallel processing of large-scale data
3. Extract statistical features (max, min, mean, positive/negative averages, etc.)
4. Match with dam basic info from 1029NG.csv
5. Output consolidated data to P1.csv

Data formats:
- Three-column data (x, y, value): F6, F7, F9, F10, F17, F18, F20, F21, F22
- Two-column data (factor, fos): F8, F19
"""

import os
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np

# Import configuration
from config import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR,
    P1_FILE, DAM_INFO_FILE, F_PARAMETER_CONFIG,
    get_full_parameter_name, ensure_directories, print_section_header, print_subsection
)

warnings.filterwarnings('ignore')


class DamDataProcessor:
    """Dam monitoring data processor with multi-threading support"""
    
    def __init__(self, num_threads=10):
        """
        Initialize the processor
        
        Args:
            num_threads: Number of threads for parallel processing
        """
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.f_config = F_PARAMETER_CONFIG
        
    def extract_dam_name(self, filename, f_code):
        """
        Extract dam name from filename
        
        Args:
            filename: Filename like 'Miyun_F10.csv'
            f_code: F code like 'F10'
        
        Returns:
            Dam name like 'Miyun'
        """
        name = filename.replace('.csv', '').replace('.png', '')
        suffix = f'_{f_code}'
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        return name
    
    def read_csv_safe(self, file_path):
        """
        Safely read CSV file with automatic encoding detection
        
        Args:
            file_path: Path to the CSV file
        
        Returns:
            DataFrame or None
        """
        encodings = ['utf-8', 'gbk', 'gb18030', 'latin1']
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, header=None, encoding=encoding)
            except:
                continue
        return None
    
    def compute_statistics_three_col(self, df, f_code=None):
        """
        Compute statistics for three-column data (x, y, value)
        Only keep max and min of the third column
        Note: 
        - For F22 (Coupling Electric Field): x >= 5 and y >= 0
        - For other parameters: only y >= 0
        
        Args:
            df: DataFrame with columns [x, y, value]
            f_code: F code to determine filtering rules
        
        Returns:
            Statistics dict {max, min}
        """
        if df is None or len(df.columns) < 3:
            return None
        
        # 根据F编码应用不同的过滤规则
        col1 = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        if f_code == 'F22':  # Coupling Electric Field 需要额外过滤 x >= 5
            col0 = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            df_filtered = df[(col0 >= 5) & (col1 >= 0)]
        else:  # 其他参数只过滤 y >= 0
            df_filtered = df[col1 >= 0]
        
        # 获取 value 列（第三列）
        values = pd.to_numeric(df_filtered.iloc[:, 2], errors='coerce').dropna()
        
        if len(values) == 0:
            return None
        
        return {
            'max': values.max(),
            'min': values.min()
        }
    
    def compute_statistics_two_col(self, df):
        """
        Compute statistics for two-column FOS data
        
        Data format: First column is FOS value, second column is displacement
        Only take the maximum of the first column
        
        Args:
            df: DataFrame with columns [FOS, displacement]
        
        Returns:
            Statistics dict {max}
        """
        if df is None or len(df.columns) < 2:
            return None
        
        # FOS值在第一列
        fos_values = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        
        if len(fos_values) == 0:
            return None
        
        return {
            'max': fos_values.max()  # Only keep max FOS value
        }
    
    def process_single_file(self, file_path, f_code, config):
        """
        Process a single CSV file
        
        Args:
            file_path: File path
            f_code: F code
            config: Parameter configuration
        
        Returns:
            (dam_name, stats_dict)
        """
        dam_name = self.extract_dam_name(file_path.name, f_code)
        df = self.read_csv_safe(file_path)
        
        if df is None:
            return dam_name, None
        
        if config['format'] == 'three_col':
            stats = self.compute_statistics_three_col(df, f_code)
        else:  # two_col (FOS data)
            stats = self.compute_statistics_two_col(df)
        
        return dam_name, stats
    
    def get_baseline_dams(self):
        """
        Get dam names from F1 folder as baseline
        
        Returns:
            Set of dam names
        """
        f1_path = RAW_DATA_DIR / 'F1'
        if not f1_path.exists():
            print("Warning: F1 folder not found, cannot establish baseline")
            return None
        
        baseline_dams = set()
        for fname in f1_path.iterdir():
            if fname.suffix in ['.png', '.csv']:
                dam_name = self.extract_dam_name(fname.name, 'F1')
                baseline_dams.add(dam_name)
        
        print(f"Baseline from F1: {len(baseline_dams)} dams")
        return baseline_dams
    
    def process_f_folder(self, f_code):
        """
        Process all CSV files in a single F folder
        
        Args:
            f_code: F code like 'F10'
        
        Returns:
            (f_code, {dam_name: stats})
        """
        config = self.f_config.get(f_code)
        if not config:
            return f_code, {}
        
        folder_path = RAW_DATA_DIR / f_code
        if not folder_path.exists():
            return f_code, {}
        
        csv_files = list(folder_path.glob('*.csv'))
        if not csv_files:
            return f_code, {}
        
        results = {}
        
        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self.process_single_file, f, f_code, config): f 
                for f in csv_files
            }
            
            for future in as_completed(futures):
                dam_name, stats = future.result()
                if stats:
                    results[dam_name] = stats
        
        return f_code, results
    
    def load_dam_info(self):
        """
        Load dam basic information file
        
        Returns:
            DataFrame or None
        """
        if not DAM_INFO_FILE.exists():
            print(f"Warning: Dam info file not found: {DAM_INFO_FILE}")
            return None
        
        try:
            df = pd.read_csv(DAM_INFO_FILE, encoding='utf-8-sig')
            print(f"Loaded dam info: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error: Failed to load dam info: {e}")
            return None
    
    def run(self, output_file=None):
        """
        Execute the complete data processing pipeline
        
        Args:
            output_file: Output file path, defaults to P1.csv
        
        Returns:
            Processed DataFrame
        """
        if output_file is None:
            output_file = P1_FILE
        else:
            output_file = Path(output_file)
        
        print_section_header("Step 1: Dam Data Processing Pipeline")
        print(f"Data directory: {RAW_DATA_DIR}")
        print(f"Thread count: {self.num_threads}")
        print()
        
        start_time = time.time()
        
        # 确保目录存在
        ensure_directories()
        
        # 获取F1基准大坝列表
        baseline_dams = self.get_baseline_dams()
        
        # 加载大坝基础信息
        df_dam_info = self.load_dam_info()
        
        # 存储所有F编号数据
        all_data = {}  # {f_code: {dam_name: stats}}
        
        # 处理所有F文件夹
        print_subsection("Processing F folders")
        total_folders = len(self.f_config)
        
        # 并行处理F文件夹
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_f_folder, f_code): f_code 
                for f_code in self.f_config.keys()
            }
            
            completed = 0
            for future in as_completed(futures):
                f_code, results = future.result()
                completed += 1
                config = self.f_config[f_code]
                
                if results:
                    all_data[f_code] = results
                    print(f"  [{completed}/{total_folders}] {f_code} ({config['name_en']}): {len(results)} dams")
                else:
                    print(f"  [{completed}/{total_folders}] {f_code}: No data or folder not found")
        
        # 合并所有大坝数据
        print_subsection("Consolidating data")
        
        # 获取所有大坝名称，以F1为基准
        if baseline_dams:
            all_dams = baseline_dams
            print(f"Using F1 baseline: {len(all_dams)} dams")
        else:
            all_dams = set()
            for data_dict in all_data.values():
                all_dams.update(data_dict.keys())
            print(f"Found {len(all_dams)} dams from F folders (no baseline)")
        
        # 构建参数DataFrame
        records = []
        for dam in sorted(all_dams):
            record = {'Dam Name': dam}
            
            for f_code, data_dict in all_data.items():
                if dam in data_dict:
                    stats = data_dict[dam]
                    config = self.f_config[f_code]
                    param_name = config['name_en']
                    
                    # 根据数据类型添加不同的统计量
                    if config['format'] == 'two_col':
                        # FOS数据只保留max
                        record[f'{param_name} Maximum'] = stats.get('max')
                    else:
                        # 三列数据只保留max和min
                        record[f'{param_name} Maximum'] = stats.get('max')
                        record[f'{param_name} Minimum'] = stats.get('min')
            
            records.append(record)
        
        df_params = pd.DataFrame(records)
        
        # 统计 F19 max > F8 max 的占比
        self._compute_fos_comparison(df_params)
        
        # 与大坝基础信息合并
        if df_dam_info is not None:
            print_subsection("Merging with dam info")
            
            # 匹配 DAM_NAME 列
            if 'DAM_NAME' in df_dam_info.columns:
                # 先去除大坝信息中的重复项（保留第一个）
                df_dam_info_unique = df_dam_info.drop_duplicates(subset=['DAM_NAME'], keep='first')
                print(f"  Dam info: {len(df_dam_info)} -> {len(df_dam_info_unique)} (after dedup)")
                
                df_merged = df_dam_info_unique.merge(
                    df_params, 
                    left_on='DAM_NAME', 
                    right_on='Dam Name', 
                    how='inner'
                )
                # 删除重复的 Dam Name 列
                if 'Dam Name' in df_merged.columns and 'DAM_NAME' in df_merged.columns:
                    df_merged = df_merged.drop(columns=['Dam Name'])
                    df_merged = df_merged.rename(columns={'DAM_NAME': 'Dam Name'})
                
                print(f"  Matched: {len(df_merged)} dams")
            else:
                df_merged = df_params
                print("  Warning: DAM_NAME column not found in dam info file")
        else:
            df_merged = df_params
        
        # 重新排列列顺序
        # 将 Dam Name 放在第一列，然后是基础信息列，最后是参数列
        cols = list(df_merged.columns)
        if 'Dam Name' in cols:
            cols.remove('Dam Name')
            cols = ['Dam Name'] + cols
            df_merged = df_merged[cols]
        
        # 过滤异常FOS数据：删除Normal FOS=1或3，Seepage FOS=1或3的数据
        print_subsection("Filtering Anomalous FOS Data")
        n_before = len(df_merged)
        
        normal_fos_col = 'Normal FOS Maximum'
        seepage_fos_col = 'Seepage FOS Maximum'
        
        # 记录过滤前的状态
        if normal_fos_col in df_merged.columns:
            fos1_normal = (df_merged[normal_fos_col] == 1).sum()
            fos3_normal = (df_merged[normal_fos_col] == 3).sum()
            print(f"  Before filtering - Normal FOS=1: {fos1_normal}, Normal FOS=3: {fos3_normal}")
        
        if seepage_fos_col in df_merged.columns:
            fos1_seepage = (df_merged[seepage_fos_col] == 1).sum()
            fos3_seepage = (df_merged[seepage_fos_col] == 3).sum()
            print(f"  Before filtering - Seepage FOS=1: {fos1_seepage}, Seepage FOS=3: {fos3_seepage}")
        
        # 过滤条件：保留Normal FOS ≠ 1 且 ≠ 3，以及Seepage FOS ≠ 1 且 ≠ 3的数据
        mask = pd.Series([True] * len(df_merged), index=df_merged.index)
        
        if normal_fos_col in df_merged.columns:
            mask = mask & (df_merged[normal_fos_col] != 1) & (df_merged[normal_fos_col] != 3)
        
        if seepage_fos_col in df_merged.columns:
            mask = mask & (df_merged[seepage_fos_col] != 1) & (df_merged[seepage_fos_col] != 3)
        
        df_merged = df_merged[mask].copy()
        n_after = len(df_merged)
        n_filtered = n_before - n_after
        
        print(f"  Filtered out {n_filtered} dams with FOS=1 or FOS=3 ({n_filtered/n_before*100:.1f}%)")
        print(f"  Remaining: {n_after} dams")
        
        # 保存结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        elapsed_time = time.time() - start_time
        
        print_subsection("Summary")
        print(f"✓ Output saved: {output_file}")
        print(f"  Total dams (after filtering): {len(df_merged)}")
        print(f"  Total columns: {len(df_merged.columns)}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        
        # 打印数据概览
        self._print_data_summary(df_merged)
        
        # 保存参数映射文件
        self._save_parameter_mapping(df_merged)
        
        return df_merged
    
    def _compute_fos_comparison(self, df):
        """Compute F19 max > F8 max percentage"""
        print_subsection("FOS Comparison: F19 vs F8")
        
        normal_fos_col = 'Normal FOS Maximum'
        seepage_fos_col = 'Seepage FOS Maximum'
        
        if normal_fos_col in df.columns and seepage_fos_col in df.columns:
            df_valid = df[[normal_fos_col, seepage_fos_col]].dropna()
            total = len(df_valid)
            
            if total > 0:
                f19_greater = (df_valid[seepage_fos_col] > df_valid[normal_fos_col]).sum()
                f8_greater = (df_valid[seepage_fos_col] < df_valid[normal_fos_col]).sum()
                equal = (df_valid[seepage_fos_col] == df_valid[normal_fos_col]).sum()
                
                print(f"  Total dams with both FOS: {total}")
                print(f"  F19 max > F8 max (Seepage > Normal): {f19_greater} ({f19_greater/total*100:.1f}%)")
                print(f"  F19 max < F8 max (Seepage < Normal): {f8_greater} ({f8_greater/total*100:.1f}%)")
                print(f"  F19 max = F8 max: {equal} ({equal/total*100:.1f}%)")
                
                # 保存统计结果
                self.fos_comparison = {
                    'total': total,
                    'f19_greater': f19_greater,
                    'f19_greater_pct': f19_greater/total*100,
                    'f8_greater': f8_greater,
                    'f8_greater_pct': f8_greater/total*100,
                    'equal': equal
                }
        else:
            print("  Warning: FOS columns not found")
    
    def _print_data_summary(self, df):
        """Print data overview"""
        print_subsection("Data Overview")
        
        # 统计各类参数
        normal_cols = [c for c in df.columns if 'Normal' in c]
        seepage_cols = [c for c in df.columns if 'Seepage' in c or 'Coupling' in c]
        
        print(f"Normal state parameters: {len(normal_cols)}")
        print(f"Seepage state parameters: {len(seepage_cols)}")
        
        # 打印关键参数的数据完整性
        key_params = [
            'Normal FOS Maximum',
            'Seepage FOS Maximum',
            'Normal Self-Potential Maximum',
            'Seepage Self-Potential Maximum',
            'Coupling Electric Field Maximum'
        ]
        
        print("\nKey parameters data completeness:")
        for param in key_params:
            if param in df.columns:
                count = df[param].notna().sum()
                percent = count / len(df) * 100
                print(f"  {param}: {count} ({percent:.1f}%)")
    
    def _save_parameter_mapping(self, df):
        """Save parameter mapping file"""
        records = []
        
        for col in df.columns:
            if col == 'Dam Name':
                continue
            
            # 查找对应的F编号
            f_code = None
            for code, config in self.f_config.items():
                if config['name_en'] in col:
                    f_code = code
                    break
            
            if f_code:
                config = self.f_config[f_code]
                records.append({
                    'Column Name': col,
                    'F Code': f_code,
                    'Parameter': config['name_en'],
                    'Unit': config['unit'],
                    'Category': config['category'],
                    'Non-null Count': df[col].notna().sum()
                })
        
        if records:
            df_map = pd.DataFrame(records)
            output_path = RESULTS_DIR / 'parameter_mapping.csv'
            df_map.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Parameter mapping saved: {output_path}")


def main():
    """Main function"""
    # Change to script directory
    os.chdir(PROJECT_ROOT)
    
    # 创建处理器并运行
    processor = DamDataProcessor(num_threads=10)
    df_result = processor.run()
    
    print("\n" + "=" * 70)
    print("Step 1 completed!")
    print("Next: Run step2_correlation_analysis.py")
    print("=" * 70)
    
    return df_result


if __name__ == "__main__":
    main()
