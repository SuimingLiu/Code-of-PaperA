"""
Dam Safety Analysis Pipeline

Data Flow:
    Step 1: Raw data -> P1.csv (processed dam parameters)
    Step 2: P1.csv -> Correlation analysis figures
    Step 3: P1.csv -> P2.csv (with model predictions)
    Step 4: P3.csv (manual input) -> Climate analysis figures

Usage:
    python main.py                  # Run all steps
    python main.py --steps 1        # Run only Step 1
    python main.py --steps 1 2 3    # Run multiple steps
    python main.py --no-clean       # Don't clean old figures
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROJECT_ROOT, RESULTS_DIR, CORRELATION_DIR, MODELS_DIR, CLIMATE_DIR,
    P1_FILE, P2_FILE, P3_FILE,
    ensure_directories, print_section_header
)


def clean_results():
    """Clean all figures and results by deleting the entire results folder"""
    print("\n" + "-" * 70)
    print("Cleaning previous results...")
    print("-" * 70)
    
    # Delete entire results folder if it exists
    if RESULTS_DIR.exists():
        try:
            shutil.rmtree(RESULTS_DIR)
            print(f"  ✓ Deleted entire results folder: {RESULTS_DIR}")
        except Exception as e:
            print(f"  ✗ Failed to delete results folder: {e}")
    
    # Recreate directories
    ensure_directories()
    print("  ✓ Recreated results directories")
    
    return 0


def run_step1():
    """Run Step 1: Data Processing -> Output P1.csv"""
    from step1_data_processing import DamDataProcessor
    processor = DamDataProcessor(num_threads=10)
    return processor.run()


def run_step2(skip_combined=False):
    """Run Step 2: FOS Analysis -> Reads P1.csv"""
    from step2_correlation_analysis import FOSAnalyzer
    analyzer = FOSAnalyzer()
    return analyzer.run(skip_combined=skip_combined)


def run_step3():
    """Run Step 3: Model Comparison -> Reads P1.csv, Output P2.csv"""
    from step3_model_comparison import ModelComparison
    comparison = ModelComparison(num_threads=4)
    return comparison.run()


def run_step4():
    """Run Step 4: Climate Analysis -> Reads P3.csv (manual input)"""
    from step4_climate_analysis import ClimateAnalyzer
    analyzer = ClimateAnalyzer()
    return analyzer.run()


def main(steps=None, clean=True):
    """
    Main function
    
    Args:
        steps: List of steps to run, e.g., [1, 2, 3, 4]. None means all
        clean: Whether to clean old results before running
    """
    print_section_header("Dam Safety Analysis Pipeline")
    print(f"Project Root: {PROJECT_ROOT}")
    print()
    
    start_time = time.time()
    
    # Ensure directories exist
    ensure_directories()
    
    # Clean old results
    if clean:
        clean_results()
    
    # Run all steps by default
    if steps is None:
        steps = [1, 2, 3, 4]
    
    results = {}
    
    # Step 1: Data Processing
    if 1 in steps:
        print_section_header("RUNNING STEP 1: Data Processing")
        try:
            results['step1'] = run_step1()
        except Exception as e:
            print(f"✗ Step 1 failed: {e}")
            results['step1'] = None
    
    # Step 2: FOS Analysis and Visualization (skip combined figure initially)
    if 2 in steps:
        print_section_header("RUNNING STEP 2: FOS Analysis")
        try:
            # Skip combined figure if step 3 will also run
            skip_combined = (3 in steps)
            results['step2'] = run_step2(skip_combined=skip_combined)
            if skip_combined:
                print("  Note: fov_analysis_combined.png will be generated after Step 3")
        except Exception as e:
            print(f"✗ Step 2 failed: {e}")
            import traceback
            traceback.print_exc()
            results['step2'] = None
    
    # Step 3: Model Comparison
    if 3 in steps:
        print_section_header("RUNNING STEP 3: Model Comparison")
        try:
            results['step3'] = run_step3()
        except Exception as e:
            print(f"✗ Step 3 failed: {e}")
            import traceback
            traceback.print_exc()
            results['step3'] = None
    
    # Regenerate fov_analysis_combined.png (after step3, if both step2 and step3 ran)
    if 2 in steps and 3 in steps:
        print_section_header("REGENERATING: fov_analysis_combined.png")
        print("Now generating combined figure with model scatter plot...")
        try:
            from step2_correlation_analysis import FOSAnalyzer
            analyzer = FOSAnalyzer()
            analyzer.load_data()
            analyzer.assign_climate_zones()
            analyzer.create_combined_figure()
            print("  ✓ fov_analysis_combined.png generated successfully")
        except Exception as e:
            print(f"  ⚠ Failed to generate combined figure: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Climate Impact Analysis
    if 4 in steps:
        print_section_header("RUNNING STEP 4: Climate Analysis")
        
        # Always regenerate P3.csv from P2.csv to ensure consistency
        if P2_FILE.exists():
            print("Regenerating P3.csv with climate zones from P2.csv...")
            print("  (This ensures filtered data and FOV predictions are included)")
            try:
                from add_climate_zone import add_climate_zones
                from config import CLIMATE_RASTER_FILE
                
                if CLIMATE_RASTER_FILE.exists():
                    # Force regenerate P3.csv
                    add_climate_zones(P2_FILE, P3_FILE, CLIMATE_RASTER_FILE)
                    print(f"  ✓ P3.csv regenerated successfully from P2.csv")
                else:
                    print(f"  ⚠ Climate raster not found: {CLIMATE_RASTER_FILE}")
                    print(f"    Please ensure the raster file exists.")
            except Exception as e:
                print(f"  ⚠ Failed to regenerate P3.csv: {e}")
                import traceback
                traceback.print_exc()
        elif not P3_FILE.exists():
            print(f"  ⚠ P2.csv not found and P3.csv does not exist")
            print(f"    Please run Step 3 first to generate P2.csv")
        
        try:
            results['step4'] = run_step4()
        except Exception as e:
            print(f"✗ Step 4 failed: {e}")
            import traceback
            traceback.print_exc()
            results['step4'] = None
    
    elapsed_time = time.time() - start_time
    
    # Print completion info
    print_section_header("PIPELINE COMPLETED")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Steps completed: {steps}")
    print()
    print("Data Flow:")
    print(f"  - Step 1 Output: {P1_FILE}")
    print(f"  - Step 3 Output: {P2_FILE}")
    print(f"  - Step 4 Input:  {P3_FILE} (auto-generated from P2.csv + climate raster)")
    print()
    print("Figure locations:")
    print(f"  - Step 2 Figures: {CORRELATION_DIR}")
    print(f"  - Step 3 Figures: {MODELS_DIR}")
    print(f"  - Step 4 Figures: {CLIMATE_DIR}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dam Safety Analysis Pipeline')
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                       help='Steps to run (1-4), e.g., --steps 1 2 3')
    parser.add_argument('--no-clean', action='store_true',
                       help='Do not clean previous results before running')
    
    args = parser.parse_args()
    
    main(steps=args.steps, clean=not args.no_clean)
