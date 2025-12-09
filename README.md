# Dam Safety Analysis Pipeline

A comprehensive analysis framework for dam safety assessment using electro-geomechanical monitoring signals.

## Overview

This pipeline processes dam monitoring data through four analytical steps: data processing, correlation analysis, model comparison, and climate impact assessment. The framework demonstrates the effectiveness of combining self-potential and coupling electric field measurements for predicting dam stability factors.

## Project Structure

```
图件pro/
├── config.py                      # Centralized configuration
├── main.py                        # Main pipeline controller
├── step1_data_processing.py       # Data extraction and consolidation
├── step2_correlation_analysis.py  # FOS correlation and visualization
├── step3_model_comparison.py      # Multi-model performance comparison
├── step4_climate_analysis.py      # Climate zone impact analysis
├── data/                          # Data directory
│   ├── raw/                       # Raw monitoring data (F1-F22 folders)
│   ├── P1.csv                     # Processed parameters (Step 1 output)
│   ├── P2.csv                     # With FOV predictions (Step 3 output)
│   └── P3.csv                     # With climate zones (Step 4 input)
├── results/                       # Analysis outputs
│   ├── correlations/              # Step 2 figures
│   ├── models/                    # Step 3 figures
│   └── climate/                   # Step 4 figures
└── README_CN.md                   # Chinese documentation
```

## Quick Start

### Installation

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn rasterio
```

### Run Full Pipeline

```bash
python main.py
```

### Run Specific Steps

```bash
python main.py --steps 1 2 3    # Run steps 1-3 only
python main.py --no-clean        # Skip cleaning previous results
```

## Pipeline Steps

### Step 1: Data Processing

**Input:** CSV files in `data/raw/F*` folders

**Output:** `data/P1.csv` - Consolidated dam parameters

**Features:**
- Multi-threaded parallel processing
- Extracts max/min from spatial data
- Merges with dam metadata
- Quality control and validation

### Step 2: Correlation Analysis

**Input:** `data/P1.csv`

**Output Figures:**
- `fig2a_key_correlations.png` - Resistivity correlations (2×2)
- `fig2b_other_correlations.png` - SP and coupling field (3×2)  
- `fov_analysis_combined.png` - FOS analysis (2×2):
  - (a) Normal vs Seepage FOS
  - (b) FOS change distribution
  - (c) FOS violin by climate
  - (d) FOS ridge plot by climate

### Step 3: Model Comparison

**Four Models:**

| Model | Features | Target |
|-------|----------|--------|
| Model 1 | Normal SP | Normal FOS |
| Model 2 | Seepage SP | Seepage FOS |
| Model 3 | Coupling Field | Seepage FOS |
| Model 4 | SP + Coupling | Seepage FOS |

**Algorithms:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR

**Output:**
- `model_scatter_comparison.png` - 2×2 model comparison
- `model_combined_scatter.png` - All models overlay
- `data/P2.csv` - Original data + FOV predictions

### Step 4: Climate Impact Analysis

**Input:** `data/P3.csv` (P2 + climate zones)

**Climate Zones:** A (Tropical), B (Arid), C (Temperate), D (Continental), E (Polar)

**Output:**
- `climate_fos_boxplot.png`
- `climate_fos_violin.png`
- `climate_stability_stacked.png`
- Statistical test results (JSON)

## Configuration

Edit `config.py` for:
- Paths and directories
- Parameter mappings
- Model configurations
- Stability thresholds
- Climate zone settings
- Plot styles and colors

## Key Features

✅ Modular design - Independent steps  
✅ Multi-threading support  
✅ Unified test sets for fair model comparison  
✅ Auto-generates P3 from P2 + climate raster  
✅ Publication-ready figures  
✅ Reproducible with fixed random seeds

## Troubleshooting

**Missing data:** Ensure files in `data/raw/F*/` folders  
**Encoding errors:** Use UTF-8 encoding  
**Python version:** Requires Python 3.8+

## License

MIT License

---

**Last Updated:** 2025-12-02
