"""
Convert CSV data (x, y, value) to VTU format for better visualization
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pyevtk.hl as vtk
from scipy.spatial import Delaunay

def read_csv_data(dam_name, f_code):
    """Read CSV data for a specific dam and F-code"""
    csv_path = Path(f'e:/PaperV2/图件pro/data/raw/{f_code}/{dam_name}_{f_code}.csv')
    if not csv_path.exists():
        return None
    
    try:
        data = pd.read_csv(csv_path, names=['x', 'y', 'value'])
        return data
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def csv_to_vtu(dam_name, f_code, output_dir):
    """Convert CSV data to VTU format"""
    data = read_csv_data(dam_name, f_code)
    if data is None or len(data) == 0:
        print(f"No data for {dam_name} - {f_code}")
        return False
    
    # Extract coordinates and values
    x = data['x'].values.astype(np.float64)
    y = data['y'].values.astype(np.float64)
    z = np.zeros_like(x, dtype=np.float64)  # 2D data, z=0
    values = data['value'].values.astype(np.float64)
    
    # Create output directory
    output_path = Path(output_dir) / f_code
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output file path (without extension, pyevtk adds .vtu)
    output_file = str(output_path / dam_name)
    
    # Create unstructured grid using Delaunay triangulation
    try:
        # Perform Delaunay triangulation in 2D
        points_2d = np.column_stack([x, y])
        tri = Delaunay(points_2d)
        
        # Get triangles (connectivity)
        triangles = tri.simplices
        
        # Prepare connectivity array for VTK
        # Each triangle has 3 vertices
        connectivity = triangles.flatten()
        offsets = np.arange(3, 3 * (len(triangles) + 1), 3)
        cell_types = np.full(len(triangles), 5, dtype=np.uint8)  # VTK_TRIANGLE = 5
        
        # Write VTU file using pyevtk
        vtk.unstructuredGridToVTK(
            output_file,
            x, y, z,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            pointData={f_code: values}
        )
        
        print(f"Converted: {dam_name} - {f_code} -> {output_file}.vtu")
        return True
        
    except Exception as e:
        print(f"Error converting {dam_name} - {f_code}: {e}")
        return False

def main():
    """Main conversion function"""
    # Load dam data
    data_file = Path('e:/PaperV2/图件pro/1109备份数据说明.csv')
    df = pd.read_csv(data_file)
    
    # F-code list (excluding image folders and two-column CSV)
    f_codes = ['F6', 'F7', 'F9', 'F10', 'F17', 'F18', 'F20', 'F21', 'F22']
    
    # Selected dams (from generate_selected_dam_plots.py)
    selected_dam_names = [
        'Dry Falls',
        'Glenmaggie', 
        'Cajuru',
        'Hongmen',
        'Naussac'
    ]
    
    # Output directory
    output_dir = Path('e:/PaperV2/图件pro/data/vtu')
    
    print("Starting CSV to VTU conversion...")
    print(f"Output directory: {output_dir}")
    print(f"F-codes: {f_codes}")
    print(f"Dams: {selected_dam_names}")
    print("-" * 60)
    
    conversion_count = 0
    total_count = 0
    
    for dam_name in selected_dam_names:
        print(f"\nProcessing {dam_name}...")
        for f_code in f_codes:
            total_count += 1
            if csv_to_vtu(dam_name, f_code, output_dir):
                conversion_count += 1
    
    print("-" * 60)
    print(f"\nConversion complete!")
    print(f"Successfully converted: {conversion_count}/{total_count} files")
    print(f"VTU files saved in: {output_dir}")

if __name__ == '__main__':
    main()
