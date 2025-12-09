"""
Add Climate Zone to Dam Data

This script reads P2.csv (dam data with coordinates) and adds climate zone 
information based on the Koppen climate classification raster data.

Input: P2.csv (with LONG_RIV, LAT_RIV columns)
Output: P3.csv (with added climate zone columns)

Climate zone data source: world_koppen1.tif
"""

import pandas as pd
import numpy as np
import rasterio
import sys
from pathlib import Path


# Climate zone mapping: GRIDCODE -> (Code, Cli_Zone, CliType, CliTypeCHN)
CLIMATE_MAPPING = {
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

# Koppen climate zone names for main categories
KOPPEN_ZONE_NAMES = {
    'A': 'Tropical',
    'B': 'Arid',
    'C': 'Temperate',
    'D': 'Continental',
    'E': 'Polar'
}


def get_climate_from_raster(lon, lat, raster_data, transform):
    """
    Get climate zone value from raster data given longitude and latitude.
    
    Args:
        lon: Longitude
        lat: Latitude
        raster_data: 2D numpy array of raster values
        transform: Rasterio transform object
        
    Returns:
        Pixel value at the given coordinates, or None if out of bounds
    """
    try:
        # Convert geographic coordinates to pixel coordinates
        # Using inverse transform: (lon, lat) -> (col, row)
        col, row = ~transform * (lon, lat)
        col, row = int(col), int(row)
        
        # Check bounds
        if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
            return raster_data[row, col]
        return None
    except Exception:
        return None


def add_climate_zones(input_csv, output_csv, raster_path, lon_col='LONG_RIV', lat_col='LAT_RIV'):
    """
    Add climate zone columns to the input CSV file.
    
    Args:
        input_csv: Path to input CSV file (P2.csv)
        output_csv: Path to output CSV file (P3.csv)
        raster_path: Path to Koppen climate raster (world_koppen1.tif)
        lon_col: Name of longitude column
        lat_col: Name of latitude column
    """
    print(f"Reading input data from: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    print(f"Loaded {len(df)} records")
    
    # Check if coordinate columns exist
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"Coordinate columns '{lon_col}' and/or '{lat_col}' not found in input file")
    
    print(f"Using coordinates: {lon_col} (longitude), {lat_col} (latitude)")
    
    # Open raster file
    print(f"Opening climate raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # Read first band
        transform = src.transform
        
        print(f"Raster shape: {raster_data.shape}")
        print(f"Raster bounds: {src.bounds}")
        
        # Initialize new columns
        climate_codes = []
        climate_zones = []
        climate_types = []
        climate_names_cn = []
        climate_names_en = []
        gridcodes = []
        
        # Process each dam
        print("Processing dam coordinates...")
        for idx, row in df.iterrows():
            lon = row[lon_col]
            lat = row[lat_col]
            
            # Get climate zone from raster
            gridcode = get_climate_from_raster(lon, lat, raster_data, transform)
            
            if gridcode is not None and gridcode in CLIMATE_MAPPING:
                code, zone, cli_type, name_cn = CLIMATE_MAPPING[gridcode]
                name_en = KOPPEN_ZONE_NAMES.get(zone, 'Unknown')
                
                climate_codes.append(code)
                climate_zones.append(zone)
                climate_types.append(cli_type)
                climate_names_cn.append(name_cn)
                climate_names_en.append(name_en)
                gridcodes.append(gridcode)
            else:
                # Handle missing or invalid values
                climate_codes.append(None)
                climate_zones.append(None)
                climate_types.append(None)
                climate_names_cn.append(None)
                climate_names_en.append(None)
                gridcodes.append(gridcode)
                
                if gridcode is None:
                    print(f"  Warning: No climate data for dam at ({lon}, {lat}) - coordinates may be outside raster bounds")
                else:
                    print(f"  Warning: Unknown gridcode {gridcode} for dam at ({lon}, {lat})")
        
        # Add new columns to dataframe
        df['Climate_Code'] = climate_codes           # e.g., 'Cfa', 'Bsk'
        df['Climate_Zone'] = climate_zones           # e.g., 'A', 'B', 'C', 'D', 'E'
        df['Climate_Type'] = climate_types           # e.g., 'Cf', 'Bs'
        df['Climate_Name_CN'] = climate_names_cn     # e.g., '常湿温暖气候'
        df['Climate_Name_EN'] = climate_names_en     # e.g., 'Temperate'
        df['Climate_GridCode'] = gridcodes           # Original raster value
    
    # Save output
    print(f"\nSaving output to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Climate Zone Distribution Summary")
    print("="*60)
    
    zone_counts = df['Climate_Zone'].value_counts().sort_index()
    total = len(df)
    valid_count = df['Climate_Zone'].notna().sum()
    
    print(f"\nTotal dams: {total}")
    print(f"Dams with climate data: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Dams without climate data: {total - valid_count}")
    
    print("\nDistribution by Climate Zone:")
    for zone, count in zone_counts.items():
        if pd.notna(zone):
            name = KOPPEN_ZONE_NAMES.get(zone, 'Unknown')
            print(f"  {zone} ({name}): {count} dams ({count/total*100:.1f}%)")
    
    # Detailed climate type distribution
    print("\nDistribution by Climate Type:")
    type_counts = df['Climate_Type'].value_counts().sort_index()
    for ctype, count in type_counts.items():
        if pd.notna(ctype):
            print(f"  {ctype}: {count} dams ({count/total*100:.1f}%)")
    
    return df


def main():
    """Main function"""
    # Try to import from config, otherwise use default paths
    try:
        sys.path.insert(0, str(Path(__file__).parent.absolute()))
        from config import P2_FILE, P3_FILE, CLIMATE_RASTER_FILE
        input_csv = P2_FILE
        output_csv = P3_FILE
        raster_path = CLIMATE_RASTER_FILE
    except ImportError:
        # Set paths manually if config not available
        project_root = Path(__file__).parent
        data_dir = project_root / 'data'
        input_csv = data_dir / 'P2.csv'
        output_csv = data_dir / 'P3.csv'
        # Try to find raster in data/climate folder
        raster_path = data_dir / 'climate' / 'world_koppen1.tif'
    
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Raster: {raster_path}")
    
    # Verify files exist
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        return
    
    if not raster_path.exists():
        print(f"Error: Climate raster not found: {raster_path}")
        return
    
    # Process data
    df = add_climate_zones(input_csv, output_csv, raster_path)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output saved to: {output_csv}")
    print("="*60)


if __name__ == '__main__':
    main()
