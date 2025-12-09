import pyvista as pv
from pathlib import Path

vtu_file = Path(r'e:\PaperV2\图件pro\data\vtu\F6\Dry Falls.vtu')

mesh = pv.read(str(vtu_file))

print("=== Mesh Information ===")
print(f"Number of points: {mesh.n_points}")
print(f"Number of cells: {mesh.n_cells}")
print(f"Point data arrays: {list(mesh.point_data.keys())}")
print(f"Cell data arrays: {list(mesh.cell_data.keys())}")
print(f"Mesh type: {type(mesh)}")
print(f"Cell types: {mesh.get_cell(0).type if mesh.n_cells > 0 else 'N/A'}")

print("\n=== Sample Points (first 5) ===")
print(mesh.points[:5])

print("\n=== Sample Values (first 5) ===")
field_name = list(mesh.point_data.keys())[0]
print(f"Field: {field_name}")
print(mesh.point_data[field_name][:5])

print("\n=== Connectivity Information ===")
if hasattr(mesh, 'faces'):
    print(f"Faces: {mesh.faces}")
if hasattr(mesh, 'cells'):
    print(f"Cells shape: {mesh.cells.shape if hasattr(mesh.cells, 'shape') else 'N/A'}")
    print(f"First few cells: {mesh.cells[:20]}")
