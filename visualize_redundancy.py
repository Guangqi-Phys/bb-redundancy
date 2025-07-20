import numpy as np
from PIL import Image, ImageDraw
import os
from main import *
from get_redundancy import find_redundancy

def create_redundancy_visualization(ell, m, redundant_row, dependencies, matrix_name, output_dir="figure"):
    """
    Creates a PNG visualization of a redundancy pattern.
    
    Args:
        ell (int): Number of columns in the grid
        m (int): Number of rows in the grid 
        redundant_row (int): The redundant row index
        dependencies (list): List of row indices that the redundant row depends on
        matrix_name (str): Name for the output file
        output_dir (str): Output directory for PNG files
    """
    # Create the redundancy set (redundant row + all dependencies)
    redundancy_set = sorted(dependencies + [redundant_row])
    
    # Create 12x12 grid (144 pixels total)
    grid_size = ell * m
    image_array = np.ones((m, ell, 3), dtype=np.uint8) * 212  # Gray background
    
    # Color mapping for redundancy elements
    redundant_color =  [229,117,117]   # Red for the redundant row itself
    dependency_color = [0,160,135]  # Blue for dependency rows
    
    # Map each element in redundancy set to pixel coordinates
    for element in redundancy_set:
        # Calculate row and column in the 12x12 grid
        # Element numbering: first column (0 to m-1), second column (m to 2m-1), etc.
        col = element // m
        row = element % m
        
        # Color the pixel
        if element == redundant_row:
            image_array[row, col] = redundant_color  # Red for redundant row
        else:
            image_array[row, col] = dependency_color  # Blue for dependencies
    
    # Convert to PIL Image and save
    os.makedirs(output_dir, exist_ok=True)
    image = Image.fromarray(image_array, 'RGB')
    
    # Scale up the image for better visibility (each pixel becomes 20x20)
    scale_factor = 20
    large_image = image.resize((ell * scale_factor, m * scale_factor), Image.NEAREST)
    
    # Add grid lines (dash lines between rows and columns)
    draw = ImageDraw.Draw(large_image)
    grid_color = (64, 64, 64)  # Dark gray for grid lines
    
    # Draw vertical lines (column separators)
    for col in range(1, ell):
        x = col * scale_factor
        for y in range(0, m * scale_factor, 4):  # Dash pattern: 4 pixels on, 4 pixels off
            draw.line([(x, y), (x, min(y + 2, m * scale_factor))], fill=grid_color, width=1)
    
    # Draw horizontal lines (row separators)
    for row in range(1, m):
        y = row * scale_factor
        for x in range(0, ell * scale_factor, 4):  # Dash pattern: 4 pixels on, 4 pixels off
            draw.line([(x, y), (min(x + 2, ell * scale_factor), y)], fill=grid_color, width=1)
    
    filename = f"{output_dir}/{ell}x{m}redundancy_{matrix_name}_row{redundant_row}.png"
    large_image.save(filename)
    
    print(f"Saved {filename}")
    print(f"Redundancy set for row {redundant_row}: {redundancy_set}")
    print(f"Dependencies ({len(dependencies)} rows): {dependencies}")
    print(f"Total elements in set: {len(redundancy_set)}\n")
    
    return redundancy_set

def visualize_all_redundancies():
    """
    Analyze and visualize all redundancy patterns for both HX and HZ matrices.
    """
    ell = 6
    m = 6
    
    print(f"Generating redundancy visualizations for {ell}x{m} grid...\n")
    
    # Generate the matrices
    x = np.kron(s_matrix(ell), np.eye(m))
    y = np.kron(np.eye(ell), s_matrix(m))
    
    A = trimatrix_sum_mod2(matrix_power(x,3), y, matrix_power(y,2))
    B = trimatrix_sum_mod2(matrix_power(y,3), x, matrix_power(x,2))
    
    HX = glue_matrices(A, B)
    HZ = glue_matrices(matrix_transpose(B), matrix_transpose(A))
    
    # Analyze redundancies
    print("=== HX Matrix Redundancies ===")
    _, _, redundant_rows_HX, dependencies_HX = find_redundancy(HX)
    
    all_redundancy_sets_HX = []
    for i, redundant_row in enumerate(redundant_rows_HX, 1):
        dependencies = dependencies_HX[redundant_row]
        redundancy_set = create_redundancy_visualization(
            ell, m, redundant_row, dependencies, f"HX{i}"
        )
        all_redundancy_sets_HX.append((redundant_row, redundancy_set))
    
    print("\n=== HZ Matrix Redundancies ===")
    _, _, redundant_rows_HZ, dependencies_HZ = find_redundancy(HZ)
    
    all_redundancy_sets_HZ = []
    for i, redundant_row in enumerate(redundant_rows_HZ, 1):
        dependencies = dependencies_HZ[redundant_row]
        redundancy_set = create_redundancy_visualization(
            ell, m, redundant_row, dependencies, f"HZ{i}"
        )
        all_redundancy_sets_HZ.append((redundant_row, redundancy_set))
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total redundancies found: {len(redundant_rows_HX) + len(redundant_rows_HZ)}")
    print(f"HX matrix redundancies: {len(redundant_rows_HX)}")
    print(f"HZ matrix redundancies: {len(redundant_rows_HZ)}")
    print("\nAll visualization files saved in the 'figure' directory.")
    
    return all_redundancy_sets_HX, all_redundancy_sets_HZ

if __name__ == "__main__":
    # Generate all redundancy visualizations
    hx_sets, hz_sets = visualize_all_redundancies()
    
    # Print final summary of all redundancy sets
    print("\n=== Final Redundancy Sets ===")
    print("\nHX Matrix:")
    for row, redundancy_set in hx_sets:
        print(f"Row {row}: {redundancy_set}")
    
    print("\nHZ Matrix:") 
    for row, redundancy_set in hz_sets:
        print(f"Row {row}: {redundancy_set}")