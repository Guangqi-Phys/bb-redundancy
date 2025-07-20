import numpy as np
from gen_bbcode import display_matrix

def find_redundancy(H):
    """Finds redundancy structure in a binary matrix using Gaussian elimination mod 2.
    
    Args:
        H (numpy.ndarray): Input matrix to analyze
        
    Returns:
        tuple: (rank, independent_rows, redundant_rows, dependencies)
            - rank: The rank of the matrix
            - independent_rows: List of row indices that form a basis
            - redundant_rows: List of row indices that are redundant
            - dependencies: Dictionary mapping redundant row indices to lists of row 
              indices they depend on
    """
    # Make a copy of the matrix to avoid modifying the original
    H_copy = np.copy(H).astype(int)
    m, n = H_copy.shape
    
    # Perform Gaussian elimination to find independent rows
    rref, pivot_cols, independent_rows = mod2_rref(H_copy)
    
    # Identify redundant rows
    redundant_rows = [i for i in range(m) if i not in independent_rows]
    
    # Find minimal dependencies for each redundant row
    dependencies = {}
    for redundant in redundant_rows:
        dependencies[redundant] = find_minimal_dependencies_for_row(H, redundant, independent_rows)
    
    return (len(independent_rows), independent_rows, redundant_rows, dependencies)


def mod2_rref(H):
    """Perform Reduced Row Echelon Form (RREF) on a binary matrix using mod 2 operations.
    
    Args:
        H (numpy.ndarray): Input matrix to transform
        
    Returns:
        tuple: (rref, pivot_cols, independent_rows)
            - rref: The matrix in RREF form
            - pivot_cols: List of column indices of pivot positions
            - independent_rows: List of row indices that form a basis
    """
    H_rref = np.copy(H).astype(int)
    m, n = H_rref.shape
    
    row_idx = 0
    pivot_cols = []
    independent_rows = []
    # Track original row indices after swapping
    row_mapping = list(range(m))  # Maps current row position to original row index
    
    for col_idx in range(n):
        # Find the first row with a 1 in the current column
        pivot_row = None
        for i in range(row_idx, m):
            if H_rref[i, col_idx] == 1:
                pivot_row = i
                break
        
        if pivot_row is not None:
            # Swap rows if necessary
            if pivot_row != row_idx:
                H_rref[[row_idx, pivot_row]] = H_rref[[pivot_row, row_idx]]
                # Also swap the row mapping to track original indices
                row_mapping[row_idx], row_mapping[pivot_row] = row_mapping[pivot_row], row_mapping[row_idx]
            
            # Eliminate ones in the current column
            for i in range(m):
                if i != row_idx and H_rref[i, col_idx] == 1:
                    # XOR operation (addition mod 2)
                    H_rref[i] = np.mod(H_rref[i] + H_rref[row_idx], 2)
            
            # Store the original row index, not the current position
            independent_rows.append(row_mapping[row_idx])
            pivot_cols.append(col_idx)
            row_idx += 1
            
            # If we've processed all rows, we're done
            if row_idx == m:
                break
    
    return H_rref, pivot_cols, independent_rows


def find_minimal_dependencies_for_row(H, row_idx, independent_rows):
    """Find the minimal set of independent rows that the given row depends on using Gaussian elimination.
    
    This function constructs and solves a linear system over GF(2) to find dependencies efficiently.
    
    Args:
        H (numpy.ndarray): The original matrix
        row_idx (int): Index of the row to find dependencies for
        independent_rows (list): List of indices of independent rows
        
    Returns:
        list: Minimal list of row indices that the target row depends on
    """
    target_row = H[row_idx]
    
    if len(independent_rows) == 0:
        return []
    
    # Extract the independent rows to form the coefficient matrix
    # We need to solve: A * x = target_row (mod 2)
    # where A is the matrix of independent rows and x is the coefficient vector
    A = H[independent_rows].T  # Transpose to get columns as independent rows
    
    # Augment the matrix with the target row
    augmented = np.column_stack([A, target_row])
    
    # Perform Gaussian elimination on the augmented matrix
    m, n = augmented.shape
    
    # Forward elimination
    pivot_row = 0
    for col in range(n - 1):  # Don't process the last column (target)
        # Find pivot
        pivot_found = False
        for row in range(pivot_row, m):
            if augmented[row, col] == 1:
                # Swap rows if needed
                if row != pivot_row:
                    augmented[[pivot_row, row]] = augmented[[row, pivot_row]]
                pivot_found = True
                break
        
        if not pivot_found:
            continue
            
        # Eliminate
        for row in range(m):
            if row != pivot_row and augmented[row, col] == 1:
                augmented[row] = np.mod(augmented[row] + augmented[pivot_row], 2)
        
        pivot_row += 1
    
    # Back substitution to find a particular solution
    solution = np.zeros(len(independent_rows), dtype=int)
    
    # Start from the bottom row and work upward
    for row in range(min(pivot_row - 1, len(independent_rows) - 1), -1, -1):
        # Find the first non-zero coefficient in this row
        pivot_col = -1
        for col in range(len(independent_rows)):
            if augmented[row, col] == 1:
                pivot_col = col
                break
        
        if pivot_col != -1:
            # Calculate the value for this variable
            value = augmented[row, -1]  # RHS value
            
            # Subtract known values from variables to the right
            for col in range(pivot_col + 1, len(independent_rows)):
                if augmented[row, col] == 1:
                    value = np.mod(value + solution[col], 2)
            
            solution[pivot_col] = value
    
    # Verify the solution and extract dependencies
    test_sum = np.zeros_like(target_row)
    dependencies = []
    
    for i, coeff in enumerate(solution):
        if coeff == 1:
            dependencies.append(independent_rows[i])
            test_sum = np.mod(test_sum + H[independent_rows[i]], 2)
    
    # Verify the solution is correct
    if np.array_equal(test_sum, target_row):
        return sorted(dependencies)
    
    # If no solution found, try alternative method using reduced row echelon form
    return find_dependencies_via_rref(H, row_idx, independent_rows)


def find_dependencies_via_rref(H, row_idx, independent_rows):
    """Alternative method using RREF when direct Gaussian elimination fails.
    
    Args:
        H (numpy.ndarray): The original matrix
        row_idx (int): Index of the row to find dependencies for
        independent_rows (list): List of indices of independent rows
        
    Returns:
        list: List of row indices that the target row depends on
    """
    target_row = H[row_idx]
    
    if len(independent_rows) == 0:
        return []
    
    # Create system matrix [A | b] where A contains independent rows as columns
    # and b is the target row
    A = H[independent_rows].T
    augmented = np.column_stack([A, target_row])
    
    # Apply RREF
    rref_matrix, _, _ = mod2_rref(augmented)
    
    # Extract solution from RREF
    solution = np.zeros(len(independent_rows), dtype=int)
    
    # Check if system is consistent
    for row in range(rref_matrix.shape[0]):
        # Check if we have a row like [0 0 0 ... 0 | 1]
        if np.sum(rref_matrix[row, :-1]) == 0 and rref_matrix[row, -1] == 1:
            # Inconsistent system - no exact dependencies found
            return []
    
    # Find pivot columns and extract solution
    for row in range(min(rref_matrix.shape[0], len(independent_rows))):
        # Find pivot column
        pivot_col = -1
        for col in range(len(independent_rows)):
            if rref_matrix[row, col] == 1:
                pivot_col = col
                break
        
        if pivot_col != -1:
            solution[pivot_col] = rref_matrix[row, -1]
    
    # Extract dependencies
    dependencies = []
    for i, coeff in enumerate(solution):
        if coeff == 1:
            dependencies.append(independent_rows[i])
    
    return sorted(dependencies)


def analyze_parity_matrices(HX):
    """Analyzes redundancy structure in parity check matrices HX."""
    print("Analyzing matrix:")
    HX_rank, HX_independent, HX_redundant, HX_dependencies = find_redundancy(HX)
    
    # Calculate dependency counts
    HX_dep_counts = {row: len(deps) for row, deps in HX_dependencies.items()}
    
    print(f"Rank: {HX_rank}")
    print(f"Independent rows: {HX_independent}")
    print(f"Redundant rows: {HX_redundant}")
    print("Redundancy structure:")
    for row, deps in HX_dependencies.items():
        count = HX_dep_counts[row]
        print(f"Row {row} is linearly dependent on {count} rows: {deps}")
    
    # Verify the redundancies are correct
    is_correct, errors = verify_redundancy(HX, HX_redundant, HX_dependencies)
    if is_correct:
        print("\nAll redundancies correctly identified!")
    else:
        print("\nERROR: Some redundancies are incorrect:")
        for row, error_info in errors.items():
            print(f"Row {row} doesn't match the sum of its dependencies")
            print(f"  Redundant row: {error_info['redundant_row']}")
            print(f"  Sum of dependencies: {error_info['sum_of_dependencies']}")
            print(f"  Dependent rows: {error_info['dependent_rows']}")
    
    print()
    
    return (HX_rank, HX_independent, HX_redundant, HX_dependencies, HX_dep_counts)


# Functions for repetition code
def generate_repetition_code_matrices(n):
    """
    Generates parity check matrices for a length-n repetition code.
    For a repetition code, the X-type parity check matrix will have (n-1) rows,
    where each row checks that adjacent bits are equal.
    The Z-type parity check matrix is just a single row of all 1s.
    
    Args:
        n (int): Length of the repetition code (number of physical qubits)
        
    Returns:
        tuple: (HX, HZ) parity check matrices
    """
    # X-type parity check matrix
    HX = np.zeros((n-1, n), dtype=int)
    for i in range(n-1):
        HX[i, i] = 1
        HX[i, i+1] = 1
    
    # Z-type parity check matrix
    HZ = np.ones((1, n), dtype=int)
    
    return HX, HZ

def test_with_repetition_code(n):
    """
    Tests the redundancy analysis functions with a length-n repetition code.
    For a repetition code:
    - The HX matrix should have rank n-1 with no redundant rows
    
    Args:
        n (int): Length of the repetition code
        
    Returns:
        tuple: Results of the analysis
    """
    print(f"\nGenerating and analyzing length-{n} repetition code:")
    HX, _ = generate_repetition_code_matrices(n)  # Ignore HZ
    
    print("\nX-type parity check matrix (HX):")
    display_matrix(HX)
    
    return analyze_parity_matrices(HX)

# Create a matrix with known redundancy patterns for testing
def create_test_matrix_with_complex_redundancies():
    """
    Creates a test matrix with complex redundancy patterns to demonstrate
    the difference between all dependencies and minimal dependencies.
    
    Returns:
        numpy.ndarray: A test matrix with complex redundancies
    """
    # Create a matrix where some redundant rows depend on multiple independent rows,
    # but can be expressed with fewer rows in some cases
    H = np.array([
        [1, 0, 0, 1, 0, 0],  # Row 0: independent
        [0, 1, 0, 0, 1, 0],  # Row 1: independent
        [0, 0, 1, 0, 0, 1],  # Row 2: independent
        [1, 1, 0, 1, 1, 0],  # Row 3: redundant, sum of rows 0 and 1
        [0, 1, 1, 0, 1, 1],  # Row 4: redundant, sum of rows 1 and 2
        [1, 1, 1, 1, 1, 1],  # Row 5: redundant, sum of rows 0, 1, and 2
                               # but also sum of rows 3 and 2, or rows 0 and 4
    ], dtype=int)
    
    return H

# Add this function to your get_redundancy.py file

def verify_redundancy(H, redundant_rows, dependencies):
    """
    Verifies that the identified redundancies are correct by checking if the sum of 
    dependent rows equals the redundant row (mod 2).
    
    Args:
        H (numpy.ndarray): The original matrix
        redundant_rows (list): List of redundant row indices
        dependencies (dict): Dictionary mapping redundant row indices to lists of row
                             indices they depend on
    
    Returns:
        tuple: (is_correct, errors)
            - is_correct: Boolean indicating if all redundancies are correctly identified
            - errors: Dictionary mapping redundant row indices to error information
                     (empty if all redundancies are correct)
    """
    errors = {}
    is_correct = True
    
    for redundant_row in redundant_rows:
        # Get the dependent rows for this redundant row
        dependent_rows = dependencies[redundant_row]
        
        # Calculate the sum of dependent rows (mod 2)
        sum_row = np.zeros_like(H[0])
        for dep_row in dependent_rows:
            sum_row = np.mod(sum_row + H[dep_row], 2)
        
        # Check if the sum equals the redundant row
        if not np.array_equal(sum_row, H[redundant_row]):
            is_correct = False
            errors[redundant_row] = {
                'redundant_row': H[redundant_row],
                'sum_of_dependencies': sum_row,
                'dependent_rows': dependent_rows
            }
    
    return is_correct, errors

if __name__ == "__main__":
    # Test with the BB code from main.py
    try:
        from main import HX, HZ
        print("Analyzing BB code from main.py:")
        analyze_parity_matrices(HX)
        print("\n" + "-"*50)
    except ImportError:
        print("Failed to import HX and HZ from main.py. Skipping BB code analysis.")
    
    # Test with repetition codes of different lengths
    test_with_repetition_code(3)  # Simple case
    print("\n" + "-"*50)
    
    # Test with a complex redundancy pattern
    print("\nTesting with matrix having complex redundancy patterns:")
    H_complex = create_test_matrix_with_complex_redundancies()
    print("Test matrix:")
    display_matrix(H_complex)
    print("\nStandard dependency analysis:")
    rank, indep, redund, deps = find_redundancy(H_complex)
    print(f"Rank: {rank}")
    print(f"Independent rows: {indep}")
    print(f"Redundant rows: {redund}")
    print("Dependencies:")
    for row, dep in deps.items():
        print(f"Row {row} depends on: {dep} (size: {len(dep)})")