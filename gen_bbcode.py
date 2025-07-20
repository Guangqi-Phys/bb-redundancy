import numpy as np

def identity_matrix(m):
    """
    Creates an identity matrix of size m x m
    
    Args:
        m (int): Size of the square matrix
        
    Returns:
        numpy.ndarray: Identity matrix of size m x m
    """
    return np.eye(m)

def s_matrix(ell):
    """
    Creates an ell x ell matrix where the i-th row has a 1 in column (i+1) mod ell
    and 0 elsewhere.
    
    Args:
        ell (int): Size of the square matrix
        
    Returns:
        numpy.ndarray: ell x ell shift matrix
    """
    S = np.zeros((ell, ell))
    for i in range(ell):
        j = (i + 1) % ell
        S[i, j] = 1
    return S

def tensor_product(A, B):
    """
    Computes the tensor (Kronecker) product of two matrices A and B.
    
    Args:
        A (numpy.ndarray): First input matrix of size m x n
        B (numpy.ndarray): Second input matrix of size p x q
        
    Returns:
        numpy.ndarray: Tensor product matrix of size (m*p) x (n*q)
    """
    return np.kron(A, B)

def matrix_power(A, p):
    """
    Computes the p-th power of matrix A (A^p)
    
    Args:
        A (numpy.ndarray): Square input matrix
        p (int): Power to raise the matrix to (can be positive, negative, or zero)
        
    Returns:
        numpy.ndarray: Matrix A raised to power p
    """
    if not isinstance(p, int):
        raise ValueError("Power must be an integer")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
        
    return np.linalg.matrix_power(A, p)

def bimatrix_sum_mod2(A, B):
    """
    Computes the sum of two matrices modulo 2: (A + B) mod 2
    
    Args:
        A (numpy.ndarray): First input matrix
        B (numpy.ndarray): Second input matrix
        
    Returns:
        numpy.ndarray: Sum of matrices modulo 2
    """
    if not (A.shape == B.shape):
        raise ValueError("All matrices must have the same shape")
        
    # Sum the matrices and take modulo 2
    return np.mod(A + B, 2)

def trimatrix_sum_mod2(A, B, C):
    """
    Computes the sum of three matrices modulo 2: (A + B + C) mod 2
    
    Args:
        A (numpy.ndarray): First input matrix
        B (numpy.ndarray): Second input matrix
        C (numpy.ndarray): Third input matrix
        
    Returns:
        numpy.ndarray: Sum of matrices modulo 2
    """
    if not (A.shape == B.shape == C.shape):
        raise ValueError("All matrices must have the same shape")
        
    # Sum the matrices and take modulo 2
    return np.mod(A + B + C, 2)

def matrix_transpose(A):
    """
    Computes the transpose of matrix A
    
    Args:
        A (numpy.ndarray): Input matrix
        
    Returns:
        numpy.ndarray: Transpose of matrix A
    """
    return A.T

def display_matrix(A):
    """
    Displays the full matrix A in a readable format
    
    Args:
        A (numpy.ndarray): Input matrix to display
        
    Returns:
        None: Prints the matrix to console
    """
    # Set print options to show full matrix without truncation
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(A)
    # Reset print options to default
    np.set_printoptions(threshold=1000, linewidth=75)

def glue_matrices(A, B):
    """
    Horizontally concatenates two matrices A and B to form (A|B)
    
    Args:
        A (numpy.ndarray): First matrix with shape (m x n)
        B (numpy.ndarray): Second matrix with shape (m x p)
        
    Returns:
        numpy.ndarray: Concatenated matrix with shape (m x (n+p))
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrices must have the same number of rows")
        
    return np.hstack((A, B))