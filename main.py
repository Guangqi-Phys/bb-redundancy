from gen_bbcode import *
from ldpc import mod2
from get_redundancy import *



if __name__ == "__main__":
    ell = 12
    m = 12

    x = np.kron(s_matrix(ell), np.eye(m))
    y = np.kron(np.eye(ell), s_matrix(m))

    A = trimatrix_sum_mod2(matrix_power(x,3), y, matrix_power(y,2))
    B = trimatrix_sum_mod2(matrix_power(y,3), x, matrix_power(x,2))


    HX = glue_matrices(A, B)
    HZ = glue_matrices(matrix_transpose(B), matrix_transpose(A))

    rankx = mod2.rank(HX)
    rankz = mod2.rank(HZ)

    k = 2*ell*m - rankx - rankz

    print(k)

    analyze_parity_matrices(HX)
    analyze_parity_matrices(HZ)
    







