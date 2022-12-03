import numpy as np

'''
Just puts a matrix A into reduced-row-echelon form

Included from:
https://towardsdatascience.com/find-the-inverse-of-a-matrix-using-python-3aeb05b48308

'''
def gauss_jordan(M, pivot_col):

    n = len(M)
    M = np.array(M)
    print(M)
    # iterate over matrix rows
    for i in range(0, len(M)):
       
        # initialize row-swap iterator
        j = 1

        k = pivot_col
        
        # select pivot value
        pivot = M[i][i]

        # find next non-zero leading coefficient
        while pivot == 0 and i + j < n:

            # perform row swap operation
            M[[i, i + j]] = M[[i + j, i]]

            print(M)
            # incrememnt row-swap iterator
            j += 1

            # get new pivot
            pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return inverse matrix
            return M

        # extract row
        row = M[i]

        # get 1 along the diagonal
        M[i] = row / pivot

        # iterate over all rows except pivot to get augmented matrix into reduced row echelon form
        for j in [k for k in range(0, n) if k != i]:
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * M[j][i]

    # return inverse matrix
    return M%2

'''
Matrix multiplication
'''
def mmul(A, B):
    return

'''
Convert a (n-m) x n parity matrix to standard form 
'''
def parmat_to_std_form(A):
    return

'''
Get the pivot of the largest square submatrix in A
'''
def largest_sq_submat(A):
    # okay, this is cheating a little
    num_rows = len(A)
    num_cols = len(A[0])
    
    # given the number of rows, return the index of the largest square submatrix of A
    piv_col = num_cols - num_rows
    return 0, piv_col


'''
Get sufficiently small subset orders for a given graph
Included from:

https://github.com/nwalton125/expanders/blob/a517de3ef741185640b64ca5c898087d6fde30f3/python/magicalECCS.py
'''
def ss(i,k):
    if k < 0:
        return []
    elif i == [] or k == 0:
        return [[]]
    smaller_sublists = ss(i[1:], k)
    all_sublists = []
    for s in smaller_sublists:
        if (s == []):
            return [[]]
        all_sublists.append(s)
        if (len(s) < k):
            all_sublists.append([s[0]] + s)
    return all_sublists

