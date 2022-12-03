import numpy

'''
Just puts a matrix A into reduced-row-echelon form

Included from:
https://elonen.iki.fi/code/misc-notes/python-gaussj/

'''
def gauss_jordan(A, pivot_row, pivot_col, eps = 1.0/(10**10)):
  """Puts given matrix (2D array) into the Reduced Row Echelon Form.
     Returns True if successful, False if 'm' is singular.
     NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
     Written by Jarno Elonen in April 2005, released into Public Domain"""
  (h, w) = (len(A), len(A[0]))
  for y in range(0,h):
    maxrow = y
    for y2 in range(y+1, h):    # Find max pivot
      if abs(m[y2][y]) > abs(m[maxrow][y]):
        maxrow = y2
    (m[y], m[maxrow]) = (m[maxrow], m[y])
    if abs(m[y][y]) <= eps:     # Singular?
      return False
    for y2 in range(y+1, h):    # Eliminate column y
      c = m[y2][y] / m[y][y]
      for x in range(y, w):
        m[y2][x] -= m[y][x] * c
  for y in range(h-1, 0-1, -1): # Backsubstitute
    c  = m[y][y]
    for y2 in range(0,y):
      for x in range(w-1, y-1, -1):
        m[y2][x] -=  m[y][x] * m[y2][y] / c
    m[y][y] /= c
    for x in range(h, w):       # Normalize row y
      m[y][x] /= c
  return True

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
    piv2 = ncol - num_rows
    piv1 = ncol - piv_2
    return piv1, piv2


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

