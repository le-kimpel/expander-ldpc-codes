import numpy

'''
Just puts a matrix A into reduced-row-echelon form
'''
def gauss_jordan(A):
    return

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

