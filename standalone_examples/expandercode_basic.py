import numpy as np
import networkx as nx
import sympy
import random
import utils
from networkx.algorithms import bipartite

'''
Generate a random bipartite graph whose edge creation has probability p, very hacky.
We need to also ensure that this graph is connected!
'''
def generate_random_graph(n, c, d, p):
    # get a bipartite graph G(W \cup N, E), with |W| = n and |N| = (c/d)*n
    B = bipartite.random_graph(n, int(float(c)/float(d))*n, p)

    while(not nx.is_connected(B)):
        # regenerate the graph ;)
        B = bipartite.random_graph(n, int(float(c)/float(d))*n, p)
        
    # get the nodes in the bipartite sets of B
    W,N = nx.bipartite.sets(B)
    W = list(W)
    N = list(N)

    sq = nx.to_numpy_matrix(B)
    print("2nd eigval: " + str(get_2nd_eigenval(sq)))
    print("cheeger const: " + str(get_cheeger_constant(B)))
    
    # construct the bipartite adjacency matrix: rows correspond to W, columns to N
    return B, [[int(B.has_edge(W[i], N[j])) for j in range(len(N))] for i in range(len(W))]

'''
Calculate second eigenvalue of a square matrix A
'''
def get_2nd_eigenval(A):
    return np.linalg.eigvals(A)[1]

'''
Calculate the rate for a particular linear code, given by:  

http://www.cs.huji.ac.il/~nati/PAPERS/expander_survey.pdf

'''
def calc_rate(code):
    return log(len(code), 2) / len(code[0])
'''
Compute the Hamming Distance between bistrings
'''
def hamming_distance(code1, code2):
    diff = 0
    for i in range(len(code1)):
        if (code1[i] != code2[i]):
            diff+=1
    return diff
'''
Calculate the distance for a particular linear code
'''
def calc_distance(code):
    min_dist = len(code[0])
    for i in range(len(code)):
        for j in range(i+1, len(code)):
            min_dist = min(min_dist, hamming_distance(code[i],code[j]))
    return min_dist

'''
Output the graphical model of B
'''
def visualize(B):
    return 

'''
Get the code by computing the generator matrix from the standard-form parity matrix H
'''
def get_generator_matrix(H):

    # get P^T, compute the transpose
    nrows = len(H)
    ncols = len(H[0])
    P_t = []
    for i in range(0, nrows):
        p_row = []
        for j in range(nrows, ncols):
            p_row.append(H[i][j])
        P_t.append(p_row)

    P_t = np.array(P_t)%2
    P = np.ndarray.transpose(P_t)

    # now concatenate the appropriate I.d. matrix
    n = P.shape[0]

    # identity matrix with same shape as A
    I = np.identity(n=n)

    # form the augmented matrix by concatenating A and I
    G = np.concatenate((P, I), axis=1)
    
    return G 

'''
Check if a word C is a codeword!
'''
def is_codeword(C,H):
    vec = np.matmul(C, np.transpose(H)) % 2
    for entry in vec:
        if(entry != 0):
            return False
    return True

'''
Encode data using a linear code whose parity-check matrix is the adjacency matrix
of a biregular (c,d) graph. 
'''
def encode(data):

    ans = random.sample(range(3, 100), 2)
    ans.sort()
    
    B, A = generate_random_graph(len(data),7,3,0.75)
    A = utils.parmat_to_std_form(A)
    G = get_generator_matrix(A)

    # multiply the two matrices to get the codeword C
    return A, np.matmul(data,G)%2
    
'''
Utilize the FLIP decoding algorithm specified by http://people.seas.harvard.edu/~madhusudan/courses/Spring2017/scribe/lect13.pdf
'''
def decode(in_data, H):

    # preliminary: stick the variables corresponding to unsatisfied constraints in the S_i
    # for each vertex in the columns of H, make sure that each is satisfied and count
    # the number of unsatisfied constraints
    S = []
    for indx in range(0,len(H)):
        S.append(indx)

    # while the set of variables is unsatified, make sure the parity check equations
    # are satisfied for each bit of the message in_data; this is O(mn)

    for indx in S:
        row = H[indx]
        count = 0
        for i in range(0,len(in_data)):
            if(row[i] == in_data[i] == 1):
                count += 1
        if (count%2 == 0):
            # constraint satisfied, get rid of the row
            S.remove(indx)
        else:
            in_data[i] = (in_data[i] + 1)%2
            
    return in_data

H, C = encode([1,1,0])

#print(np.matmul(C,np.transpose(H)))
ans = decode(C,H)
print(is_codeword(ans,H))


