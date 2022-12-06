import numpy as np
import networkx as nx
import sympy
import random
import math
import utils
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite


'''
Output the graphical model of B!!
'''
def visualize(B):
    nx.draw(B, with_labels=True)
    plt.show()
    return 

'''
Use the Margulis-Gabber-Galil construction to generate an expander and then form a bipartite
graph which inherits its expansion property
'''
def generate_random_graph_v2(n):
    G1 = nx.margulis_gabber_galil_graph(n)
    G2 = nx.margulis_gabber_galil_graph(n)
    
    G1= nx.relabel_nodes(G1, { n: str(n) if n==0 else 'a-'+str(n) for n in  G1.nodes })
    G2= nx.relabel_nodes(G2, { n: str(n) if n==0 else 'b-'+str(n) for n in  G2.nodes })


    vert = list(G2.nodes())
    G_prime =  nx.union(G1,G2)

    for i in G1.nodes:
        indx = 0
        for j in G1.nodes:
            k = vert[indx]
            if G1.has_edge(i,j):
                G_prime.add_edge(i, k)
                if G_prime.has_edge(i,j):
                    G_prime.remove_edge(i,j)
                indx+=1

    for i in G2.nodes:
        for j in G2.nodes:
            if G2.has_edge(i,j):
                if G_prime.has_edge(i,j):
                    G_prime.remove_edge(i,j)

    # remove self-loops
    G_prime.remove_edges_from(nx.selfloop_edges(G_prime))

    # remove disconnected vertices
    G_prime.remove_nodes_from(list(nx.isolates(G_prime)))
    visualize(G_prime)

    # compute the 2nd eigenval
    sq = nx.to_numpy_matrix(G_prime)
    print("2nd eigenval [Margulis]: " + str(get_2nd_eigenval(sq)))
    
    if bipartite.is_bipartite(G_prime):
        W,N = nx.bipartite.sets(G_prime)
        W = list(W)
        N = list(N)
        return G_prime, [[int(G_prime.has_edge(W[i], N[j])) for j in range(len(N))] for i in range(len(W))]
    else:
        return None, None 





'''
Generate a bipartite (d,c) regular graph, very hacky.
We need to also ensure that this graph is connected!
'''
def generate_random_graph(n, c, d):
    # get a bipartite graph G(W \cup N, E), with |W| = n and |N| = (c/d)*n
    B = bipartite.complete_bipartite_graph(n, int(float(c)/float(d))*n)
        
    # get the nodes in the bipartite sets of B
    W,N = nx.bipartite.sets(B)
    W = list(W)
    N = list(N)

    # make sure it is (d,c)-regular, with c > d
    # only need to iterate over the first set and remove enough edges so that each vertex is d-regular
    for i in range(0,len(W)):
        # if the degree of the vertex is not d, remove edges until it is
        j = len(W)
        while(B.degree[i]>d):
            B.remove_edge(i,j)
            j+=1

    for i in range(len(W), len(W)+len(N)):
        j = 0
        while(B.degree[i]>c):
            B.remove_edge(j,i)
            j += 1

    sq = nx.to_numpy_matrix(B)
    print("2nd eigval [Hacky]: " + str(get_2nd_eigenval(sq)))

    if not nx.is_connected(B):
        print("Oops, graph isn't connected, try again with different parameters. [Suggestion: c, d = 2c].")
        return None, None
    
    visualize(B)
    
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
def calc_rate(codeword):
    return math.log(len(codeword), 2) / len(codeword)
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
def encode(data, c, d):
    
    B, A = generate_random_graph(len(data),c,d)
    B2, A2 = generate_random_graph_v2(len(data))
    if (B == A == None and B2 == A2 == None):
        return -1
    A = utils.parmat_to_std_form(A)
    G = get_generator_matrix(A)

    A2 = utils.parmat_to_std_form(A)
    G2 = get_generator_matrix(A2)

    if len(G2) != len(data):
        data.append(0*len(G2)-len(data))
        
    # multiply the two matrices to get the codeword C
    return A, A2, np.matmul(data,G)%2,  np.matmul(data,G2)%2
    
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
    # are satisfied for each bit of the message in_data; while this is *supposed* to be O(mn),
    # I'd hesitate to say with certainty that we've achieved completely linear runtime here ;)

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

# comparing the two generation techniques 
H1, H2, C1, C2 = encode([1,1,0], 20, 10)
print("Rate C1: " + str(calc_rate(C1)))
print("Rate C2: " + str(calc_rate(C2)))

# Multiply the codeword by H
print(np.matmul(C1,np.transpose(H1))%2)
print(np.matmul(C2,np.transpose(H2))%2)

ans = decode((C1+1)%2,H1)
ans2 = decode((C2+1)%2,H2)

print("Decode (C1): " + str(ans))
print("Decode (C2): " + str(ans2))

print(str(is_codeword(ans,H1)))
print(str(is_codeword(ans2, H2)))
