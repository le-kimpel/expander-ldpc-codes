import numpy
import networkx as nx
import sympy
from networkx.algorithms import bipartite

'''
Generate a random bipartite graph whose edge creation has probability p
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
    
    # construct the bipartite adjacency matrix: rows correspond to W, columns to N
    return [[int(B.has_edge(W[i], N[j])) for j in range(len(N))] for i in range(len(W))]

'''
Calculate the cheeger constant of a particular graph
'''
def get_cheeger_constant(B):
    return 

'''
Calculate the rate for a particular linear code
'''
def calc_rate(B):
    return
'''
Calculate the distance for a particular linear code
'''
def calc_distance(B):
    return 

'''
Output the graphical model of B
'''
def visualize(B):
    return 

'''
Get the code by computing the generator matrix
'''
def get_generator_matrix(A):
    return 

'''
Encode data using a linear code whose parity-check matrix is the adjacency matrix
of a biregular (c,d) graph. 
'''
def encode(data):
    return

def decode():
    return 


A = generate_random_graph(3,7,3,0.5)
print(A)
