import numpy
import networkx as nx
from networkx.algorithms import bipartite

'''
Generate an unbalanced bipartite (d,c) regular graph with order n
and return its adjacency matrix
'''
def generate_random_graph(d,c,n):
    # get a bipartite graph G(W \cup N, E), with |W| = n and |N| = (c/d)*n
    B = bipartite.complete_bipartite_graph(n, int(float(c)/float(d))*n)
    W,N = nx.bipartite.sets(B) 

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
    return nx.adjacency_matrix(B)

def encode():
    return

def decode():
    return 


print(generate_random_graph(1,2,3))
