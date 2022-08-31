#####################################################################
# MC Transformation : Swap 2 edges, 4 separate nodes, do not swap
# into existing edges
# This preserve the number of nodes N, the number of links and the
# degree sequence {z_i} 
####################################################################

import random
import copy
import networkx as nx
from measuresFunctions import getMeasures
import numpy as np

def rebalance_incoming_edges_for_node(G, node: int):
    '''G directed graph
       node node to rebalance incoming weight so that incoming weight '''
    pass

def TReconnectDestinationOfEdgeToOtherNode(g_orig, inPlace=True):
    
    "MCMC transformation that preserves number of edges and nodes, and outgoing degree distribution, but not incoming degree distribution"
    "Applicable for directed networks"
    "This transfromation also preserve the sum of outgoing weights from a node, but not incoming ==> does NOT preserver (1,1,1,1..) eigenvector"
    
    if inPlace == False:
        g=copy.deepcopy(g_orig)
    else:
        g=g_orig

    '''Reconnect incoming edge'''
    N=g.number_of_nodes()
    L=g.number_of_edges()

    #Select a random edge to reconnect
    edge_to_reconnect = random.choice(list(g.edges(data=True)))
    print(f'edge to reconnect:{edge_to_reconnect}')

    #Select another random node to connect that edge to
    node2 = [node for node in g.nodes() if node != edge_to_reconnect[1]][np.random.randint(0,N-1)]
    print(f'dest node selected {node2}')
    
    new_weight = edge_to_reconnect[2]['weight']
    if (edge_to_reconnect[0], node2) in g.edges:
        new_weight += g.get_edge_data(edge_to_reconnect[0], node2)['weight']
    g.add_weighted_edges_from([(edge_to_reconnect[0], node2, new_weight)])
    print(f'new edge: {(edge_to_reconnect[0], node2, new_weight)}')
    g.remove_edge(edge_to_reconnect[0], edge_to_reconnect[1])
    print(f"removed edge: {edge_to_reconnect}, {edge_to_reconnect[2]['weight']}")

    #nodes to rebalance : node2 (added incoming edge), edge_to_reconnect[1] (removed incoming edge)
    # is the rebalance a symmetric operation
    rebalance_incoming_edges_for_node(g, node2)
    rebalance_incoming_edges_for_node(g, edge_to_reconnect[1])
    return(g)

def TReconnectOriginOfEdgeToOtherNode(g_orig, inPlace=True):
    """Transformation for Metropolis Hasting algorithm that keeps out-degree constant, but not in-degree
        MCMC transformation that preserves number of edges and nodes, and incoming degree distribution but not outgoing degree distribution"
        Applicable for directed networks"
        This transfromation also preserve the sum of incoming weights from a node, therefore will preserve eigenvector (1,1,1,...1) if there is one"

    Args:
        g_orig (_type_): original networkx digraph to apply the transformation to
        inPlace (bool, optional): apply the transformation to g_orig or to a deepcopy of it. Defaults to True.

    Returns:
        _type_: newtworkx diGraph
    """

    if inPlace == False:
        g=copy.deepcopy(g_orig)
    else:
        g=g_orig

    '''Reconnect incoming edge'''
    N=g.number_of_nodes()
    L=g.number_of_edges()
    
    # Determine if network edges are weighted
    weighted=True
    if (nx.get_edge_attributes(g, 'weight') == {}):
        weighted=False

    #Select a random edge to reconnect
    edge_to_reconnect = random.choice(list(g.edges(data=True)))

    #Select a random edge that does not exist - keep edge to reconnect destination to where it is
    no_edges = [(i,edge_to_reconnect[1]) for i in g.nodes() if (i,edge_to_reconnect[1]) not in g.edges()]
    
    try:
        new_edge = random.choice(no_edges)
    except IndexError:
        func_name = 'TReconnectOriginOfEdgeToOtherNode' #sys._getframe().f_code.co_name
        raise Exception(f"IndexError: This graph doesn't have any room for swapping edges with {func_name}. No non-existing edge to be created found. Graph not transformed")
    
    if weighted:
        new_weight = edge_to_reconnect[2]['weight']
        g.add_weighted_edges_from([(new_edge[0], new_edge[1], new_weight)])
    else:
        g.add_edges_from([(new_edge[0], new_edge[1])])
        
    #print(f'new edge: {(new_edge[0], new_edge[1], new_weight )}')
    g.remove_edge(edge_to_reconnect[0], edge_to_reconnect[1])
    #print(f"removed edge: {edge_to_reconnect}, {edge_to_reconnect[2]['weight']}")

    return g

def TDeleteEdgeAddEdge(g_orig, inPlace=True):
    ''' Designed for undirected graph - will need to review for directed graph '''
    
    "MCMC transformation that preserves number of edges and nodes, but not degree distribution"
    
    if inPlace == False:
        g=copy.deepcopy(g_orig)
    else:
        g=g_orig

    '''Swap edges'''
    N=g.number_of_nodes()
    L=g.number_of_edges()
    edge = list(g.edges())[np.random.randint(0,L)]
    nodes_with_no_links = [ (i, j) for i in g.nodes() for j in g.nodes() if (i,j) not in g.edges() and i != j]
    node_pair = random.choice(nodes_with_no_links)

    g.add_edge(node_pair[0], node_pair[1])
    g.remove_edge(edge[0], edge[1])
    return(g)

def TSwapEdges(g_orig, inPlace=True):
    ''' Designed for undirected graph - will need to review for directed graph '''
    
    if inPlace == False:
        g=copy.deepcopy(g_orig)
    else:
        g=g_orig
        
    '''Pick 2 links and swap one of the node of each link'''
    
    '''Check N and degree sequence before transformation'''
    N_before = g.number_of_nodes()
    L_before = g.number_of_edges()
    deg_sequence_before = [d for v, d in g.degree()]
    
    
    validSwapExist=False
    i=0
    
    while not validSwapExist and i < 1000:
        edge1 = random.choice(list(g.edges()))
        found=False
        while not found:
            edge2 = random.choice([ e for e in g.edges() if e not in g.edges(edge1[0]) and e not in g.edges(edge1[1]) ])
            if edge1 != edge2:
                found = True

        a1 = (edge1[0], edge2[0])
        a2 = (edge1[1], edge2[1])
        b1 = (edge1[0], edge2[1])
        b2 = (edge1[1], edge2[0])
        #possible_new_edges = [ [a1,a2] if (g.has_edge(a1) or g.has_edge(a2)) is False ,  [b1,b2] if (g.has_edge(b1) or g.has_edge(b2)) is False]
        possible_new_edges = [ [i,j] for [i,j] in [[a1,a2], [b1,b2]] if (g.has_edge(*i) or g.has_edge(*j)) is False ]
        if len(possible_new_edges) != 0:
            validSwapExist=True
            new_edges=random.choice(possible_new_edges)
        i+=1
  
    if (i==1000):
        print("Cannot find a valid transformation, your graph may be too dense")
        return g  #cannot find a valid transformation

    g.add_edge(*new_edges[0])
    g.add_edge(*new_edges[1])
    g.remove_edge(*edge1)
    g.remove_edge(*edge2)
    
    '''Check N and degree sequence after transformation, it needs to be the same as before'''
    N_after = g.number_of_nodes()
    L_after = g.number_of_edges()
    deg_sequence_after = [d for v, d in g.degree()]
    
    if ( N_before != N_after or L_before != L_after or deg_sequence_before != deg_sequence_after):
        print('Your transformation is corrupt, it exits Omega, please review your code')
        print('N: {} to {}'.format(N_before, N_after))
        print('L: {} to {}'.format(L_before, L_after))
        print('Degree Sequence: {} to {}'.format(deg_sequence_before, deg_sequence_after))
    
    return(g)

#################################################################################
# Function to Apply a MCMC transformation T t times and collect measures
#
# Parameters:
# - graph g
# - T transformation (e.g. TSwapEdges)
#
#################################################################################
def markov_iter(g, T, tmax, thinning, measures=None, sampling=False, nb_samples=10):
    
    thinning=int(round(thinning))
    
    if sampling==True:
        if nb_samples > tmax:
            print('Not enough iterations to get the required number of samples, exiting...')
            return
        start_sampling=1
        print('Starting sampling at time {}'.format(start_sampling*thinning))

        s=0
        samples=None
        print('Setting up sampling completed.')
        measures_t=[0 for i in range(tmax)]
        samples_t=[0 for i in range(tmax)]
    
    for t in range(tmax):
        for i in range(thinning):
            g=T(g)
        measures=getMeasures(g, measureList=True, measures=measures)
        measures_t[t]=t*thinning
        if sampling==True:
            if t>=start_sampling:
                print('Taking a sample at time {}'.format(t*thinning))
                samples=getMeasures(g, measureList=True, measures=samples)
                samples_t[s]=t*thinning
                s+=1
        
        
    return { 'measures': measures, 'measures_t': measures_t, 'samples': samples, 'samples_t': samples_t }


