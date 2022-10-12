import networkx as nx
import numpy as np
import random


sumColumns = lambda C :  C.sum(axis=0, keepdims=True)

def makeColumnStochastic(g, with_random_weights_initialization=True):
    C=nx.to_numpy_array(g)
    n = C.shape[0]
    
    newC = C
    if (with_random_weights_initialization==True):    
        weights=np.random.uniform(low=0,high=1,size=(n,n))
        newC = np.multiply(weights, C)

    newC = newC/newC.sum(axis=0, keepdims=True)
    
    create_using=nx.Graph
    if (type(g) == type(nx.DiGraph())):
        create_using=nx.DiGraph

    return nx.from_numpy_array(np.array(newC), create_using=create_using)
    

def getDirectedErdosRenyi(n,p,max_trials=50):
    doesNotHaveZeroSumColumn=False
    number_of_trials = 0
    while (doesNotHaveZeroSumColumn==False and number_of_trials <= max_trials):
        g = nx.erdos_renyi_graph(n=n, p=p, directed=True)
        C=nx.to_numpy_array(g)
        n = C.shape[0]
        sumColumn = C.sum(axis=0)[None, :].reshape(1,n)
        if np.all(sumColumn):
            doesNotHaveZeroSumColumn=True
        number_of_trials+=1

    if (doesNotHaveZeroSumColumn==False):
        raise Exception(f"Erdos Renyi (n={n},p={p}) is too sparse, cannot get at least 1 incoming link for every node after {number_of_trials} trials")
    return g

def getDirectedConfigurationModel(din, dout, withSelfLoops=False, return_graph = True):
    g = nx.directed_configuration_model(din, dout, create_using=nx.DiGraph)
    
    if withSelfLoops == False:
        g.remove_edges_from(nx.selfloop_edges(g))
    
    if (return_graph==True):
        return g
    else:
        return nx.to_numpy_array(g)
       
def getSBM(sizes, ps, seed=None, withSelfLoops=False):   
    return nx.stochastic_block_model(sizes, ps, directed=True, selfloops=withSelfLoops, sparse=True, seed=seed)

fixedDegreeSequence = lambda n, din :  [din for _ in range(n)]

def randomDegreeSequence(n, tot):
    d = np.random.uniform(low=0,high=1,size=(n))
    d = (d * tot)/ d.sum(axis=0, keepdims=True)

    d = np.round(d).astype(int)
    diff = tot - d.sum(axis=0)
    if diff > 0:
        d[0]+=diff
    if diff < 0:
        valid_i = random.choice([i for i, deg in enumerate(d) if deg + diff > 0])
        d[valid_i]+=diff

    return list(d)

def flattenIncomingDegree(g, expectedDin):
    '''
    The configuration model doesn't respect the incoming degree or outgoing degree specs 
    after we have removed self loops and double edges.
    To make Din all equals, we're just going to add the missing edges
    (This will obviously also perturb Dout)
    
    Run before adding the weights in the edges
    '''
    din = list(d for _, d in g.in_degree())
    nodeIndexWhereDinNotAsExpected = [i for i, d in g.in_degree() if d != expectedDin]
    
    for node in nodeIndexWhereDinNotAsExpected:
        diff = expectedDin - g.in_degree(node)
        if diff < 0: 
            raise "Something's fishy, if we lost some edges by removing them, fix your code"
        
        for _ in range(diff):
            originNode = random.choice([n for n in g.nodes() if n != node and (n, node) not in g.edges()])
            g.add_edges_from([(originNode, node)])

    return g

def getDirectedColumnStochasticErdosRenyi(n, p, return_graph = True, max_trials=50):
    
    g = getDirectedErdosRenyi(n,p,max_trials)

    gst = makeColumnStochastic(g)
    if (return_graph==True):
        return gst
    else:
        return nx.to_numpy_array(gst)