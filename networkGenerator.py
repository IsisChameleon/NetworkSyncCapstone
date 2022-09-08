import networkx as nx
import numpy as np


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

def getDirectedConfigurationModel(din, dout, withSelfLoops=False, withRandomWeightsInitialization=False,  return_graph = True):
    g = nx.directed_configuration_model(din, dout)
    
    g = nx.DiGraph(g)
    
    if withSelfLoops == False:
        g.remove_edges_from(nx.selfloop_edges(g))
    
    actualDin = list(d for _, d in g.in_degree())
    actualDout = list(d for _, d in g.out_degree())
    
    g = makeColumnStochastic(g, withRandomWeightsInitialization)
    if (return_graph==True):
        return g
    else:
        return nx.to_numpy_array(g)
    
fixedDegreeSequence = lambda n, din :  [din for _ in range(n)]

def randomDegreeSequence(n, tot):
    d = np.random.uniform(low=0,high=1,size=(1,n))
    print(d)
    d = (d * tot)/ d.sum(axis=0, keepdims=True)
    print('random degree sequence 2:', d)
    d = np.round(d).astype(int)
    print('random degree sequence 3:', d, np.sum(d))
    return list(d)

def getDirectedColumnStochasticErdosRenyi(n, p, return_graph = True, max_trials=50):
    
    g = getDirectedErdosRenyi(n,p,max_trials)

    gst = makeColumnStochastic(g)
    if (return_graph==True):
        return gst
    else:
        return nx.to_numpy_array(gst)