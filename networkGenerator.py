import networkx as nx
import numpy as np

def makeColumnStochastic(g):
    C=nx.to_numpy_array(g)
    n = C.shape[0]
    weights=np.random.uniform(low=0,high=1,size=(n,n))

    newC = np.multiply(weights, C)

    sumColumn = newC.sum(axis=0)[None, :].reshape(1,n)   #Note: does it work with negative edge weights??
    newC = np.where(sumColumn != 0, np.divide(newC, sumColumn),0)

    return nx.from_numpy_array(np.array(newC), create_using=nx.DiGraph)

def getDirectedErdosRenyi(n,p,max_trials=10):
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
    return g

def getDirectedColumnStochasticErdosRenyi(n, p, return_graph = True, max_trials=10):
    
    g = getDirectedErdosRenyi(n,p,max_trials)

    gst = makeColumnStochastic(g)
    if (return_graph):
        return gst
    else:
        return nx.to_numpy_array(gst)