from argparse import Action
import unittest
import networkx as nx
import numpy as np
import random
from math import isclose
import copy
import time

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from metropolisHastings import MetropolisHasting, Acceptance, iterMHBeta, plotMetropolisHastingsResult, analyzeMetropolisHastingsGraphs, loadSamplesFromPickle, loadFromPickle
from markovTransforms import TReconnectOriginOfEdgeToOtherNode
from networkSigma import projectedCovarianceMatrixForDiscreteDynamicalProcesses, discreteSigma2Analytical
from networkGenerator import makeColumnStochastic, getDirectedErdosRenyi, getDirectedColumnStochasticErdosRenyi
from measuresFunctions import getMeasuresDirected
from pickleUtil import pickleSave, pickleLoad

class Test_Acceptance(unittest.TestCase):

    def test_1_Acceptance_Transitivity(self):
        # Acceptance(g, gnext, measure_fn, **parameters):, paramters['beta']
        np.random.seed(30)
        random.seed(30)

        g = getDirectedColumnStochasticErdosRenyi(20, 0.5, return_graph = True)
        gnext = TReconnectOriginOfEdgeToOtherNode(g, inPlace=False)
        parameters = { 'beta': 10 }
        measure_fn = nx.transitivity

        # Act
        p = Acceptance(g, gnext, measure_fn, **parameters)

        self.assertTrue(p <= 1)
        self.assertEqual(0.9913054569904229, p)

    def test_2_Acceptance_Sigma(self):
        # Acceptance(g, gnext, measure_fn, **parameters):, paramters['beta']
        np.random.seed(30)
        random.seed(30)

        g = getDirectedColumnStochasticErdosRenyi(20, 0.5, return_graph = True)
        gnext = TReconnectOriginOfEdgeToOtherNode(g, inPlace=False)
        parameters = { 'beta': 10 }
        measure_fn = discreteSigma2Analytical

        # Act
        p = Acceptance(g, gnext, measure_fn, **parameters)

        self.assertTrue(p <= 1)
        self.assertEqual(0.9971162196654398, p)
        
    def test_3_Acceptance_Sigma_Unweighted_Undirected(self):
        # Acceptance(g, gnext, measure_fn, **parameters):, paramters['beta']
        np.random.seed(30)
        random.seed(30)

        g = nx.random_regular_graph(4, 20)
        g = makeColumnStochastic(g)
        gnext = TReconnectOriginOfEdgeToOtherNode(g, inPlace=False)
        parameters = { 'beta': 10 }
        measure_fn = discreteSigma2Analytical

        # Act
        p = Acceptance(g, gnext, measure_fn, **parameters)
        print('accetpance p = ', p)

        self.assertTrue(p <= 1)
        self.assertEqual(0.9308416520915694, p)

class Test_MetropolisHasting(unittest.TestCase):

    def test_1_MetropolisHasting(self):
        # Arrange
        np.random.seed(30)
        random.seed(30)
        n=5
        Gstart = getDirectedColumnStochasticErdosRenyi(20, 0.5, return_graph = True)
        G = copy.deepcopy(Gstart)
        T = TReconnectOriginOfEdgeToOtherNode
        measure_fn = discreteSigma2Analytical
        b=100
        parameters={'beta':100}

        result=MetropolisHasting(G, T, number_of_samples=1, thinning=50, max_propositions=50, constraint_measure_fn=measure_fn, sample_measure_fn=getMeasuresDirected, **parameters)
        Glast=result['lastnet']
        print(result)

        self.assertTrue(True)

    def test_2_MetropolisHasting(self):
        # Arrange
        np.random.seed(30)
        random.seed(30)
        
        n=5
        Gstart = getDirectedColumnStochasticErdosRenyi(20, 0.5, return_graph = True)
        G = copy.deepcopy(Gstart)
        T = TReconnectOriginOfEdgeToOtherNode
        measure_fn = discreteSigma2Analytical
        b=100
        parameters={'beta':1000}

        result=MetropolisHasting(G, T, number_of_samples=2, thinning=50, max_propositions=100, constraint_measure_fn=measure_fn, sample_measure_fn=getMeasuresDirected, **parameters)
        Glast=result['lastnet']
        print(result)

        self.assertTrue(True)

    def test_1_IterMHBeta_random_regular(self):
        betas = [-100, 0, 1000]

        n=20
        Gstart = nx.random_regular_graph(4, n)
        Gstart = makeColumnStochastic(Gstart, with_random_weights_initialization=False)
        G = copy.deepcopy(Gstart)
        T = TReconnectOriginOfEdgeToOtherNode
        thinning=G.number_of_edges()/2

        tic = time.perf_counter()

        result=iterMHBeta(G, T, 
                    number_of_samples=5, 
                    betas=betas,  
                    relaxation_time=thinning, 
                    constraint_measure_fn=discreteSigma2Analytical, 
                    picklename=f'test_resultKRegular_4_{n}', 
                    sample_measure_fn=getMeasuresDirected,
                    max_propositions=2000)

        toc = time.perf_counter()
        print()
        print("--------------------------------------------------------")
        print("Metropolis hastings in {:.04f} seconds".format(toc-tic))
        print("--------------------------------------------------------")

        plotMetropolisHastingsResult(result, measurename=discreteSigma2Analytical.__name__, betas=betas)
        
        np.save('test_1_IterMHBeta_random_regular_lastnet_C', nx.to_numpy_array(result[-1]['lastnet']))
        # C = nx.to_numpy_array(result[-1]['lastnet'])
        # expectedC = np.load('test_1_IterMHBeta_random_regular_lastnet_C.npy')

        # self.assertTrue(np.array_equal(expectedC, C))
        self.assertTrue(True)
        
    def test_2_IterMHBeta_GraphTooDense(self):
        betas = [-100, 0, 1000]

        n=10
        Gstart = nx.random_regular_graph(4, n)
        Gstart = makeColumnStochastic(Gstart, with_random_weights_initialization=False)
        G = copy.deepcopy(Gstart)
        T = TReconnectOriginOfEdgeToOtherNode
        thinning=G.number_of_edges()/2

        with self.assertRaises(Exception):
            result=iterMHBeta(G, T, 
                        number_of_samples=5, 
                        betas=betas,  
                        relaxation_time=thinning, 
                        constraint_measure_fn=discreteSigma2Analytical, 
                        picklename=f'test_resultRandomeER_0.1_{n}', 
                        sample_measure_fn=getMeasuresDirected,
                        max_propositions=2000)
            
    def test_3_IterMHBeta_random_regular(self):
        betas = [-100, 1000]

        n=20
        Gstart = getDirectedColumnStochasticErdosRenyi(20, 0.1, return_graph = True)
        G = copy.deepcopy(Gstart)
        T = TReconnectOriginOfEdgeToOtherNode
        thinning=G.number_of_edges()/2

        tic = time.perf_counter()

        result=iterMHBeta(G, T, 
                    number_of_samples=3, 
                    betas=betas,  
                    relaxation_time=thinning, 
                    constraint_measure_fn=discreteSigma2Analytical, 
                    picklename=f'test_resultRandomER_0.1_{n}', 
                    sample_measure_fn=getMeasuresDirected,
                    max_propositions=200)

        toc = time.perf_counter()
        print()
        print("--------------------------------------------------------")
        print("Metropolis hastings in {:.04f} seconds".format(toc-tic))
        print("--------------------------------------------------------")

        plotMetropolisHastingsResult(result, measurename=discreteSigma2Analytical.__name__, betas=betas)
        
        C = nx.to_numpy_array(result[-1]['lastnet'])
        expectedC = np.load('test_3_IterMHBeta_random_regular_lastnet_C.npy')
        #np.save('test_3_IterMHBeta_random_regular_lastnet_C', nx.to_numpy_array(result[-1]['lastnet']))

        self.assertTrue(np.array_equal(expectedC, C))
        self.assertTrue(True)
        
class Test_MetropolisHasting(unittest.TestCase):
    def test_1_IterMHBeta_random_regular(self):
    
        np.random.seed(30)
        random.seed(30)

        pickleroot = './data/r_ER-100-p0.1-InDegree-NoSelf-RandomW'

        result = loadFromPickle(pickleroot=pickleroot, measurenames=[], gml=False, errorbar=True, title=None, figsize=None)

class Test_MetropolisHasting_AnalyseResults(unittest.TestCase):
    def test_1_loadSamples_And_Analyze(self):
    
        np.random.seed(30)
        random.seed(30)

        df = loadSamplesFromPickle('r_FixIn-100-DegIn8-InDegree-NoSelf-FixedW', datafolder='./data')
        df, dfm = analyzeMetropolisHastingsGraphs(df, nx.average_clustering)
        dfm.plot(x='beta', y='average_clustering_mean', kind='bar')
        dfm.plot(x='beta', y='average_clustering_mean', yerr=dfm['average_clustering_std'], kind='bar', capsize=4, rot=90)
        df.plot(x='beta', y='average_clustering', kind='scatter')


unittest.main(argv=[''], verbosity=2, exit=False)