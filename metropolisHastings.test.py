from argparse import Action
import unittest
import networkx as nx
import numpy as np
import random
from math import isclose
import copy

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from metropolisHastings import MetropolisHasting, Acceptance
from markovTransforms import TReconnectOriginOfEdgeToOtherNode
from networkSigma import projectedCovarianceMatrixForDiscreteDynamicalProcesses, discreteSigma2Analytical
from networkGenerator import makeColumnStochastic, getDirectedErdosRenyi, getDirectedColumnStochasticErdosRenyi
from measuresFunctions import getMeasuresDirected

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

        result=MetropolisHasting(G, T, number_of_samples=1, thinning=50, max_propositions=50, measure_fn=measure_fn, sample_measure_fn=getMeasuresDirected, **parameters)
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

        result=MetropolisHasting(G, T, number_of_samples=2, thinning=5000, max_propositions=10000, measure_fn=measure_fn, sample_measure_fn=getMeasuresDirected, **parameters)
        Glast=result['lastnet']
        print(result)

        self.assertTrue(True)


unittest.main(argv=[''], verbosity=2, exit=False)