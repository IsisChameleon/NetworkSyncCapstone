import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import unittest
import random

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from networkSigma import  discreteSigma2Analytical
from networkGenerator import makeColumnStochastic, sumColumns

class TestDiscreteSigma2Analytical(unittest.TestCase):
    
    def test_with_random_regular_graph_unweighted_undirected(self):

        np.random.seed(30)
        random.seed(30)

        g = nx.random_regular_graph(4, 20)
        measure_fn = discreteSigma2Analytical

        # Act

        self.assertRaises(Exception, measure_fn, g)
        
    def test_with_random_regular_graph_columnstochastic_undirected(self):

        np.random.seed(30)
        random.seed(30)

        g = nx.random_regular_graph(4, 20)
        measure_fn = discreteSigma2Analytical
        g2=makeColumnStochastic(g, with_random_weights_initialization=False)

        # Act
        C2 = nx.to_numpy_array(g2)
        eigs=np.linalg.eigvals(C2)
        sigma=measure_fn(g2)

        self.assertEqual(type(g2), type(nx.Graph()))
        self.assertTrue(np.array_equal(sumColumns(C2), np.ones((1,20))))
        print('random_regular_graph_columnstochastic_undirected eigenvalues:' ,np.linalg.eigvals(C2))
        self.assertAlmostEqual(eigs[0],1, places=12)
        
        print(sigma)
        
        # Assert
        self.assertEqual(sigma, 1.2986629678816053)

unittest.main(argv=[''], verbosity=2, exit=False)