import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import unittest
import random
from math import isclose

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from measuresFunctions import getMeasuresDirected, printSamplesMeasuresMeanAndStd
from networkGenerator import fixedDegreeSequence, randomDegreeSequence, flattenIncomingDegree, getDirectedConfigurationModel, makeColumnStochastic

class TestGetMeasuresDirected(unittest.TestCase):
    
    def test_GetMeasuresDirected_NonColumnStochasticGraph_Regular(self):

        np.random.seed(30)
        random.seed(30)

        g = nx.random_regular_graph(4, 20)
        measures = getMeasuresDirected(g)

        # Act

        printSamplesMeasuresMeanAndStd(measures)
        
        self.assertTrue(isclose(0.15, measures['average_clustering'], abs_tol=1e-6))
        
    def test_GetMeasuresDirected_ColumnStochasticFixedIndegree(self):

        np.random.seed(30)
        random.seed(30)

        expected_din=4
        N=50
        din = fixedDegreeSequence(N, expected_din)
        dout = randomDegreeSequence(N, N*expected_din)
        self.assertEqual(sum(dout), sum(din))
        g = getDirectedConfigurationModel(din, dout, withSelfLoops=False, return_graph = True)
        g = flattenIncomingDegree(g, expected_din )
        g = makeColumnStochastic(g)
        
        measures = getMeasuresDirected(g)

        # Act

        printSamplesMeasuresMeanAndStd(measures)
        
        self.assertTrue(isclose(1.4249, measures['discreteSigma2Analytical'], abs_tol=1e-4))
        
    
unittest.main(argv=[''], verbosity=2, exit=False)