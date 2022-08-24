import sys
import unittest
import networkx as nx
import numpy as np
import random
from math import isclose
import copy

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from markovTransforms import TReconnectOriginOfEdgeToOtherNode
from networkGenerator import makeColumnStochastic, getDirectedErdosRenyi, getDirectedColumnStochasticErdosRenyi

# https://gist.github.com/mogproject/fc7c4e94ba505e95fa03
# https://stackoverflow.com/questions/40172281/unit-tests-for-functions-in-a-jupyter-notebook

''' def makeColumnStochastic(g) UNIT TEST '''

import sys
import unittest
import networkx as nx
import numpy as np
from math import isclose

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

class TestMakeColumnStochastic(unittest.TestCase):
    
    def test_WhenAllOnesShouldReturnOneThirds(self):

        C = np.array([[1,1,1],[1,1,1],[1,1,1]])
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g)
        C2 = nx.to_numpy_array(g2)
        self.assertEqual(C.shape, C2.shape)
        self.assertTrue( isclose(n, C2.sum(axis=0)[None, :].reshape(1,n).sum(),abs_tol=1e-6 ) )
        e, v = np.linalg.eig(C2)
        biggest_eigenvalue = np.max(e)
        if biggest_eigenvalue.imag != 0:
            biggest_eigenvalue = np.max(e).real  #https://dsp.stackexchange.com/questions/22807/how-can-i-get-maximum-complex-value-from-an-array-of-complex-values524289-in-p
        print('T1 - eigenvalues: ', e,'-', biggest_eigenvalue)
        
        self.assertTrue( isclose(1, biggest_eigenvalue, abs_tol=1e-6))

    def test_WhenErdosRenyiShouldBeCorrect(self):
        g = getDirectedErdosRenyi(n=10, p=0.5, max_trials=10)
        C = nx.to_numpy_array(g)
        n= C.shape[0]
        g1 = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g1)
        C2 = nx.to_numpy_array(g2)
        self.assertEqual(C.shape, C2.shape)
        self.assertTrue( isclose(n, C2.sum(axis=0)[None, :].reshape(1,n).sum(),abs_tol=1e-6 ) )
        e, v = np.linalg.eig(C2)
        biggest_eigenvalue = np.max(e)
        if biggest_eigenvalue.imag != 0:
            biggest_eigenvalue = np.max(e).real
        print('T2 - eigenvalues: ', e,'-', biggest_eigenvalue)

        self.assertTrue( isclose(1, biggest_eigenvalue, abs_tol=1e-6))

    def test_WhenErdosRenyi2ShouldBeCorrect(self):
        g = getDirectedErdosRenyi(n=3, p=0.5, max_trials=10)
        C = nx.to_numpy_array(g)
        n= C.shape[0]
        g1 = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g1)
        C2 = nx.to_numpy_array(g2)
        self.assertEqual(C.shape, C2.shape)
        self.assertTrue( isclose(n, C2.sum(axis=0)[None, :].reshape(1,n).sum(),abs_tol=1e-6 ) )
        e, v = np.linalg.eig(C2)
        biggest_eigenvalue = np.max(e)
        if biggest_eigenvalue.imag != 0:
            biggest_eigenvalue = np.max(e).real
        print('T3 - eigenvalues: ', e, ' - ', biggest_eigenvalue)

        self.assertTrue( isclose(1, biggest_eigenvalue, abs_tol=1e-6))


unittest.main(argv=[''], verbosity=2, exit=False)