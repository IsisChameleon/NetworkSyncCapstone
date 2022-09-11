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

from networkGenerator import flattenIncomingDegree, makeColumnStochastic, getDirectedErdosRenyi, \
sumColumns, randomDegreeSequence, fixedDegreeSequence, \
getDirectedConfigurationModel, flattenIncomingDegree

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
    
    def test_sumColumns(self):
        C = np.array([[0	,0.38074849,	0.66304791,	0.16365073,	0.96260781],
                      [0.34666184,	0	,0.2350579,	0.58569427,	0.4066901    ],
                      [0.13623432,	0.54413629,	0	,0.76685511,	0.93385014],
                      [0.08970338,	0.19577126,	0.99419368,	0	,0.23898637],
                      [0.62909983	,0.73495258	,0.68834438	,0.03113075,	0]])
        expectedSum = np.array([[1.20169937,	1.85560862,	2.58064387,	1.54733086	,2.54213442]]).reshape(1, C.shape[1])
        
        # act
        sumC = sumColumns(C)
        
        # assert
        self.assertTrue(np.all(np.isclose(sumC, expectedSum, atol=1e-06)))
        
    def test_makeColumnStochastic_simpleC_withoutWeightInitialization(self):
        np.random.seed(30)
        random.seed(30)
        
        C = np.array([[0	,0.38074849,	0.66304791,	0.16365073,	0.96260781],
                      [0.34666184,	0	,0.2350579,	0.58569427,	0.4066901    ],
                      [0.13623432,	0.54413629,	0	,0.76685511,	0.93385014],
                      [0.08970338,	0.19577126,	0.99419368,	0	,0.23898637],
                      [0.62909983	,0.73495258	,0.68834438	,0.03113075,	0]])
        
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g, with_random_weights_initialization=False)
        C2 = nx.to_numpy_array(g2)
        
        sumC2 = sumColumns(C2)

        self.assertTrue(np.array_equal(sumC2, np.array([[1,1,1,1,1]])))
        
    def test_makeColumnStochastic_simpleC_withWeightInitialization(self):
        np.random.seed(30)
        random.seed(30)
        
        C = np.array([[0	,0.38074849,	0.66304791,	0.16365073,	0.96260781],
                      [0.34666184,	0	,0.2350579,	0.58569427,	0.4066901    ],
                      [0.13623432,	0.54413629,	0	,0.76685511,	0.93385014],
                      [0.08970338,	0.19577126,	0.99419368,	0	,0.23898637],
                      [0.62909983	,0.73495258	,0.68834438	,0.03113075,	0]])
        
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g, with_random_weights_initialization=True)
        C2 = nx.to_numpy_array(g2)
        
        sumC2 = sumColumns(C2)

        #self.assertTrue(np.array_equal(sumC2, np.array([[1,1,1,1,1]])))
        self.assertTrue(np.all(np.isclose(sumC2, np.array([[1,1,1,1,1]]), atol=1e-12)))
    
    def test_WhenAllOnesShouldReturnOneThirds(self):

        C = np.array([[1,1,1],[1,1,1],[1,1,1]])
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g, with_random_weights_initialization=False)
        C2 = nx.to_numpy_array(g2)

        self.assertEqual(C.shape, C2.shape)
        self.assertTrue(np.array_equal(C/3, C2))
        self.assertTrue( isclose(n, C2.sum(axis=0)[None, :].reshape(1,n).sum(),abs_tol=1e-6 ) )
        e, v = np.linalg.eig(C2)
        biggest_eigenvalue = np.max(e)
        if biggest_eigenvalue.imag != 0:
            biggest_eigenvalue = np.max(e).real  #https://dsp.stackexchange.com/questions/22807/how-can-i-get-maximum-complex-value-from-an-array-of-complex-values524289-in-p
        print('T1 - eigenvalues: ', e,'-', biggest_eigenvalue)
        
        self.assertTrue( isclose(1, biggest_eigenvalue, abs_tol=1e-6))
        
    def test_When1PerColumnShouldReturnSame(self):

        C = np.array([[1,1,1],[0,0,0],[0,0,0]])
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)
        g2 = makeColumnStochastic(g)
        C2 = nx.to_numpy_array(g2)

        self.assertEqual(C.shape, C2.shape)
        self.assertTrue(np.array_equal(C, C2))
        self.assertTrue( isclose(n, C2.sum(axis=0)[None, :].reshape(1,n).sum(),abs_tol=1e-6 ) )
        e, v = np.linalg.eig(C2)
        biggest_eigenvalue = np.max(e)
        if biggest_eigenvalue.imag != 0:
            biggest_eigenvalue = np.max(e).real  #https://dsp.stackexchange.com/questions/22807/how-can-i-get-maximum-complex-value-from-an-array-of-complex-values524289-in-p
        print('T11 - eigenvalues: ', e,'-', biggest_eigenvalue)
        
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
        
    def test_makeColumnsStochastic_undirected(self):
        np.random.seed(30)
        random.seed(30)
        
        # arrange
        g = nx.random_regular_graph(4, 5)
        
        # act
        g2 = makeColumnStochastic(g)
        
        C2 = nx.to_numpy_array(g2)
        C = nx.to_numpy_array(g)
        n= C.shape[0]
        print('g2 matrix:', C2, ',sum columns:', C2.sum(axis=0).reshape(1,n))

        self.assertEqual(C.shape, C2.shape)
        #self.assertTrue( isclose(n, C2.sum(axis=0)[None, :].reshape(1,n).sum(),abs_tol=1e-6 ) )
        
        e, v = np.linalg.eig(C2)
        biggest_eigenvalue = np.max(e)
        if biggest_eigenvalue.imag != 0:
            biggest_eigenvalue = np.max(e).real
        print('T4 - eigenvalues: ', e, ' - ', biggest_eigenvalue)

        # self.assertTrue( isclose(1, biggest_eigenvalue, abs_tol=1e-6))
        
        self.assertEqual(type(g), type(g2))
        self.assertEqual(type(g), type(nx.Graph()))
        
class TestConfigurationModel(unittest.TestCase):
    def test_FixedDegreeSequence_works(self):
        actual = fixedDegreeSequence(10, 3)
        expected = [3 for _ in range(10)]
        self.assertEqual(actual, expected)
        
    def test_RandomDegreeSequence(self):
        np.random.seed(30)
        random.seed(30)
        actual = randomDegreeSequence(10, 100)
        expectedLength = 10
        expectedSum = 100
        self.assertEqual(len(actual), expectedLength)
        self.assertEqual(sum(actual), expectedSum)
        print('test_RandomDegreeSequence actual sequence:', actual)
        
    def test_getDirectedConfigurationModel(self):
        np.random.seed(30)
        random.seed(30)
        din = fixedDegreeSequence(8, 5)
        dout = randomDegreeSequence(8, 8*5)

        g = getDirectedConfigurationModel(din, dout, withSelfLoops=False, return_graph = True)

        self.assertEqual([4, 4, 2, 4, 3, 3, 3, 3], list(d for _, d in g.in_degree()))
        self.assertEqual([6, 2, 4, 1, 4, 2, 6, 1], list(d for _, d in g.out_degree()))
        
    def test_flattenDin(self):

        expected_din=5
        din = fixedDegreeSequence(8, expected_din)
        dout = randomDegreeSequence(8, 8*expected_din)
        g = getDirectedConfigurationModel(din, dout, withSelfLoops=False, return_graph = True)
        g = flattenIncomingDegree(g, expected_din )
        self.assertEqual(list(d for _, d in g.in_degree()), [expected_din for _ in range(8)])
        
    def test_columnStochasticConfigurationModelWithFixedInDegree(self):
        
        expected_din=5
        N=10
        din = fixedDegreeSequence(N, expected_din)
        dout = randomDegreeSequence(N, N*expected_din)
        self.assertEqual(sum(dout), sum(din))
        g = getDirectedConfigurationModel(din, dout, withSelfLoops=False, return_graph = True)
        g = flattenIncomingDegree(g, expected_din )
        g = makeColumnStochastic(g)
        self.assertEqual(list(d for _, d in g.in_degree()), [expected_din for _ in range(N)])
                
        C = nx.to_numpy_array(g)
        sumC = sumColumns(C)
        self.assertTrue(np.all(np.isclose(sumC, np.array([[1 for _ in range(N)]]), atol=1e-12)))
        
    def test_columnStochasticConfigurationModelWithFixedInDegree(self):
        
        expected_din=5
        N=8
        din = fixedDegreeSequence(N, expected_din)
        dout = randomDegreeSequence(N, N*expected_din)
        self.assertEqual(sum(dout), sum(din))
        g = getDirectedConfigurationModel(din, dout, withSelfLoops=False, return_graph = True)
        g = flattenIncomingDegree(g, expected_din )
        g = makeColumnStochastic(g, with_random_weights_initialization=False)
        
        self.assertEqual(list(d for _, d in g.in_degree()), [expected_din for _ in range(N)])
                
        C = nx.to_numpy_array(g)
        sumC = sumColumns(C)
        self.assertTrue(np.all(np.isclose(sumC, np.array([[1 for _ in range(N)]]), atol=1e-12)))
        #self.assertTrue(np.all(C == 0.2 or C == 0))
        
        

unittest.main(argv=[''], verbosity=2, exit=False)