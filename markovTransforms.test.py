

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

class Test_TReconnectOriginOfEdgeToOtherNode(unittest.TestCase):
    
    def test_1_transform(self):
        np.random.seed(30)
        random.seed(30)
        C = np.array([[0,1,1,0],
                      [1,0,0,2],
                      [0,3,2,1],
                      [2,0,1,0]])
        print('T1 - Before Transformation  TReconnectOriginOfEdgeToOtherNode: ', C)
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)

        new_g = TReconnectOriginOfEdgeToOtherNode(g, inPlace=False)
        new_C = nx.to_numpy_array(new_g)
        print('T1 - After  Transformation  TReconnectOriginOfEdgeToOtherNode:: ', new_C)
    
        expected_new_C = np.array([[0,1,1,0],
                                   [1,0,1,2],
                                   [0,3,2,1],
                                   [2,0,0,0]])
        self.assertTrue(np.array_equal(expected_new_C, new_C))
        self.assertEqual(new_g.number_of_nodes(), g.number_of_nodes())
        self.assertEqual(new_g.number_of_edges(), g.number_of_edges())

        sumColumn_C = np.sum(C, axis=0)
        sumColumn_new_C = np.sum(new_C, axis=0)
        self.assertTrue(np.array_equal(sumColumn_C, sumColumn_new_C))

    def test_2_multiple_transform(self):
        np.random.seed(30)
        random.seed(30)
        C = np.array([[0,1,1,0],
                      [1,0,0,2],
                      [0,3,2,1],
                      [2,0,1,0]])
        print('T2 - Before Transformation  TReconnectOriginOfEdgeToOtherNode: ', C)
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)

        new_g = copy.deepcopy(g)
        for _ in range(20):
            new_g = TReconnectOriginOfEdgeToOtherNode(new_g, inPlace=False)
        new_C = nx.to_numpy_array(new_g)
        print('T2 - After  Transformation  TReconnectOriginOfEdgeToOtherNode:: ', new_C)
    
        expected_new_C = np.array([[1., 1., 1. ,0.],
                                   [2., 0., 1. ,0.],
                                   [0., 3., 0., 1.],
                                   [0., 0., 2., 2.]])
        self.assertTrue(np.array_equal(expected_new_C, new_C))
        self.assertEqual(new_g.number_of_nodes(), g.number_of_nodes())
        self.assertEqual(new_g.number_of_edges(), g.number_of_edges())

        sumColumn_C = np.sum(C, axis=0)
        sumColumn_new_C = np.sum(new_C, axis=0)
        self.assertTrue(np.array_equal(sumColumn_C, sumColumn_new_C))


unittest.main(argv=[''], verbosity=2, exit=False)