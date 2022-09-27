

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

from markovTransforms import TReconnectOriginOfEdgeToOtherNode, TDeleteEdgeAddEdge
from networkGenerator import sumColumns

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
        print('T2 Edges attributes', nx.get_edge_attributes(g, "weight"))

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

class Test_AddEdgeDeleteEdge(unittest.TestCase):
    
    def test_3_transformAddEdgeDeleteEdge(self):
        np.random.seed(30)
        random.seed(30)
        C = np.array([[0,1,1,0],
                      [1,0,0,2],
                      [0,3,2,1],
                      [2,0,1,0]])
        print('T3 - Before Transformation  TDeleteEdgeAddEdge: ', C)
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)

        new_g = TDeleteEdgeAddEdge(g, inPlace=False)
        new_C = nx.to_numpy_array(new_g)
        print('T3 - After  Transformation  TDeleteEdgeAddEdge:: ', new_C)
    
        expected_new_C = np.array([[0,1,1,0],
                                   [1,0,0,2],
                                   [2,3,0,1],
                                   [2,0,1,0]])
        self.assertTrue(np.array_equal(expected_new_C, new_C))
        self.assertEqual(new_g.number_of_nodes(), g.number_of_nodes())
        self.assertEqual(new_g.number_of_edges(), g.number_of_edges())
        
    def test_3_transformAddEdgeDeleteEdge_maintainColumnStochastic(self):
        np.random.seed(30)
        random.seed(30)
        C = np.array([[0,1,1,0],
                      [1,0,0,2],
                      [0,3,2,1],
                      [2,0,1,0]])
        print('T4 - Before Transformation  TDeleteEdgeAddEdge: ', C)
        n= C.shape[0]
        g = nx.from_numpy_array(C, create_using=nx.DiGraph)

        new_g = TDeleteEdgeAddEdge(g, inPlace=False, preserveColumnStochastic=True)
        new_C = nx.to_numpy_array(new_g)
        print('T4 - After  Transformation  TDeleteEdgeAddEdge:: ', new_C)
        expected_new_C = np.array([[0,0.25,0.5,0],
                                   [0.2,0,0,2/3],
                                   [0.4,0.75,0,1/3],
                                   [0.4,0,0.5,0]])
        self.assertTrue(np.array_equal(expected_new_C, new_C))
        self.assertEqual(new_g.number_of_nodes(), g.number_of_nodes())
        self.assertEqual(new_g.number_of_edges(), g.number_of_edges())
        sumC = sumColumns(new_C)
        self.assertTrue(np.all(np.isclose(sumC, np.array([[1 for _ in range(new_C.shape[0])]]), atol=1e-12)))



unittest.main(argv=[''], verbosity=2, exit=False)