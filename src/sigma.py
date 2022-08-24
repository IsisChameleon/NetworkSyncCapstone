import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy


createGraphFromC = lambda C : nx.from_numpy_matrix(np.matrix(C), create_using=nx.DiGraph)

# ===========================================================================
# Maximum relative error
# ===========================================================================

SMALL = 2.2204e-16

_maxRelativeError = lambda dX, X : np.max(np.abs(dX) / np.where(np.abs(X) < SMALL , SMALL, np.abs(X)))

_isNegligible = lambda dX, X, tolerance : np.count_nonzero((np.max(np.abs(dX)) > tolerance * SMALL * max(np.max(np.abs(X)), SMALL))) == 0

# ===========================================================================
# Averaging operator U
# https://rich-d-wilkinson.github.io/MATH3030/2-4-centering-matrix.html
# ===========================================================================

''' Centering matrix for a row vector of size N '''

_getCenteringMatrix = lambda N : np.identity(N) - (1/N)*np.ones((N,N))

# ===========================================================================
# Covariance Using analytical formula
# Aim : Calculate projected covariance matrix Ω (Ω_u) that is projected into the orthogonal space to (1,1,1,1,1,...)
# https://www.dropbox.com/s/qlnx56hrtlgucxq/MotifsContributingToSync.pdf?dl=0
# ===========================================================================

'''
(1) For continuous systems

'''

def projectedCovarianceMatrixForContinuousDynamicalProcesses(C, tolerance=10 , max_iterations=10000):
    '''
    C : weighted adjacency matrix of our network of dynamical processes
    tolerance : tol = 100 # Multiples of machine epsilon within which we want to consider a value to be zero.
    max_iterations : maximum terms to calculate in the power series to evaluate the projected covariance matrix
    '''
    
    N = C.shape[0]
    U = _getCenteringMatrix(N)

    # m = 0 first term of the series

    projM = 0.5*U # remember : U.T == U property of this thing U, and U@U == U

    # calculating terms for m=1 ==> infinity (more practically until additional term is negligible)

    leftMultiplier = U @ C.T
    rightMultiplier = C @ U
    dProjM = 0.5 * U

    for i in range(max_iterations):
        # dProjM holds the previous C'^i * C^i term
        # ProjM holds the sum of previous projected C'^i * C^i terms
        
        dProjM = ( leftMultiplier @ dProjM  + dProjM @ rightMultiplier ) / 2
        if _isNegligible(dProjM,projM,tolerance):
            break
        projM = projM + dProjM

    mre = _maxRelativeError(dProjM, projM)

    if np.isnan(mre) or np.isinf(mre):
        print('ERROR power series calculation of projected covariance matrix failed to converge')
        return

    if i == max_iterations:
        mac = np.mean(np.mean(np.abs(projM)))
        print(f'WARNING power series did not converge within the maximum allowed iterations')
        return

    return projM

'''
(2) For discrete systems
'''

def projectedCovarianceMatrixForDiscreteDynamicalProcesses(C, tolerance=10 , max_iterations=10000):
    '''
    C : weighted adjacency matrix of our network of dynamical processes
    tolerance : tol = 100 # Multiples of machine epsilon within which we want to consider a value to be zero.
    max_iterations : maximum terms to calculate in the power series to evaluate the projected covariance matrix
    '''
    
    N = C.shape[0]
    U = _getCenteringMatrix(N)

    # m = 0 first term of the series

    projM = U # remember : U.T == U property of this thing U, and U@U == U

    # calculating terms for m=1 ==> infinity (more practically until additional term is negligible)

    leftComponent=U
    rightComponent=U
    leftMultiplier = U @ C.T
    rightMultiplier = C @ U

    for i in range(max_iterations):
        # dProjM holds the previous C'^i * C^i term
        # ProjM holds the sum of previous projected C'^i * C^i terms
        
        leftComponent = leftComponent @ leftMultiplier
        rightComponent = rightMultiplier @ rightComponent
        dProjM=leftComponent @ rightComponent
        if _isNegligible(dProjM,projM,tolerance):
            break
        projM = projM + dProjM

    mre = _maxRelativeError(dProjM, projM)

    if np.isnan(mre) or np.isinf(mre):
        print('ERROR power series calculation of projected covariance matrix failed to converge')
        return

    if i == max_iterations:
        mac = np.mean(np.mean(np.abs(projM)))
        print(f'WARNING power series did not converge within the maximum allowed iterations')
        return

    return projM


def initialize_nodes_values(g):
    ''' setup initial x values on graph g'''
    for node in g.nodes():
        g.nodes[node]['x']=0.1

def initialize_edges_weights(g):
    for edge in g.edges():
        g.edges[edge]['weight']=0.1


class ProcessOnNetwork(): 
    def __init__(self, G, discrete=False):
        super().__init__()

        # Processing parameters

        # allowed_kwargs = {'initializer'}
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        # initi = kwargs.get('blob')

        # properties

        self.g = copy.deepcopy(G)
        self.C = nx.to_numpy_array(G)
        self.node_initializer = initialize_nodes_values
        self.edge_weight_initializer = initialize_edges_weights
        self.projectedCovarianceMatrix = projectedCovarianceMatrixForDiscreteDynamicalProcesses if discrete==True \
            else projectedCovarianceMatrixForContinuousDynamicalProcesses
        self.updateXs = self._updateXsDiscrete if discrete==True else self._updateXsContinuous
        self.N = G.number_of_nodes()
        
        # check graph has weights on its edges

        edges_attributes_dict_keys = next(iter(self.g.edges(data=True)))[2].keys()

        if (not 'weight' in edges_attributes_dict_keys):
            print('This network has no weights on its edges, initializing with random values')
            self.edge_weight_initializer(self.g)

        # initialize node values
        self.node_initializer(self.g)
        self.Xs = np.array(list(nx.get_node_attributes(self.g, 'x').values())).reshape(1,1,self.N)
            
    def initialize(self):
        self.node_initializer(self.g)

    def projectedG(self):
        U = _getCenteringMatrix(self.C.shape[0])
        projC = U @ self.C @ U
        print(f'projected C: {projC}')
        print(f'eigenvalues and vectors of projected C: {np.linalg.eig(projC)}')
        print(f'eigenvalues and vectors of C: {np.linalg.eig(self.C)}')
        projG = nx.from_numpy_matrix(projC, create_using=nx.DiGraph)
        return projG

    def sigma_2_empirical(self, num_timesteps=1000):

        if (self.Xs.shape[0] == 1):
            print('Please simulate your dynmical process - no empirical measures possible otherwise')
            print('Returning covariance of x(0)')
            return np.cov(X, bias=True)

        if (self.Xs.shape[0] < num_timesteps):
            num_timesteps = self.Xs.shape[0]

        avgCov=0
        for i in range(num_timesteps):
            X = self.Xs[-i,:,:].reshape(1, self.Xs.shape[2])  # Nodes processes values at time -i
            avgCov+=np.cov(X, bias=True)

        avgCov/=num_timesteps
        
        return avgCov

    def sigma_2_analytical(self):
        return np.trace(self.projectedCovarianceMatrix(self.C))/self.C.shape[0]


    def _updateXsDiscrete(self, noise=True):
        # 
        N = self.N

        # take X the value of the dynamical process on each node as a row vector
        #X = np.array(nx.get_node_attributes(self.g, 'x')).reshape(1, N)
        X = self.Xs[-1]
        X.reshape(1,N)

        # Weighted adjacency matrix
        C = self.C

        # I - C
        I = np.identity(N)

        newX = X @ C

        if noise==True:
            # r(t) = mean zero uit variance Gaussian noise
            r= np.random.standard_normal(X.shape)
            newX = newX + r

        self.Xs = np.vstack((self.Xs,newX.reshape(1,1,N)))

    def _updateXsContinuous(self, dt=0.001, noise=True):
        # 
        N = self.N

        # take X the value of the dynamical process on each node as a row vector
        #X = np.array(nx.get_node_attributes(self.g, 'x')).reshape(1, N)
        X = self.Xs[-1]
        X.reshape(1,N)

        # Weighted adjacency matrix
        C = self.C

        # I - C
        I = np.identity(N)
        I_C = I - C

        dX = -X @ I_C * dt

        if noise==True:
            # dw Wiener process
            dW= np.sqrt(dt)*np.random.standard_normal(X.shape)
            dX = dX + dW

        Xnew = X + dX

        self.Xs = np.vstack((self.Xs,Xnew.reshape(1,1,N)))

        #nx.set_node_attributes(self.g, values=Xnew, name='x')

    def updateNetworkWitXs(self):
        nx.set_node_attributes(self.g, values=self.Xs[-1].reshape(1,N),  name='x')

