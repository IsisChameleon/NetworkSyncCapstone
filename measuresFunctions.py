import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


###########################################################################################################################################
# https://networkx.org/documentation/networkx-1.9.1/_modules/networkx/algorithms/components/connected.html#connected_component_subgraphs
###########################################################################################################################################
def connected_component_subgraphs(G, copy=True):
    for c in nx.connected_components(G):
        if copy:
            yield G.subgraph(c).copy()
        else:
            yield G.subgraph(c)

#############################################################################
#   findClique
#############################################################################

def findCliques(G):
    # can be long for big graphs
    return list(enumerate_all_cliques(G))

def clustering(g):
    #Cnet and nx.clustering doesn't work for multigraphs, so we need to make sure the graph is a simple one
    if type(g) == type(nx.Graph()):
        multigraph=0
    else:
        multigraph=1
        print('Warning: clustering calculated on a multigraph')
    g2=nx.Graph(g)
    Cnet=nx.transitivity(g2)
    return 

#############################################################################
#   getMeasures(g) returns all required measures
#############################################################################

import numpy as np
import networkx as nx
def getMeasures(g, measureList=False, measures=None):
    
    N=g.number_of_nodes()
    
    sum_degree_nodes=np.sum([g.degree[i] for i in g.nodes])
    
    average_degree = sum_degree_nodes/ N
    
    Ex2=(1/N)*(np.sum([g.degree[i]**2 for i in g.nodes]))
    Ex=average_degree
    sigma_z=np.sqrt(Ex2  - (Ex)**2)
    
    degree_variability=sigma_z/average_degree
      
    #Cnet and nx.clustering doesn't work for multigraphs, so we need to make sure the graph is a simple one
    if type(g) == type(nx.Graph()):
        multigraph=0
    else:
        multigraph=1
    g2=nx.Graph(g)
    average_clustering=sum(nx.clustering(g2).values()) / N
    Cnet=nx.transitivity(g2)
    
    #We measures the distances on the largest connected component
    number_of_components = nx.number_connected_components(g)
    number_of_nodes_in_largest_component = N
    if number_of_components == 1:
        diameter=nx.diameter(g)
        average_di=nx.shortest_paths.average_shortest_path_length(g)
    else:
        g1 = max(connected_component_subgraphs(g, copy=True), key=len)
        diameter=nx.diameter(g1)
        average_di=nx.shortest_paths.average_shortest_path_length(g1)
        number_of_nodes_in_largest_component = g1.number_of_nodes()
        
    
    #size largest clique
    cliques= list(nx.find_cliques(g))
    cliques.sort(key=lambda x:len(x), reverse=True)
    N_largest_clique=len(cliques[0])
    
    #degree sequence
    degree_sequence= [d for v, d in g.degree()]
    
    
        
    if (measureList==False):
        return { \
            'N': N, \
            'L': g.number_of_edges(), \
            'average_degree': average_degree, \
            'sigma_z': sigma_z, \
            'average_clustering': average_clustering, \
            'diameter': diameter,
            'Cnet': Cnet, \
            'average_di': average_di, \
            'degree_variability': degree_variability, \
            'number_of_components': number_of_components, \
            'N_K1': number_of_nodes_in_largest_component, \
            'multigraph': multigraph, \
            'N_C1': N_largest_clique \
               }
    if (measures==None):
        return { \
            'N': [N], \
            'L': [g.number_of_edges()], \
            'average_degree': [average_degree], \
            'sigma_z': [sigma_z], \
            'average_clustering': [average_clustering], \
            'diameter': [diameter], \
            'Cnet': [Cnet], \
            'average_di': [average_di], \
            'degree_variability': [degree_variability], \
            'number_of_components' : [number_of_components], \
            'N_K1': [number_of_nodes_in_largest_component], \
            'multigraph': [multigraph], \
            'N_C1': [N_largest_clique] \
               }
    else:
        measures['N'].append(N)
        measures['L'].append(g.number_of_edges())
        measures['average_degree'].append(average_degree)
        measures['sigma_z'].append(sigma_z)
        measures['average_clustering'].append(average_clustering)
        measures['diameter'].append(diameter)
        measures['Cnet'].append(Cnet)
        measures['average_di'].append(average_di)
        measures['degree_variability'].append(degree_variability)
        measures['number_of_components'].append(number_of_components)
        measures['N_K1'].append(number_of_nodes_in_largest_component)
        measures['multigraph'].append(multigraph)
        measures['N_C1'].append(N_largest_clique)
        return measures

from networkSigma import discreteSigma2Analytical
def getMeasuresDirected(g, measureList=False, measures=None):
    
    N=g.number_of_nodes()
    L=g.number_of_edges()
    
    sum_degree_nodes=np.sum([g.degree[i] for i in g.nodes])
    average_degree = sum_degree_nodes/ N
    
    Ex2=(1/N)*(np.sum([g.degree[i]**2 for i in g.nodes]))
    Ex=average_degree
    sigma_z=np.sqrt(Ex2  - (Ex)**2)
    degree_variability=sigma_z/average_degree
      
    #Cnet and nx.clustering doesn't work for multigraphs, so we need to make sure the graph is a simple one
    weighted_average_clustering=sum(nx.clustering(g, weight='weight').values()) / N
 
    sigma2 = discreteSigma2Analytical(g)
        
    if (measureList==False):
        return { \
            'N': N, \
            'L': L, \
            'average_degree': average_degree, \
            'sigma_z': sigma_z, \
            'weighted_average_clustering': weighted_average_clustering, \
                'discreteSigma2Analytical': sigma2
               }
    if (measures==None):
        return { \
            'N': [N], \
            'L': [L], \
            'average_degree': [average_degree], \
            'sigma_z': [sigma_z], \
            'weighted_average_clustering': [weighted_average_clustering], \
                 'discreteSigma2Analytical': [sigma2]
               }
    else:
        measures['N'].append(N)
        measures['L'].append(L)
        measures['average_degree'].append(average_degree)
        measures['sigma_z'].append(sigma_z)
        measures['weighted_average_clustering'].append(weighted_average_clustering)
        measures['discreteSigma2Analytical'].append(sigma2)
        return measures
    

def plotDegreeDistribution(g):
    deg_sequence =  [d for v, d in g.degree()]
    fig, ax = plt.subplots()
    plt.hist(deg_sequence,density=True,bins=range(np.max(deg_sequence)), histtype='bar')
    return fig, ax



def printMeasures(measures):
    if type(measures['N'])!=list:
        print('Number of nodes N:              {}'.format(measures['N']))
        print('Number of links L:              {}'.format(measures['L']))
        print('Average node degree <z>:        {:.04f}'.format(measures['average_degree']))
        print('Deviation of degree sigma_z:    {:.04f}'.format(measures['sigma_z']))
        print('Degree variability sigma_z/<z>: {:.04f}'.format(measures['degree_variability']))
        print('Number of components:           {}'.format(measures['number_of_components']))
        print('Size of largest component:      {}'.format(measures['N_K1']))
        print('Size of largest clique:         {}'.format(measures['N_C1']))
        print('Distances are measured on the largest component:')
        print('Diameter :                      {:.04f}'.format(measures['diameter']))
        print('<<d>>:                          {:.04f}'.format(measures['average_di']))
        print('For multigraph Cnet and <C> is measured after transforming the graph into a simple graph')
        print('Multigraph: 1 - Yes, 0 - No     {}'.format(measures['multigraph']))
        print('Cnet:                           {:.04f}'.format(measures['Cnet'])) 
        print('<C>:                            {:.04f}'.format(measures['average_clustering']))      
    else:
        print('Number of nodes N:              {:.04f}  +/-  {:.04f}'.format(np.average(measures['N']), np.std(measures['N']) ))
        print('Number of links L:              {:.04f}  +/-  {:.04f}'.format(np.average(measures['L']), np.std(measures['L'])))
        print('Average node degree <z>:        {:.04f}  +/-  {:.04f}'.format(np.average(measures['average_degree']),np.std(measures['average_degree'])))
        print('Deviation of degree sigma_z:    {:.04f}  +/-  {:.04f}'.format(np.average(measures['sigma_z']), np.std(measures['sigma_z'])))
        print('Degree variability sigma_z/<z>: {:.04f}  +/-  {:.04f}'.format(np.average( measures['degree_variability']),np.std(measures['degree_variability'])))
        print('Number of components:           {:.04f}  +/-  {:.04f}'.format(np.average(measures['number_of_components']), np.std(measures['number_of_components'])))
        print('Size of largest component:      {:.04f}  +/-  {:.04f}'.format(np.average(measures['N_K1']), np.std(measures['N_K1'])))
        print('Size of largest clique:         {:.04f}  +/-  {:.04f}'.format(np.average(measures['N_C1']), np.std(measures['N_C1'])))
        print('Distances are measured on the largest component:')
        print('<<d>>:                          {:.04f}  +/-  {:.04f}'.format(np.average(measures['average_di']), np.std(measures['average_di'])))
        print('Diameter :                      {:.04f}  +/-  {:.04f}'.format(np.average(measures['diameter']), np.std(measures['diameter'])))
        print('For multigraph Cnet is measured after transforming the graph into a simple graph')
        print('Multigraph: 1 - Yes, 0 - No     {:.04f}  +/-  {:.04f}'.format(np.average(measures['multigraph']), np.std(measures['multigraph'])))
        print('Cnet:                           {:.04f}  +/-  {:.04f}'.format(np.average(measures['Cnet']), np.std(measures['Cnet'])))
        print('<C>:                            {:.04f}  +/-  {:.04f}'.format(np.average(measures['average_clustering']), np.std(measures['average_clustering'])))    
        
###########################################################################
# Plot measures
###########################################################################
def plotMeasures(measures, measure_names=None, x_range=1, show=True, relaxation_time=0):
    
    if measure_names ==None:
        measure_names=[k for k in measures]
    
    if type(measure_names)!=list:
        print('Please provide a *list* of measure names, exiting...')
        return
    
    if len(measure_names)==0:
        print('No measures to print, exiting...')
        return
    
    print('measures available: ', [k for k in measures])
    if not all(item in [k for k in measures] for item in measure_names):
        print('Measure names requested do no exist, exiting...')
        return
    
    print('measures to print:', measure_names)
    
    if (type(measures['N'])!=list):
        print('No list of measures to print in measures object, exiting...')
        return
        
    tmax = len(measures['N'])
    
    print('number of samples:', tmax)
    
    number_of_plots=len(measure_names)
    fig, axs = plt.subplots(number_of_plots, 1, figsize=(9, 6*number_of_plots))
                
    for i, m_name in enumerate(measure_names):
        plt.grid(linestyle='-', linewidth=0.5)
        avg_measure=np.average(measures[m_name][2:])
        sigma_measure=np.std(measures[m_name][2:])
        print('{} : {:.04f} +/- {:.04f} '.format(m_name, avg_measure, sigma_measure))
        if len(measure_names)==1:
            axs.grid(linestyle='-', linewidth=0.5)
            axs.plot([x for x in range(0, tmax*x_range, x_range)], [measures[m_name][t] for t in range(tmax)] , label=m_name)
            axs.set_title(m_name)
            axs.set_xlabel('t')
            if relaxation_time != 0:
                axs.axvline(x = relaxation_time, color = 'g', label = 'relaxation time', linestyle='dashed')
            axs.axhline(y=avg_measure, color='b', label='avg', linestyle='dotted')  
            axs.axhline(y=(avg_measure+2*sigma_measure), color='cyan', label='avg+2sigma', linestyle='dotted')  
            axs.axhline(y=(avg_measure-2*sigma_measure), color='cyan', label='avg-2sigma', linestyle='dotted')  
        else:
            axs[i].grid(linestyle='-', linewidth=0.5)
            axs[i].plot([x for x in range(0, tmax*x_range, x_range)], [measures[m_name][t] for t in range(tmax)] , label=m_name)
            axs[i].set_title(m_name)
            axs[i].set_xlabel('t')
            if relaxation_time != 0:
                axs[i].axvline(x = relaxation_time, color = 'g', label = 'relaxation time', linestyle='dashed')
            axs[i].axhline(y=avg_measure, color='b', label='avg', linestyle='dotted')
            axs[i].axhline(y=(avg_measure+2*sigma_measure), color='cyan', label='avg+2sigma', linestyle='dotted')  
            axs[i].axhline(y=(avg_measure-2*sigma_measure), color='cyan', label='avg-2sigma', linestyle='dotted')
        
    if show is True:
        plt.show()
    return fig, axs

def plotRelaxationTime(axs, measures, measure_names, relaxation_time, x_range, show=True):
    for i, m_name in enumerate(measure_names):
        
        # we are taking a sample every x_range time therefore the first sample after relaxation time is
        #first_sample=int(round(relaxation_time/x_range))+1
        avg_measure=np.average(measures[m_name][1:])
        sigma_measure=np.std(measures[m_name][1:])
        
        print('Measure name: ', m_name, end=', ')
        print('{:.04f}  +/-  {:.04f}'.format(avg_measure, sigma_measure) )

        if len(measure_names)==1:
            axs.grid(linestyle='-', linewidth=0.5)
            axs.axvline(x = relaxation_time, color = 'g', label = 'relaxation time', linestyle='dashed')
            axs.axhline(y=avg_measure, color='b', label='avg', linestyle='dotted')  
            axs.axhline(y=(avg_measure+2*sigma_measure), color='cyan', label='avg+2sigma', linestyle='dotted')  
            axs.axhline(y=(avg_measure-2*sigma_measure), color='cyan', label='avg-2sigma', linestyle='dotted')  
        else:
            axs[i].grid(linestyle='-', linewidth=0.5)
            axs[i].axvline(x = relaxation_time, color = 'g', label = 'relaxation time', linestyle='dashed')
            axs[i].axhline(y=avg_measure, color='b', label='avg', linestyle='dotted')
            axs[i].axhline(y=(avg_measure+2*sigma_measure), color='cyan', label='avg+2sigma', linestyle='dotted')  
            axs[i].axhline(y=(avg_measure-2*sigma_measure), color='cyan', label='avg-2sigma', linestyle='dotted')
    
    if show is True:
        plt.show()
    return axs
        