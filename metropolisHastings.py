from cmath import nan
import copy
import networkx as nx
import numpy as np
import random
import pandas as pd
import re
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from measuresFunctions import getMeasures, printSamplesMeasuresMeanAndStd, printMeasures, plotMeasures
from pickleUtil import pickleLoad, pickleSave

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from globalParams import DATAFOLDER
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def Acceptance(g, gnext, measure_fn, **parameters):
    '''
    Determine the probability of acceptance of the new proposed network gnext
    When there is only one measure as a constraint
    The measure function takes as input the network, returns a real number
    E.g. of measures nx.transitivity to measure the clustering coefficient
    '''
    # acceptance = 1 if P(g')/P(g) > 1 or np.exp(beta*(Cg' - Cg)) > 1 or beta*(Cg' - Cg) >0 
    # acceptance = np.exp(beta*(Cg' - Cg)) otherwise
    
    Xg = measure_fn(g)
    Xgnext = measure_fn(gnext)
    beta = parameters['beta']
    
    if (beta * (Xgnext - Xg)) >= 0:
        accept = 1
    else: 
        accept=np.exp(beta*(Xgnext - Xg))
      
    return accept
   

def MetropolisHasting(g_orig, T, number_of_samples, thinning, max_propositions, constraint_measure_fn=nx.transitivity, sample_measure_fn=getMeasures, sample_sigma_name=None, **parameters):
    #g  : graph to sample
    #P  : function to calculate P(g) 
    #T  : transformation T from g to g'
    #number_of_iter : = number of samples
    #thinning : number of iterations between each sample (relaxation time)

    # doing all the transformation on a deepcopy of the original network
           
    g = copy.deepcopy(g_orig)
    
    samples=None

    samples_t=[0 for x in range(number_of_samples)]
    samples_t[0]=0
    rejected=0
    accepted=0
        
    if sample_sigma_name == None:
        sample_sigma_name=constraint_measure_fn.__name__

    if max_propositions < thinning:
        print('max_propositions should be much larger than thinning. \
            Thinning is the number of accepted swaps before taking a sample. \
                Max_propositions is a limit to the number of transformation we do before we taking a sample (in case we never get many accepted)')
    
    t=0
    for i in range(number_of_samples):

        accepted_swaps =0
        total_propositions=0
        while accepted_swaps < thinning and total_propositions < max_propositions:
            
            gnext = T(g, inPlace=False)
            total_propositions+=1

            Acc=Acceptance(g, gnext, constraint_measure_fn, **parameters)

            if Acc == 1:       
                # when the next graph is more probable take it!
                g=gnext
#                 measures=getMeasures(g, measureList=True, measures=measures)
#                 measures_t=t
                accepted+=1
                accepted_swaps+=1
            else:
                # Accept move with probability Acc

                r=random.uniform(0, 1)
                if r < Acc:
                    # accept transformation as well
                    g=gnext
                    accepted+=1
                    accepted_swaps+=1
                else:
                    rejected+=1    
                # keep the measure for either case (a new gnext, or a repeat of g)
                # measures=getMeasures(g, measureList=True, measures=measures)
                # measures_t=t
            t+=1
                    
        # Now save a sample , after a number (=thinning) of accepted transitions
        # save the last graph measures as sample
        samples=sample_measure_fn(g, measureList=True, measures=samples)
        samples_t=t
        print(f"Sample taken at time {t} with {constraint_measure_fn.__name__} =  {samples[sample_sigma_name][-1]:.04f} after {accepted_swaps} accepted swaps (target accepted swaps before sampling = {thinning}).")


    print('# Rejected:', rejected)
    print('# Accepted:', accepted)
    if accepted > 0:
        print('Proportion rejected:', rejected/(accepted+rejected))
                
    # After doing all the iterations return
    return { 'samples': samples, 'samples_t': samples_t, 'lastnet': g, 'rejections': rejected/(accepted+rejected) }

def iterMHBeta(Gstart, T, number_of_samples, betas, relaxation_time, constraint_measure_fn, picklename, sample_measure_fn=getMeasures, max_propositions=0, burnin=5000, sample_sigma_name=None):
    
# Example parameters:
#     number_of_iter=20
#     relaxation time with swap edge = L/2
#     beta = [-500, -300, -100, 1, 100, 200, 225,  250, 300, 350, 400, 600, 800]


    thinning=int(round(relaxation_time)+1)

    print('Number of samples requested: ', number_of_samples)
    print('Number of accepted swaps between samples:', thinning)
    print('{} burning iterations at the start before taking any samples'.format(burnin))

    result_beta=[None for i in range(len(betas))]

    i=0
    G=copy.deepcopy(Gstart)
    
    # burnin iterations
    
    b=betas[0]
    parameters={'beta':b}
    result_burnin=MetropolisHasting(G, T, number_of_samples=1, thinning=burnin, max_propositions=max_propositions, constraint_measure_fn=constraint_measure_fn, sample_measure_fn=sample_measure_fn, sample_sigma_name=sample_sigma_name, **parameters)
    G=result_burnin['lastnet']
    pickleSave(result_burnin, picklename + '_burnin_'+str(b), DATAFOLDER )
    
    # taking samples iterations (number_samples for each beta)
    bprev=betas[0]
    for b in betas:
        print('--------------------------------------------------------------')
        print('                    Beta = ', b)
        print('--------------------------------------------------------------')
        parameters={'beta':b}
        result_beta[i]=MetropolisHasting(G, T, number_of_samples, thinning, max_propositions, constraint_measure_fn, sample_measure_fn=sample_measure_fn, sample_sigma_name=sample_sigma_name, **parameters)
        printSamplesMeasuresMeanAndStd(result_beta[i]['samples'], sample_measure_fn=sample_measure_fn)
        G=result_beta[i]['lastnet']
        
        # saving pickle for each beta
        if bprev <= b:
            updown = 'up'
        else:
            updown='down'
        #pickleSave(result_beta[i], 'r_' + picklename + '_'+ updown + '_beta_'+str(b), DATAFOLDER )
        pickleSave(result_beta[i], picklename + '_'+ updown + '_beta_'+str(b), DATAFOLDER )
        
        # setting up next iter
        i+=1
        bprev=b
        
    return result_beta

def plotMetropolisHastingsResult(result, measurename, betas, graph=None, col='purple', label=None, errorbar=True, title=None):
    
    # betas contains the list of betas
    # for each betas we have taken a bucket of samples
    # each "r" {dictionary object} in result is a dictionary that contains all the info for the corresponding beta and its bucket of samples
    # in that "r" set , there is the 'samples' , it contains all the measures taken on each sample as dictionary keys
    
    M = [np.average(r['samples'][measurename]) for r in result]
    M_err = [np.std(r['samples'][measurename]) for r in result]
    
    # M and M_err contains the mean and standard deviation for that measurename for each beta

    
    if label==None:
        label=measurename

    if graph==None:
        fig, axs = plt.subplots(1, 1, figsize=(9, 6))
    else:
        fig = graph[0]
        axs = graph[1]
        
    plt.grid(linestyle='-', linewidth=0.5)
    if title==None:
        axs.set_title(measurename + r" vs $\beta$")
    else:
        axs.set_title(title)
    axs.set_xlabel(r"$\beta$")
    axs.set_ylabel(measurename)
    axs.legend()
    #axs.axhline(y=C_star, color='b', label='C*', linestyle='dotted')

    if (errorbar==True):
        plt.errorbar(betas, M, M_err, marker='.', ls='dotted', color=col, label=label)
    else:
        plt.plot(betas,M,label=label, color=col)
#     for i,j in zip(beta,M):
#         axs.annotate(str(i),xy=(i -50,j+0.02))

#-----------------------------
# PLOT FROM LIST OF PICKLE FILES
#-----------------------------

#     E.G. PICKLEROOT='r_Gcatu_TSE_'

def loadFromPickle(pickleroot, measurenames=[], gml=False, errorbar=True, title=None, figsize=None):

    # name like r_Gcatu_TSE_up_beta_3000.pkl
    baseUri=Path(DATAFOLDER)

    pklFiles=[x for x in baseUri.glob('**/' + pickleroot + '*.pkl')]

    patternBeta = r"beta_(-?[0-9]*)\.pkl"
    patternUp = r"_up_"

    dataUp={ 'betas': [], 'up': [], 'results': [], 'filepaths':[]}
    dataDown={ 'betas': [], 'up': [], 'results': [],'filepaths':[]}

    for i, fn in enumerate(pklFiles):

        beta=int(re.search(patternBeta, fn.name).group(1))

        up=False
        if re.search(patternUp, fn.name)!= None:
            up=True
        else:
            up=False


        #loading pickle file
        result=pickleLoad(str(fn.stem), str(fn.parents[0]), silent=True)

        #saving as gml
        if gml==True:
            nx.write_gml(result['lastnet'], str(fn.stem) + '.gml')

        if (up):
            dataUp['betas'].append(beta)
            dataUp['up'].append(up)
            dataUp['results'].append(result)
            dataUp['filepaths'].append(fn)
        else:
            dataDown['betas'].append(beta)
            dataDown['up'].append(up)
            dataDown['results'].append(result)
            dataDown['filepaths'].append(fn)


    df = pd.DataFrame(dataUp)
    df = df.sort_values(by='betas')

    dfd = pd.DataFrame(dataDown)
    dfd = dfd.sort_values(by='betas')
    
    if figsize==None:
         figsize=(9,6)
    else:
         figsize=figsize
    
    graphs=[]
    for measurename in measurenames:

        fig, axs = plt.subplots(1,1, figsize=figsize)
        plotMetropolisHastingsResult(df['results'], measurename, df['betas'], graph=(fig, axs), col='tab:blue', label=r"increasing $\beta$", errorbar=errorbar, title=title)
        plotMetropolisHastingsResult(dfd['results'], measurename, dfd['betas'], graph=(fig, axs), col='tab:orange', label=r"decreasing $\beta$", errorbar=errorbar, title=title)
        plt.legend()
        graphs.append((fig, axs))
        #fig.savefig("{}-{:03}.png".format(title, np.random.randint(1000000)))
        
    return {'up': df, 'down': dfd, 'figax': graphs}


'''
#############################################################################################
LOAD SAMPLE FROM PICKLE FILES
-------------------------------
Reads all the pickle files start with <experiment_name> in the folder <data_folder>
Assembles them into a dataframe containing all samples from the experiment for each beta
Columns of the dataframe are :
beta : parameter beta
up : obtained while going up in Beta or down (True or False)
g : sample graph
time : iteration number, used to order samples along with beta
file : pickle file name on <datafolder>
experiment : experiment name
comment : to determine if this is a startnet, or a burnin sample
#############################################################################################
'''
from networkSigma import discreteSigma2Analytical, continuousSigma2Analytical

def getTransformation(filename):
    patternTReconnectDestinationOfEdgeToOtherNode = r"TReconnectDestinationOfEdgeToOtherNode"
    patternTReconnectOriginOfEdgeToOtherNode = r"TReconnectOriginOfEdgeToOtherNode"
    patternTDeleteEdgeAddEdge = r"TDeleteEdgeAddEdge"
    patternTDeleteEdgeAddEdgeWeightedColumnStochastic = r"TDeleteEdgeAddEdgeWeightedColumnStochastic"
    patternTReconnectOriginOfEdgeToOtherNodeNoSelfLoops = r"TReconnectOriginOfEdgeToOtherNodeNoSelfLoops"
    patternTSwapEdgesDirected = r"TSwapEdgesDirected"
    
    transformation='undetermined'
    if re.search(patternTReconnectDestinationOfEdgeToOtherNode, filename)!= None:
        transformation = 'TReconnectDestinationOfEdgeToOtherNode'
    if re.search(patternTReconnectOriginOfEdgeToOtherNode, filename)!= None:
        transformation = 'TReconnectOriginOfEdgeToOtherNode'
    if re.search(patternTDeleteEdgeAddEdge, filename)!= None:
        transformation = 'TDeleteEdgeAddEdge'
    if re.search(patternTDeleteEdgeAddEdgeWeightedColumnStochastic, filename)!= None:
        transformation = 'TDeleteEdgeAddEdgeWeightedColumnStochastic'
    if re.search(patternTReconnectOriginOfEdgeToOtherNodeNoSelfLoops, filename)!= None:
        transformation = 'TReconnectOriginOfEdgeToOtherNodeNoSelfLoops'
    if re.search(patternTSwapEdgesDirected, filename)!= None:
        transformation = 'TSwapEdgesDirected'
        
    return transformation

def getStartingNetwork(filename):
    patternRandom = r"ER-100-(p0.[0-9]*)-"
    patternRandom2 = r"FixIn-100-ER(0.[0-9]*)"
    patternFixIn = r"FixIn-100-DegIn([2-9])-"
    patternFixIn2 = r"FixIn-100-FixInDegreeSequence([2-9])-"
    patternRegular = r"-Regular([1-9])-"
    
    startingNetwork = 'undetermined'
    if re.search(patternRandom, filename)!= None:
        p = re.search(patternRandom, filename)[1]
        print('... random prob:', p)
        startingNetwork = f'ErdosRenyi-{p}'
    if re.search(patternRandom2, filename)!= None:
        p = re.search(patternRandom2, filename)[1]
        print('... random prob:', p)
        startingNetwork = f'ErdosRenyi-p{p}'
    if re.search(patternFixIn, filename)!= None:
        degIn = re.search(patternFixIn, filename)[1]
        print('... fixed incoming degree:', degIn)
        startingNetwork = f'Fixed Incoming degree-{degIn}'
    if re.search(patternFixIn2, filename)!= None:
        degIn = re.search(patternFixIn2, filename)[1]
        print('... fixed incoming degree:', degIn)
        startingNetwork = f'Fixed Incoming degree-{degIn}'
    if re.search(patternRegular, filename)!= None:
        deg = re.search(patternRegular, filename)[1]
        print('... regular with degree:', deg)
        startingNetwork = f'Regular-{deg}'
        
    return startingNetwork

def getDiscreteOrContinuous(filename):
    
    patternContinuous=r"[c,C]ontinuous"
    patternDiscrete=r"[d,D]iscrete"
    discreteOrContinuous = 'undetermined'
    if re.search(patternContinuous, filename)!= None:
        discreteOrContinuous = 'continuous'
    if re.search(patternDiscrete, filename)!= None:
        discreteOrContinuous = 'discrete'
    
    return discreteOrContinuous

def getFixedWeightsOrNot(g):
    weights = nx.get_edge_attributes(g, 'weight').values()
    wm = np.array(list(weights)).mean()
    wstd = np.array(list(weights)).std()
    if wstd == 0:
        return f'Fixed {wm:.04f}'
    else:
        return f'Varying {wm:.04f}+-{wstd:.04f}'

def loadSamplesFromPickle(experiment_name, datafolder='./data'):

    # name like r_Gcatu_TSE_up_beta_3000.pkl
    baseUri=Path(datafolder)

    pklFiles=[x for x in baseUri.glob('**/' + experiment_name + '*.pkl')]

    if pklFiles==[]:
        print(f'No files found here {datafolder}/{experiment_name}*.pkl')
        return

    patternBeta = r"beta_(-?[0-9]*)\.pkl"
    patternUp = r"_up_"
    patternStartNet = r"StartNet"
    patternBurnin = r"_burnin_(-?[0-9]*)\.pkl"
    patternMin = r"-MIN-"
    patternMax = r"-MAX-"
    patternRandomWeights = r"-[r,R]andom[w,W]"
    patternFixedWeights = r"-[f, F]ixed[w,W]"

    data  ={ 'beta': [], 'up': [], 'g': [], 'time':[], 'file':[], \
        'experiment':[], 'transformation':[], 'startingNetwork':[], 'comment':[], 'fixedOrVaryingWeights': [], \
            'discreteSigma2Analytical': [], 'continuousSigma2Analytical': [], 'minOrMax': [], 'optimizationBasedOn': [], 'discreteOrContinuous': []}

    for i, fn in enumerate(pklFiles):
        
        print(f'Processing file {i}, {fn}')
        
        minOrMax = 'MIN-MAX'
        if re.search(patternMin, fn.name)!= None:
            print('... MIN file')
            minOrMax = 'MIN'
        else:
            if re.search(patternMax, fn.name)!= None:
                print('... MAX file')
                minOrMax = 'MAX'
        
        # Saving initial network 
        if re.search(patternStartNet, fn.name)!= None:
            print('... StartNet file')
            result=pickleLoad(str(fn.stem), str(fn.parents[0]), silent=False)
            data['beta'].append(nan)
            up = True
            if minOrMax == 'MIN':
                up=False
            data['up'].append(up)
            data['g'].append(result)
            data['time'].append(0)
            data['file'].append(fn)
            data['experiment'].append(fn.name.rstrip('.pkl'))
            data['startingNetwork'].append(getStartingNetwork(fn.name))
            data['transformation'].append(getTransformation(fn.name))
            data['comment'].append('startNet')  
            data['discreteSigma2Analytical'].append(discreteSigma2Analytical(result))
            data['continuousSigma2Analytical'].append(continuousSigma2Analytical(result))
            data['minOrMax'].append(minOrMax)
            data['optimizationBasedOn'].append(constraint_name)
            data['discreteOrContinuous'].append(getDiscreteOrContinuous(fn.name))
            data['fixedOrVaryingWeights'].append(getFixedWeightsOrNot(result))
            continue
        
        comment=''
        if re.search(patternBurnin, fn.name)!=None:
        # Reading burning results
            print('... burnin file')
            beta = int(re.search(patternBurnin, fn.name).group(1)) 
            up = True
            if minOrMax == 'MIN':
                up=False
            comment='burnin'
        else:
        # Reading standard result
            beta=int(re.search(patternBeta, fn.name).group(1))
            up=False
            if re.search(patternUp, fn.name)!= None:
                up=True
            else:
                up=False

        #loading pickle file
        result=pickleLoad(str(fn.stem), str(fn.parents[0]), silent=True)
        
        constraint_name='undetermined'
        if 'discreteSigma2Analytical' in result['samples']:
            constraint_name = 'discreteSigma2Analytical'
        if 'continuousSigma2Analytical' in result['samples']:
            constraint_name = 'continuousSigma2Analytical'

        if 'g' in result['samples']:
            for i, (sample, constraint_measure) in enumerate(zip(result['samples']['g'], result['samples'][constraint_name])):

                data['beta'].append(beta)
                data['up'].append(up)
                data['g'].append(sample)
                if 'continuousSigma2Analytical' in result['samples']:
                    data['continuousSigma2Analytical'].append(constraint_measure)
                    data['discreteSigma2Analytical'].append(discreteSigma2Analytical(sample))
                    #data['discreteSigma2Analytical'].append(None)
                if 'discreteSigma2Analytical' in result['samples']:
                    data['discreteSigma2Analytical'].append(constraint_measure)
                    data['continuousSigma2Analytical'].append(continuousSigma2Analytical(sample))
                    #data['continuousSigma2Analytical'].append(None)
                data['time'].append(i)
                data['file'].append(fn)
                data['experiment'].append(fn.name.rstrip('.pkl'))
                data['startingNetwork'].append(getStartingNetwork(fn.name))
                data['transformation'].append(getTransformation(fn.name))
                data['comment'].append(comment)
                data['minOrMax'].append(minOrMax)
                data['optimizationBasedOn'].append(constraint_name)
                data['discreteOrContinuous'].append(getDiscreteOrContinuous(fn.name))
                data['fixedOrVaryingWeights'].append(getFixedWeightsOrNot(sample))
        else:
            for i, constraint_measure in enumerate(result['samples'][constraint_name]):
                data['beta'].append(beta)
                data['up'].append(up)
                data['g'].append(None)
                if 'continuousSigma2Analytical' in result['samples']:
                    data['continuousSigma2Analytical'].append(constraint_measure)
                    data['discreteSigma2Analytical'].append(discreteSigma2Analytical(sample))
                    #data['discreteSigma2Analytical'].append(None)
                if 'discreteSigma2Analytical' in result['samples']:
                    data['discreteSigma2Analytical'].append(constraint_measure)
                    data['continuousSigma2Analytical'].append(continuousSigma2Analytical(sample))
                    #data['continuousSigma2Analytical'].append(None)
                data['time'].append(i)
                data['file'].append(fn)
                data['experiment'].append(fn.name.rstrip('.pkl'))
                data['startingNetwork'].append(getStartingNetwork(fn.name))
                data['transformation'].append(getTransformation(fn.name))
                data['comment'].append(comment)
                data['minOrMax'].append(minOrMax)
                data['optimizationBasedOn'].append(constraint_name)
                data['discreteOrContinuous'].append(getDiscreteOrContinuous(fn.name))
                fixedOrNot = 'undetermined'
                if re.search(patternRandomWeights, fn.name)!= None:
                    fixedOrNot = 'Varying'
                if re.search(patternFixedWeights, fn.name)!= None:
                    fixedOrNot = 'Fixed'  
                data['fixedOrVaryingWeights'].append(fixedOrNot)

    df = pd.DataFrame.from_dict(data)
    df = df.sort_values(by=['beta', 'time'])
            
    return df

applyMeasureToGraph = lambda df, measure_fn : df.g.apply(lambda g : measure_fn(g))

def analyzeMetropolisHastingsGraphs(df, measure_fn, plot=True, ascending=True):
    '''
    #############################################################################################
    LOAD SAMPLE FROM PICKLE FILES
    -------------------------------

    To be used after loading samples from pickle into a dataframe with loadSamplesFromPickle

    Parameters:
        df : dataframe with at least the following columns ' :
            'g' to contain the graph
            'beta': beta value at which this graph was produced
            
        measure_fn : function that takes only the graph as parameter and returns 
        
    Returns df with an extra column, the measure name that was requested

    #############################################################################################
    '''
    
    # df : pandas.Dataframe that contains a sequence of graphs and their associated betas
    
    measure_name = measure_fn.__name__
    
    def getMeasureFor(g):
        if g!= None:
            return measure_fn(g)
        else:
            return nan
    
    if measure_name not in df.keys():
        df[measure_name]=df.g.apply(lambda g : getMeasureFor(g))
    
    M = df[[measure_name, 'beta']].dropna(axis=0).groupby(['beta'], as_index=False).mean().sort_values(by='beta', ascending=ascending)
    M_err = df[[measure_name, 'beta']].dropna(axis=0).groupby(['beta'], as_index=False).std().sort_values(by='beta', ascending=ascending)
    
    # M and M_err contains the mean and standard deviation for that measure_name for each beta
       
    df_avg = pd.concat( [M['beta'],M[measure_name], M_err[measure_name] ], keys=['beta', f'{measure_name}_mean', f'{measure_name}_std'], axis=1)

    if plot==True:
        label=measure_name

        fig, axs = plt.subplots(1, 1, figsize=(9, 6))
            
        plt.grid(linestyle='-', linewidth=0.5)

        axs.set_title(f'{measure_name} vs beta')
        axs.set_xlabel('beta')
        axs.set_ylabel(measure_name)
        axs.legend()

        plt.errorbar(M['beta'], M[measure_name], M_err[measure_name], marker='.', ls='dotted', color='purple', label=label)

    
    return df, df_avg


