import numpy as np
from sklearn.metrics import accuracy_score
from itertools import combinations
import pandas as pd 


def get_neurons_by_categories(neurons_atframes, stim_idx, cats, selected_categories, exemplars_per_cat=4):
    """
    Returns the neuron response for each exemplar of the selected categories

    Parameters:
    -----------

    selected_categories: tuple
    
    tuple of the selected categories
    """

    selected_neurons = np.zeros((exemplars_per_cat*len(selected_categories), neurons_atframes.shape[0], stim_idx.shape[1]))
    id_exemplar = 0 
    for category in selected_categories:
        cat = cats[category]
        for exemplar in range(exemplars_per_cat):
            selected_neurons[id_exemplar] = neurons_atframes[:, stim_idx[cat+exemplar]]
            id_exemplar+=1
    return selected_neurons

def get_only_neurons_with_var(neurons_at_cats,exemplars_per_cat=4, verbose = False):

    """
    Elimininate neurons that dont have any variance for a given category

    Parameters:
    -----------

    neurons_at_cats: array (exemplars,neurons)
        neuronal responses for each exemplar

    At grabing the number of exemplars per category we filter for neurons without variance in a category

    Returns:
    --------

    Filtered population and idx of neurons with var for those categories

    """
    var_over_reps = neurons_at_cats.std(-1)
    cat1_no_var = np.unique(np.where(var_over_reps[:exemplars_per_cat]==0)[1])
    cat2_no_var = np.unique(np.where(var_over_reps[exemplars_per_cat:]==0)[1])
    neurons_wo_variance = np.concatenate([cat1_no_var,cat2_no_var]) 
    neurons_with_var_ = [neuron for neuron in range(neurons_at_cats.shape[1]) if neuron not in neurons_wo_variance]
    if verbose == True:
        print(f"Neurons without variance:")
        print("-------------------------")
        print(f"category 1: {len(cat1_no_var)}")
        print(f"category 2: {len(cat2_no_var)}")
    return neurons_at_cats[:,neurons_with_var_,:], neurons_with_var_

def PairwiseDprimeDecoder(neurons_atframes, stim_idx, iplane, cats, n_categories = 8, n_samples = 4, layer = 1 , avg_reps = False, method = 'dprime'):
    pairs = list(combinations(np.arange(n_categories), 2)) 
    train_textures = np.arange(n_samples)
    categories_dict = {
    "0":"leaves",
    "1":"circles",
    "2":"dryland",
    "3":"rocks",
    "4":"tiles",
    "5":"squares",
    "6":"round leaves",
    "7":"paved"}
    spops = np.empty((len(pairs),n_samples), dtype=object)
    below_tsh_neurons = np.empty((len(pairs),n_samples), dtype=object)
    above_tsh_neurons = np.empty((len(pairs),n_samples), dtype=object)
    accurracy = []
    for ix_pair, pair in enumerate(pairs):
        #print(f"******************category pair: {categories_dict[str(pair[0])]}/{categories_dict[str(pair[1])]}***********************")
        neurons_at_cats = get_neurons_by_categories(neurons_atframes, stim_idx, cats, selected_categories = pair, exemplars_per_cat=n_samples)
        Xtrain = neurons_at_cats[:,:,::2]
        Xtest = neurons_at_cats[:,:,1::2]
        Xtrain_with_var, idx_neurons_with_var_ = get_only_neurons_with_var(Xtrain,exemplars_per_cat=n_samples) # Only even repeats for training
        Xtest_with_var = Xtest[:,idx_neurons_with_var_,:]
        Xtrain_varneurons_iplane = iplane[idx_neurons_with_var_]
        mu_ = Xtrain_with_var.mean(-1)
        sd_ = Xtrain_with_var.std(-1)
        mean_diff = (mu_[:n_samples] - mu_[n_samples:])
        avg_sd = (sd_[:n_samples]+sd_[n_samples:])/2
        dprime_ = mean_diff/avg_sd
        #print("*************TRAINING**************")
        for train_texture in train_textures:
            if method == 'dprime':
                tsh = np.percentile(dprime_[train_texture], 99)
                if layer == 2: 
                    neurons_abvtresh_ = (dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane < 10) #Dprime threshold
                    neurons_blwtresh_ = (-dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane < 10)
                elif layer == 1:
                    neurons_abvtresh_ = (dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane >= 10) #Dprime threshold
                    neurons_blwtresh_ = (-dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane >= 10)
                spop_ = Xtest_with_var[:, neurons_abvtresh_, :].mean(1) - Xtest_with_var[:,neurons_blwtresh_, :].mean(1)
                #print(f"neurons abv: {(neurons_abvtresh_).sum()} & blw: {(neurons_blwtresh_).sum()} thresh")
            elif method == 'top':
                trained_dprime = dprime_[train_texture]
                n = int(np.ceil(len(trained_dprime)*0.30)) 
                if layer == 2: 
                    neurons_abvtresh_ = trained_dprime[Xtrain_varneurons_iplane < 10].argsort()[::-1][:n]
                    neurons_blwtresh_ = trained_dprime[Xtrain_varneurons_iplane < 10].argsort()[:n]
                elif layer == 1:
                    neurons_abvtresh_ = trained_dprime[Xtrain_varneurons_iplane >= 10].argsort()[::-1][:n]
                    neurons_blwtresh_ = trained_dprime[Xtrain_varneurons_iplane >= 10].argsort()[:n]
                spop_ = Xtest_with_var[:, neurons_abvtresh_, :].mean(1) - Xtest_with_var[:,neurons_blwtresh_, :].mean(1)
            else:
                raise ValueError("Method not implemented")
            #scoring
            if avg_reps:
                spop_avg = spop_.mean(-1)
                s = spop_avg.reshape(2,-1)
            else:
                s = spop_.reshape(2,-1)
            spops[ix_pair,train_texture] = s
            above_tsh_neurons[ix_pair,train_texture] = neurons_abvtresh_
            below_tsh_neurons[ix_pair,train_texture] = neurons_blwtresh_
            pred = s[0]>s[1]
            y = np.ones(s.shape[1])
            score_ = accuracy_score(y, pred) * 100
            #print(f"training pair: {train_texture}, accuracy: {score_}")
            accurracy.append(score_)
            
    accurracy = np.array(accurracy).reshape(len(pairs),n_samples)
    return accurracy, pairs, spops, categories_dict, above_tsh_neurons, below_tsh_neurons

def get_stim_class_and_samples_ix(subset_stim, n_categories=8, samples_per_cat=4, nrep=None):
    """
    Gets the idx for each repeat of each exemplar, for the specified categories.

    stim_idx is a vector containing the idx of each repeat of each exemplar stimuli (exemplar, repeats)
    cats_idx is a vector with the idx that starts a new category
    """
    total_samples = n_categories * samples_per_cat
    _, nc = np.unique(subset_stim, return_counts = True)
    nc = nc[:total_samples]
    nreps = np.min(nc)
    if nrep is not None:
        assert nreps>nrep, "more specified reps than effective reps"
        nreps = nrep
    for exemplar in range(total_samples):
        if exemplar == 0:
            stim_idx = np.expand_dims(np.where(subset_stim==exemplar+1)[0][:nreps],axis=0)
        else:
            stim_idx= np.append(stim_idx, np.expand_dims(np.where(subset_stim==exemplar+1)[0][:nreps],axis=0),axis=0)
    cats_idx = np.arange(0, total_samples, samples_per_cat)
    #print(f"{cats_idx.shape[0]} categories, {stim_idx.shape[0]} exemplars, {stim_idx.shape[1]} repeats")
    return cats_idx, stim_idx, nreps

def get_generalization_margings(spops, n_pairs = 28, n_textures = 4):
    """
    Gets the generalization margins for each posible training texture, only works for spop averaged over reapeats.
    """
    generalization_margings = np.empty((n_pairs,n_textures), dtype=object)
    margins_matrix = np.empty((n_pairs,n_textures,n_textures), dtype=object)
    for pair in range(n_pairs):
        for train_tex in range(n_textures):
            margins = spops[pair,train_tex][0] - spops[pair,train_tex][1]
            margins = np.round(margins,4)
            train_marging = margins[train_tex]
            if train_marging == 0:
                margin_ratio = np.zeros(4)
                margin_ratio[train_tex] = 1
            elif train_marging < 0:
                print(f"train tex: {train_tex}, in cate pair {pair} contains negative margin")
                margin_ratio = margins/train_marging
                margin_ratio = np.where(margin_ratio<0,margin_ratio,margin_ratio*-1)
                margin_ratio[train_tex] = 1
            else:
                margin_ratio = margins/train_marging
                margin_ratio[train_tex] = 1
            generalization_margings[pair,train_tex] = margin_ratio
            margins_matrix[pair,train_tex] = margins
    return generalization_margings, margins_matrix

def get_margin_per_category(margins,category_pair):
    """
    Retrieves the margins for a given category_pair, or if avg=True it returns the avg marging for each training texture.
    """
    margins_atcatpair = np.concatenate(margins[category_pair,:],axis=0).reshape(4,4)
    margins_atcatpair[margins_atcatpair==1] = np.nan
    mean_margins_atcatpair = np.nanmean(margins_atcatpair,axis=1)
    std_margins_atcatpair = np.nanstd(margins_atcatpair,axis=1)
    return mean_margins_atcatpair, std_margins_atcatpair

def get_neurons_atframes(timeline, spks):
    """
    Get the neurons at each frame, and the subset of stimulus before the recording ends.

    Parameters
    ----------
    spks : array
        Spikes of the neurons.
    Timeline : array
        Timeline of the experiment.

    Returns
    -------
    neurons_atframes : array
        Neurons at each frame.
    subset_stim: array
        Stimuli before recording ends
    """
    _, nt = spks.shape
    tlag = 1  # this is the normal lag between frames and stimuli
    istim = timeline["stiminfo"].item()["istim"]
    frame_start = timeline["stiminfo"].item()["frame_start"]
    frame_start = np.array(frame_start).astype("int")
    frame_start0 = frame_start + tlag
    ix = frame_start0 < nt
    frame_start0 = frame_start0[ix]
    neurons_atframes = spks[
        :, frame_start0
    ]  # sample the neurons at the stimulus frames
    subset_stim = istim[ix]
    return neurons_atframes, subset_stim

def signal_variance(sstim, S):
    NN = S.shape[0]
    istim = np.unique(sstim)
    k = 0
    s2 = np.zeros((2, len(istim), NN), 'float32')
    for j in range(len(istim)):
        ix = np.nonzero(sstim==istim[j])[0]
        if len(ix)==2:
            s2[0, k] = S[:, ix[0]]
            s2[1, k] = S[:, ix[1]]
            k +=1
    s2 = s2[:,:k]


    ss = s2 - np.mean(s2,1)[:,np.newaxis,:]
    ss = ss / np.mean(ss**2,1)[:,np.newaxis,:]**.5

    csig = (ss[0] * ss[1]).mean(0)
    print('signal variance is %2.2f'%csig.mean())
    if csig.mean()<0.05:
        print('The signal variance should be at least 0.05 for a normal window with most neurons in visual cortex.')

    return csig, ss

def categorypairs_parser(cat_dict,pairs):
    """
    Simple parser between category indexes and names.
    """
    category_pairs = []
    for pair in pairs:
        category_pairs.append(f"{cat_dict[str(pair[0])]}/{cat_dict[str(pair[1])]}")
    return category_pairs

def DprimeDecoder(neurons_atframes, stim_idx, iplane, cats, n_categories = 8, n_samples = 4, layers = [1,2], percentil = 95):
    from itertools import combinations
    from sklearn.metrics import accuracy_score
    pairs = list(combinations(np.arange(n_categories), 2)) 
    train_textures = np.arange(n_samples)
    categories_dict = {
    "0":"leaves",
    "1":"circles",
    "2":"dryland",
    "3":"rocks",
    "4":"tiles",
    "5":"squares",
    "6":"round leaves",
    "7":"paved"}
    n_layers = len(layers)
    spops = np.empty((len(pairs),n_samples,n_layers), dtype=object)
    below_tsh_neurons = np.empty((len(pairs),n_samples,n_layers), dtype=object)
    distances = np.empty((len(pairs),n_samples,n_layers), dtype=object)
    margins = np.empty((len(pairs),n_samples,n_layers), dtype=object)
    above_tsh_neurons = np.empty((len(pairs),n_samples,n_layers), dtype=object)
    accuracy = np.empty((len(pairs),n_samples,n_layers), dtype=object)
    #accurracy = []
    for ix_pair, pair in enumerate(pairs):
        #print(f"******************category pair: {categories_dict[str(pair[0])]}/{categories_dict[str(pair[1])]}***********************")
        neurons_at_cats = get_neurons_by_categories(neurons_atframes, stim_idx, cats, selected_categories = pair, exemplars_per_cat=n_samples)
        Xtrain = neurons_at_cats[:,:,::2]
        Xtest = neurons_at_cats[:,:,1::2]
        Xtrain_with_var, idx_neurons_with_var_ = get_only_neurons_with_var(Xtrain,exemplars_per_cat=n_samples) # Only even repeats for training
        Xtest_with_var = Xtest[:,idx_neurons_with_var_,:]
        Xtrain_varneurons_iplane = iplane[idx_neurons_with_var_]
        mu_ = Xtrain_with_var.mean(-1)
        sd_ = Xtrain_with_var.std(-1)
        mean_diff = (mu_[:n_samples] - mu_[n_samples:])
        avg_sd = (sd_[:n_samples]+sd_[n_samples:])/2
        dprime_ = mean_diff/avg_sd
        #print("*************TRAINING**************")
        for train_texture in train_textures:
            for layer in layers:
                tsh = np.percentile(dprime_[train_texture], percentil)
                if layer == 2: 
                    neurons_abvtresh_ = (dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane < 10) #Dprime threshold
                    neurons_blwtresh_ = (-dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane < 10)
                elif layer == 1:
                    neurons_abvtresh_ = (dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane >= 10) #Dprime threshold
                    neurons_blwtresh_ = (-dprime_[train_texture] > tsh) * (Xtrain_varneurons_iplane >= 10)
                spop_ = Xtest_with_var[:, neurons_abvtresh_, :].mean(1) - Xtest_with_var[:,neurons_blwtresh_, :].mean(1)
                    #print(f"neurons abv: {(neurons_abvtresh_).sum()} & blw: {(neurons_blwtresh_).sum()} thresh")
                #scoring
                spop_avg = spop_.mean(-1)
                s = spop_avg.reshape(2,-1)
                distance = s[0]-s[1]
                #distance = np.sqrt((s[0]**2)+(s[1]**2))
                ttex_mask = np.zeros((n_samples))
                ttex_mask[train_texture] = 1
                ttex_mask = ttex_mask.astype(bool)
                if distance[train_texture] < 0:
                    margin = distance/distance[train_texture]
                    margin[~ttex_mask] = 0
                    margin[train_texture] = 1
                    score_ = 0
                else:
                    margin = distance/distance[train_texture]
                    pred = s[0]>s[1]
                    pred = pred[~ttex_mask] #remove training texture
                    y = np.ones_like(pred)
                    score_ = accuracy_score(y, pred) * 100  
                spops[ix_pair,train_texture,layer-1] = s
                distances[ix_pair,train_texture,layer-1] = distance
                margins[ix_pair,train_texture,layer-1] = margin
                above_tsh_neurons[ix_pair,train_texture,layer-1] = neurons_abvtresh_
                below_tsh_neurons[ix_pair,train_texture,layer-1] = neurons_blwtresh_
                accuracy[ix_pair,train_texture,layer-1] = score_       
    return accuracy, pairs, spops, categories_dict, above_tsh_neurons, below_tsh_neurons, distances, margins

def overall_margins_stats(margins,pairs=28,layers=[1,2]):
    mean_margin_per_traininstance_per_layer = np.empty((pairs,len(layers)), dtype=object)
    std_margin_per_traininstance_per_layer = np.empty((pairs,len(layers)), dtype=object)
    ovrl_marginmean_percategory_layer = np.zeros((pairs,len(layers)))
    ovrl_marginstd_percategory_layer = np.zeros((pairs,len(layers)))
    for layer in layers:
        for pair in range(pairs):
            margins_category = np.concatenate(margins[pair,:,layer-1]).reshape(4,4)
            margins_category = np.where(margins_category==1,np.nan,margins_category)
            marginsstd_category = np.nanstd(margins_category,axis=1)
            margins_category = np.nanmean(margins_category,axis=1)
            mean_margin_per_traininstance_per_layer[pair,layer-1] = margins_category
            std_margin_per_traininstance_per_layer[pair,layer-1] = marginsstd_category
            ovrl_marginmean_percategory_layer[pair,layer-1] = margins_category.mean()
            ovrl_marginstd_percategory_layer[pair,layer-1] = margins_category.std()
    return mean_margin_per_traininstance_per_layer, std_margin_per_traininstance_per_layer, ovrl_marginmean_percategory_layer, ovrl_marginstd_percategory_layer

def summary_region(mice, N_PAIRS=28, SAMPLES_PER_CAT=4, LAYERS=[1,2]):
    summ_id = []
    summ_date = []
    summ_block = []
    summ_layer = []
    summ_traintex = []
    summ_acc = []
    summ_marginmean = []
    summ_catpair = []
    summ_marginstd = []
    summ_region = []
    for mouse in mice:
        for region in mouse.byregion.keys():
            accuracy = mouse.byregion[region]['accuracy']
            pairs = mouse.byregion[region]['pairs']
            mean_instance = mouse.byregion[region]['mean_instance']
            std_instance = mouse.byregion[region]['std_instance']
            cat_dict = mouse.byregion[region]['cat_dict']
            pairs = mouse.byregion[region]['pairs']
            cat_pairs = categorypairs_parser(cat_dict,pairs)
            summ_id.append([mouse.name] * N_PAIRS*SAMPLES_PER_CAT*len(LAYERS))
            summ_date.append([mouse.datexp] * N_PAIRS*SAMPLES_PER_CAT*len(LAYERS))
            summ_block.append([mouse.blk] * N_PAIRS*SAMPLES_PER_CAT*len(LAYERS))
            summ_region.append([region] * N_PAIRS*SAMPLES_PER_CAT*len(LAYERS))
            summ_layer.append(np.append(np.repeat(1,SAMPLES_PER_CAT*N_PAIRS),np.repeat(2,SAMPLES_PER_CAT*N_PAIRS)))
            summ_catpair.append(np.tile(np.repeat(np.array(cat_pairs),SAMPLES_PER_CAT),len(LAYERS)))
            summ_traintex.append(np.tile(np.tile(np.arange(SAMPLES_PER_CAT),N_PAIRS),len(LAYERS)))
            summ_acc.append(np.concatenate((accuracy[:,:,0].flatten(),accuracy[:,:,1].flatten())))
            summ_marginmean.append(np.concatenate((np.concatenate(mean_instance[:,0],axis=0),np.concatenate(mean_instance[:,1],axis=0))))
            summ_marginstd.append(np.concatenate((np.concatenate(std_instance[:,0],axis=0),np.concatenate(std_instance[:,1],axis=0))))

    summary = pd.DataFrame({
    #"Mouse": np.concatenate(np.array(summ_mouse, dtype=object),axis=0),
    "ID": np.concatenate(np.array(summ_id, dtype=object),axis=0),
    "Date": np.concatenate(np.array(summ_date, dtype=object),axis=0),
    "Block": np.concatenate(np.array(summ_block, dtype=object),axis=0),
    "Region": np.concatenate(np.array(summ_region, dtype=object),axis=0),
    "Layer": np.concatenate(np.array(summ_layer, dtype=object),axis=0),
    "Category_pair": np.concatenate(np.array(summ_catpair, dtype=object),axis=0),
    "Train_texture": np.concatenate(np.array(summ_traintex, dtype=object),axis=0),
    "Accuracy": np.concatenate(np.array(summ_acc, dtype=object),axis=0),
    "Margin_mean": np.concatenate(np.array(summ_marginmean, dtype=object),axis=0),
    "Margin_std": np.concatenate(np.array(summ_marginstd, dtype=object),axis=0)
    })
    summary["Date"] = pd.to_datetime(summary["Date"], format="%Y_%m_%d")
    summary.loc[summary['Margin_mean']==0,['Margin_mean','Margin_std','Accuracy']] = np.nan
    summary.reset_index(drop=True, inplace=True)
    return summary