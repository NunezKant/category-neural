import numpy as np 
from scipy.stats import zscore
from src import utils 
import pandas as pd

def get_r1r2(sp, istart, stimid, nmax = 16000, nrepmax = 2, dt = [1, 3]):
    NN,NT  = sp.shape

    resp = np.zeros((NN, len(istart))) 
    for j in range(dt[0], dt[1]):
        resp += sp[:,  np.minimum(NT-1, istart+j)] 

    r = np.zeros((2, NN, nmax))    

    #ivalid = np.zeros(1000,'bool')
    k = 0
    for j in range(nmax):
        ix = (stimid==j).nonzero()[0]
        if len(ix)>=2:            
            r[0,:,k] = resp[:, ix[:nrepmax:2]].mean(1)
            r[1,:,k] = resp[:, ix[1:nrepmax:2]].mean(1)

            k += 1
            #ivalid[j] = 1
    r = r[:,:,:k]
    return resp, r

def cat_variance_old(resp, stimid, icat = [0, 1]):
    
    ss = np.arange(len(stimid))    
    isub = (ss<1000000) * np.isin(stimid, icat)

    #print(isub.sum())
    R = resp[:, isub]
    R = zscore(R, 1)

    cat1 = stimid[isub]==icat[0]
    cat2 = stimid[isub]==icat[1]
    cc =  .5 * (R[:,cat1] *  (R[:, cat1].sum(1, keepdims=True) - R[:,cat1])).mean(1)/(cat1.sum()-1) 
    cc += .5 * (R[:,cat2] *  (R[:, cat2].sum(1, keepdims=True) - R[:,cat2])).mean(1)/(cat2.sum()-1)


    dmu = R[:, cat1].mean(1) - R[:, cat2].mean(1)
    dsd = R[:, cat1].std(1)  + R[:, cat2].std(1)
    dp = 2 * dmu / dsd

    return cc, dp


def sig_variance(resp, stimid, zscore = False):
    # this function computers signal variance based on repeated presentations of the same stimuli
    # if you have more than two repeats, this can take advantage of that (equivalent to forming all pairs of repeats)

    # resp = (neurons, #stimuli)
    # stimid = (#stimuli)

    # example cc = sig_variance(resp, stimid)

    iunq = np.unique(stimid)
    if zscore:
        R = zscore(resp, 1)
    else:
        R = resp
    NN = resp.shape[0]
    cc = np.zeros(NN,)    
    nsum = 0
    for j in range(len(stimid)):
        iss = stimid==stimid[j]
        if iss.sum()<2:
            continue;
        cc += (R[:, j] * (R[:, iss].sum(1) - R[:, j])) / (iss.sum()-1)
        nsum += 1 
    if nsum<2:
        raise ValueError('Found %d stimuli with at least two repeats. Requires at least 2.'%nsum)
    cc /= nsum
    return cc

       
def cat_variance(resp, stimid, stimcat, zscore = False):
    # this function computers how much variance a category variable explains. 
    # it does not require stimulus repeats, but it does require multiple images in the same category. 

    # resp    = (neurons, #stimuli)
    # stimid  = (#stimuli)
    # stimcat = (#stimuli)

    # example: rr = cat_variance(resp, stimid , (stimid//500).astype('int32'))
    # where stimid is an index into Farah's text16000 stimuli (first 500 are category 1, second 500 are category 2 etc). 


    iunq = np.unique(stimid)
    icat = np.unique(stimcat)
    if zscore:
        R = zscore(resp, 1)
    else:
        R = resp
    NN = resp.shape[0]

    ccat = np.zeros(NN,)
    nsum = 0 

    Rcat = np.zeros((NN, len(icat)))
    for t in range(len(icat)):
        icc = stimcat==icat[t]
        Rcat[:, t] = R[:,icc].sum(1)


    for j in range(len(stimid)):
        iss = stimid==stimid[j]
        scat = stimcat[iss]
        if len(np.unique(scat))>1:
            raise ValueError('The exemplar has different categories in different repeats. Something is wrong with the indices?')

        icc = stimcat==stimcat[j]
        jj = (icat==stimcat[j]).nonzero()[0][0]
        if len(np.unique(stimid[icc]))<2:
            continue;

        ccat += R[:, j] * (Rcat[:, jj] - R[:, iss].sum(1)) / (icc.sum() - iss.sum())
        nsum += 1
 
    #if nsum<2:
     #   raise ValueError('Found %d categories with at least two exemplars. Requires at least 2.'%nsum)

    ccat /= nsum

    return ccat

def build_categories(stimid, n_instances = 32, n_per_cat = 4):
    texture_ids, _ = np.unique(stimid, return_counts = True)
    tex8x4_ids = np.arange(0, n_instances+1, n_per_cat)
    cat_i = 1
    category_id = np.zeros_like(stimid)
    n_loop = len(tex8x4_ids)-1
    # this build the categories for the 8x4 texture stimuli
    for i in range(n_loop):
        selected_ids = np.where((stimid >= tex8x4_ids[i]+1) & (stimid <= tex8x4_ids[i+1]))[0] 
        category_id[selected_ids] = cat_i
        cat_i += 1
    # this build the categories for the rest of the stimuli
    for tex in texture_ids:
        if tex not in range(1,n_instances+1):
            selected_ids = np.where(stimid == tex)[0]
            category_id[selected_ids] = cat_i
            cat_i += 1
    category_id = category_id.astype(int)
    return category_id

def get_stim_response_matrix(MouseObject: object, area: str, plane: int, cc_tsh: int = 0.1):
    """
    This function returns a matrix of zscored responses for the given area and plane.

    Parameters
    ----------
    MouseObject : object
        Mouse object containing all the data
    area : str
        Area of the brain to consider
    plane : int
        Plane to consider
    cc_tsh : int
        Threshold for the signal variance coefficient

    Returns
    -------
    zs_rm : np.array
        Matrix of zscored responses for the given area and plane with cc>cc_tsh
        shape (neurons, #stimuli/cats, #reps)
    """


    from src import invariance

    firstn_cats = 8
    n_instances = 4
    ix_area = utils.get_region_idx(MouseObject.iarea, area)
    if plane == 1:
        ix_plane = (MouseObject._iplane >= 10)
    elif plane == 2:
        ix_plane = (MouseObject._iplane < 10)
    elif plane == 0:
        ix_plane = np.ones_like(MouseObject._iplane, dtype=bool)
    else:
        raise ValueError("Layer must be 1 or 2 for depths, or 0 for all planes")

    category_id = invariance.build_categories(MouseObject.subset_stim) # builds the vector of categories with based on the stimids shape (stimids,)
    stim_ids = MouseObject.subset_stim[category_id <= firstn_cats] # gets the stimids for the first 8 categories (the 32 textures of the 8x4 dataset)
    neurons = MouseObject.neurons_atframes[:,category_id <= firstn_cats] # gets the neurons at frames for the first 8 categories
    cc = invariance.sig_variance(neurons, stim_ids) # gets the signal variance for each neuron only for the 8x4 dataset
    if MouseObject.name.startswith(("DR","TX")):
        neurons = zscore(neurons, axis = 1) # for old mice we presented other textures too, recentering them on the 8x4 dataset.
    #neurons_plane_region = MouseObject.neurons_atframes[ix_plane * ix_area] 
    neurons_plane_region = neurons[ix_plane * ix_area]
    cc_plane_region = cc[ix_plane * ix_area]
    if cc_tsh<1: 
        sig_neurons = neurons_plane_region[cc_plane_region>cc_tsh]
    else:
        tsh = np.percentile(cc_plane_region, cc_tsh)
        sig_neurons = neurons_plane_region[cc_plane_region>tsh]
    total_samples = firstn_cats * n_instances
    _, nc = np.unique(MouseObject.subset_stim, return_counts = True)
    nc = nc[:total_samples]
    nreps = np.min(nc)
    NN = sig_neurons.shape[0]
    stim_response = np.zeros((NN,total_samples,nreps))
    np.random.seed(333)
    for i in range(1,33): # this loop creates the NN, 32, reps matrix
        #instance_idx = np.where(MouseObject.subset_stim == i)[0]
        instance_idx = np.where(stim_ids == i)[0]
        instance_idx = np.random.permutation(instance_idx)
        instance_idx = instance_idx[:nreps]
        stim_response[:,i-1,:] = sig_neurons[:,instance_idx]
    zs_rm = stim_response 
    #sanity check:
    if len(np.where(np.isnan(zs_rm))[0])>0:
        #print("There are NaNs in the zscored representation matrix")
        bad_neurons = np.unique(np.where(np.isnan(zs_rm))[0])
        #print(f"Neuron no.: {bad_neurons}")
        #print(f"Instances: {np.unique(np.where(np.isnan(zs_rm))[1])}")
        #print(f"reps: {np.unique(np.where(np.isnan(zs_rm))[2])}")
        good_neurons = np.array([i for i in range(NN) if i not in bad_neurons])
        #print(f"Keeping {good_neurons.shape[0]} neurons")
        zs_rm = zs_rm[good_neurons,:,:] # only keep the neurons that are not nan 
    return zs_rm


def get_stim_response_matrix_areas(MouseObject: object, area: int, plane: int, ctype: str, cc_tsh: int = 0.1):
    """
    This function returns a matrix of zscored responses for the given area and plane.

    Parameters
    ----------
    MouseObject : object
        Mouse object containing all the data
    area : int
        Area of the brain to consider
    plane : int
        Plane to consider
    ctype : str
        Cell type to consider ('exc' or 'inh')
    cc_tsh : int
        Threshold for the signal variance coefficient

    Returns
    -------
    zs_rm : np.array
        Matrix of zscored responses for the given area and plane with cc>cc_tsh
        shape (neurons, #stimuli/cats, #reps)
    """

    firstn_cats = 8
    n_instances = 4
    ix_area = np.isin(MouseObject.iarea, area)
    if plane == 1:
        ix_plane = (MouseObject._iplane >= 10) #deep 
    elif plane == 2:
        ix_plane = (MouseObject._iplane < 10) #sup
    elif plane == 0:
        ix_plane = np.ones_like(MouseObject._iplane, dtype=bool)
    else:
        raise ValueError("Layer must be 1 or 2 for depths, or 0 for all planes")
    if ctype == "exc":
        ix_type = ~MouseObject.isred[:,0].astype(bool)
    elif ctype == "inh":
        ix_type = MouseObject.isred[:,0].astype(bool)
    else:
        raise ValueError("Cell type must be 'exc' or 'inh'")

    category_id = build_categories(MouseObject.subset_stim) # builds the vector of categories with based on the stimids shape (stimids,)
    stim_ids = MouseObject.subset_stim[category_id <= firstn_cats] # gets the stimids for the first 8 categories (the 32 textures of the 8x4 dataset)
    neurons = MouseObject.neurons_atframes[:,category_id <= firstn_cats] # gets the neurons at frames for the first 8 categories
    cc = sig_variance(neurons, stim_ids) # gets the signal variance for each neuron only for the 8x4 dataset
    if MouseObject.name.startswith(("DR","TX")):
        neurons = zscore(neurons, axis = 1) # for old mice we presented other textures too.
    #neurons_plane_region = MouseObject.neurons_atframes[ix_plane * ix_area] 
    neurons_plane_region = neurons[ix_plane * ix_area * ix_type]
    cc_plane_region = cc[ix_plane * ix_area * ix_type]
    if cc_tsh<1: 
        sig_neurons = neurons_plane_region[cc_plane_region>cc_tsh]
    else:
        tsh = np.nanpercentile(cc_plane_region, cc_tsh)
        sig_neurons = neurons_plane_region[cc_plane_region>tsh]
    #print(f"neurons in {area} and plane {plane} with cc>{cc_tsh} : {sig_neurons.shape[0]}")
    total_samples = firstn_cats * n_instances
    _, nc = np.unique(MouseObject.subset_stim, return_counts = True)
    nc = nc[:total_samples]
    nreps = np.min(nc)
    NN = sig_neurons.shape[0]
    stim_response = np.zeros((NN,total_samples,nreps))
    np.random.seed(333)
    for i in range(1,33): # this loop creates the NN, 32, reps matrix
        #instance_idx = np.where(MouseObject.subset_stim == i)[0]
        instance_idx = np.where(stim_ids == i)[0]
        instance_idx = np.random.permutation(instance_idx)
        instance_idx = instance_idx[:nreps]
        stim_response[:,i-1,:] = sig_neurons[:,instance_idx]
    zs_rm = stim_response # fot the FX mice, the 8x4 dataset was the only one used and spks (neurons at frames) come already zscored.
    #sanity check:
    if len(np.where(np.isnan(zs_rm))[0])>0:
        #print("There are NaNs in the zscored representation matrix")
        bad_neurons = np.unique(np.where(np.isnan(zs_rm))[0])
        #print(f"Neuron no.: {bad_neurons}")
        #print(f"Instances: {np.unique(np.where(np.isnan(zs_rm))[1])}")
        #print(f"reps: {np.unique(np.where(np.isnan(zs_rm))[2])}")
        good_neurons = np.array([i for i in range(NN) if i not in bad_neurons])
        #print(f"Keeping {good_neurons.shape[0]} neurons")
        zs_rm = zs_rm[good_neurons,:,:] # only keep the neurons that are not nan 
    return zs_rm

def get_representation_matrix_areas(MouseObject: object, area: str, plane: int, ctype: str, cc_tsh: int = 0.1):
    """
    This function returns a matrix of the representation of the stimuli for the given area and plane, based on the zscored responses.
    in this case, the 32 first instances correspond to the 8x4 texture dataset, and the rest to the rest of the stimuli. 
    Parameters:
    MouseObject : object
        Mouse object containing all the data
    area : str
        Area of the brain to consider
    plane : int
        Plane to consider
    cc_tsh : int
        Threshold for the signal variance coefficient
    Returns:
    representation_matrix : np.array
        Matrix of the representation of the stimuli for the given area and plane
        shape (total_samples, total_samples)
    """
    firstn_cats = 8
    n_instances = 4
    total_samples = firstn_cats * n_instances
    stim_response = get_stim_response_matrix_areas(MouseObject, area, plane, ctype, cc_tsh)
    representation_matrix = np.full((total_samples, total_samples), np.nan)
    zs_rm = stim_response
    nreps = zs_rm.shape[2]


    for i in range(total_samples):
        first_h = nreps//2
        instance_response = zs_rm[:,i,:]
        fh_neurons = instance_response[:,:first_h].mean(axis=1)
        sh_neurons = instance_response[:,first_h:].mean(axis=1)
        representation_matrix[i,i] = np.corrcoef(fh_neurons, sh_neurons)[0,1]


    from itertools import combinations
    pairs = list(combinations(np.arange(total_samples),2))
    for pair in pairs:
        intance_a = zs_rm[:,pair[0],:]
        intance_b = zs_rm[:,pair[1],:]
        fh_neurons = intance_a.mean(axis=1)
        sh_neurons = intance_b.mean(axis=1)
        representation_matrix[pair[0],pair[1]] = np.corrcoef(fh_neurons, sh_neurons)[0,1]
        representation_matrix[pair[1],pair[0]] = np.corrcoef(fh_neurons, sh_neurons)[0,1]
    return representation_matrix

def get_representation_matrix(MouseObject: object, area: str, plane: int, cc_tsh: int = 0.1):
    """
    This function returns a matrix of the representation of the stimuli for the given area and plane, based on the zscored responses.
    in this case, the 32 first instances correspond to the 8x4 texture dataset, and the rest to the rest of the stimuli. 
    Parameters:
    MouseObject : object
        Mouse object containing all the data
    area : str
        Area of the brain to consider
    plane : int
        Plane to consider
    cc_tsh : int
        Threshold for the signal variance coefficient
    Returns:
    representation_matrix : np.array
        Matrix of the representation of the stimuli for the given area and plane
        shape (total_samples, total_samples)
    """
    firstn_cats = 8
    n_instances = 4
    total_samples = firstn_cats * n_instances
    stim_response = get_stim_response_matrix(MouseObject, area, plane, cc_tsh)
    representation_matrix = np.zeros((total_samples, total_samples))
    zs_rm = stim_response
    nreps = zs_rm.shape[2]


    for i in range(total_samples):
        first_h = nreps//2
        instance_response = zs_rm[:,i,:]
        fh_neurons = instance_response[:,:first_h].mean(axis=1)
        sh_neurons = instance_response[:,first_h:].mean(axis=1)
        representation_matrix[i,i] = np.corrcoef(fh_neurons, sh_neurons)[0,1]


    from itertools import combinations
    pairs = list(combinations(np.arange(total_samples),2))
    for pair in pairs:
        intance_a = zs_rm[:,pair[0],:]
        intance_b = zs_rm[:,pair[1],:]
        fh_neurons = intance_a.mean(axis=1)
        sh_neurons = intance_b.mean(axis=1)
        representation_matrix[pair[0],pair[1]] = np.corrcoef(fh_neurons, sh_neurons)[0,1]
        representation_matrix[pair[1],pair[0]] = np.corrcoef(fh_neurons, sh_neurons)[0,1]
    return representation_matrix

def intracat_invariance(ss_m, categories: int = 8, instances: int  = 4): 
    """
    Computes the intra-category invariance of a representation matrix

    Parameters
    ----------
    ss_m : array
        Representation matrix
    
    Returns
    -------
    intra_inv : array
        Intra-category invariance for each category
    """
    n_total = categories * instances
    cat_instance = np.arange(0,n_total,instances)
    a, b = np.triu_indices(instances, k = 1)
    intra_mean = []
    for cat in cat_instance:
        current_a = a + cat
        current_b = b + cat
        mean = []
        for idx in zip(current_a,current_b):
            mean.append(ss_m[idx[0],idx[1]])
        intra_mean.append(np.mean(np.array(mean)))
    return np.array(intra_mean)

def condense_matrix(ss_m, categories: int = 8, instances: int = 4):
    """
    Condenses a representation matrix into a matrix of mean intra-inter category invariance

    Parameters
    ----------
    ss_m : array
        Representation matrix
    categories : int
        Number of categories
    instances : int
        Number of instances per category

    Returns
    -------
    condensed_matrix : array
        Matrix of mean intra-inter category invariance
        shape (categories, categories)
    """

    total_instances = categories * instances
    cat_init = np.arange(0,total_instances,instances)
    cat_end = np.arange(instances,total_instances+1,instances)
    cats = np.arange(0,categories)
    condensed_matrix = np.zeros((categories,categories))
    for slctd_ct in cats:
        row_from_matrix = ss_m[cat_init[slctd_ct]:cat_end[slctd_ct],:]
        mean_per_row = []
        for cat in cats:
            if cat == slctd_ct:
                cat_responses = row_from_matrix[:,cat_init[cat]:cat_end[cat]]
                a,b = np.triu_indices(instances,1)
                mean = []
                for idx in zip(a,b):
                    mean.append(cat_responses[idx[0],idx[1]])
                mean = np.array(mean).mean()
                mean_per_row.append(mean)
            else:
                inter_mean = row_from_matrix[:,cat_init[cat]:cat_end[cat]].mean()
                mean_per_row.append(inter_mean)
        mean_per_row = np.array(mean_per_row)
        condensed_matrix[slctd_ct,:] = mean_per_row
    return condensed_matrix

def intercat_invariance(condensed_matrix, chosen_cat: int):
    """
    Computes the inter-category invariance of a condensed matrix o a chosen category

    Parameters
    ----------

    condensed_matrix : array
        Condensed matrix of mean intra-inter category invariance
        shape (categories, categories)
    chosen_cat : int
        Chosen category

    Returns
    -------
    intercat_invariance : float
        Inter-category invariance for the chosen category
    """
    n_cats = condensed_matrix.shape[0]
    cats = np.arange(n_cats)
    row_vector = condensed_matrix[chosen_cat, :]
    intercat_invariance = row_vector[cats != chosen_cat].mean()
    return intercat_invariance

def reduce_to_behav(ss_m, categories: int = 8, instances: int  = 4):
    """ 
    Reduces a representation matrix to ony textures used in the behavioral task

    Parameters
    ----------
    ss_m : array
        Representation matrix
    categories : int
        Number of categories
    instances : int
        Number of instances per category

    Returns
    -------
    behav_matrix : array
        Representation matrix for the textures used in the behavioral task
    """
    n_total = categories * instances
    cat_instance = np.arange(0,n_total,instances)
    behav_matrix = np.zeros((16,32))
    behav_instances = np.arange(0,16,2)
    for bhv, cat in zip(behav_instances, cat_instance):
        if cat == 12: # special case of rocks 
            behav_matrix[bhv,:] = ss_m[cat,:]
            behav_matrix[bhv+1,:] = ss_m[cat+2,:]
        else:
            behav_matrix[bhv,:] = ss_m[cat,:]
            behav_matrix[bhv+1,:] = ss_m[cat+1,:]
    cat_i = cat_instance + 1 
    col_cats = np.sort(np.concatenate((cat_instance, cat_i)))
    col_cats[np.where(col_cats == 13)] = 14
    behav_matrix = behav_matrix[:,col_cats]
    return behav_matrix

def get_pair_invariance_df(mtx: np.array):
    cat = ['Leaves', 'Circles', 'Dryland', 'Rocks', 'Tiles', 'Squares', 'Rleaves', 'Paved'] 
    areas = ['V1', 'medial', 'lateral', 'anterior']
    n_features = len(mtx.shape)
    layers = [1, 2] 
    df_pair = pd.DataFrame()
    if n_features == 5:
        for m in range(mtx.shape[0]):
            for i_a, area in enumerate(areas):
                for layer in layers:
                    df = pd.DataFrame()
                    matrix = mtx[m, i_a, layer-1, :, :]
                    condensed = condense_matrix(matrix, 8, 4)
                    a, b = np.triu_indices(8,1)
                    positive_cat = []
                    negative_cat = []
                    invariance_index = []
                    for i in zip(a,b):
                        positive_cat.append(cat[i[0]])
                        negative_cat.append(cat[i[1]])
                        invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
                    df["positive_category"] = np.array(positive_cat)
                    df["negative_category"] = np.array(negative_cat)
                    df["pair_invariance"] = np.array(invariance_index)
                    df["area"] = area
                    df["layer"] = layer
                    df["mouse"] = m  
                    df_pair = pd.concat([df_pair, df])
        df_pair.reset_index(inplace=True, drop=True)
    elif n_features == 4:
        for m in range(mtx.shape[0]):
            for i_a, area in enumerate(areas):
                df = pd.DataFrame()
                matrix = mtx[m, i_a, :, :]
                condensed = condense_matrix(matrix, 8, 4)
                a, b = np.triu_indices(8,1)
                positive_cat = []
                negative_cat = []
                invariance_index = []
                for i in zip(a,b):
                    positive_cat.append(cat[i[0]])
                    negative_cat.append(cat[i[1]])
                    invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
                df["positive_category"] = np.array(positive_cat)
                df["negative_category"] = np.array(negative_cat)
                df["pair_invariance"] = np.array(invariance_index)
                df["area"] = area
                df["mouse"] = m
                df_pair = pd.concat([df_pair, df])
    else:
        raise ValueError("The input matrix must have 4 or 5 dimensions, got %d dimensions." % n_features)
    return df_pair