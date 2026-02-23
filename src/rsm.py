import numpy as np 
import pandas as pd
from src import utils
from scipy.stats import zscore


def filter_neurons(m, area, celltype):
        try:
            ia = utils.get_region_idx(m.iarea, area)
        except:
            AssertionError("area must be 'V1', 'medial', 'lateral' or 'anterior'")
        if celltype == "Pyr":
            selected_type = np.logical_not(m.isred[:,0]).astype(bool)
        elif celltype == "Int":
            selected_type = m.isred[:,0].astype(bool)
        else:
            AssertionError("celltype must be 'Pyr' or 'Int' or None")
        
        selection = ia * selected_type
        prop = selection.sum() / ia.sum()
        return selection, prop


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


def get_sig_neurons(m, cc, area: str, celltype: str, cc_tsh: float = 0.1) -> np.ndarray:
    selection, _ = filter_neurons(m, area, celltype)
    # ensure column exists (zero-based, as created above)
    if 'stim_id_matrix' not in m.frameselector.columns:
        mask_B = m.frameselector['trial_type'].isin(['non rewarded', 'non rewarded test'])
        base = m.frameselector['istim'].astype('Int64') - 1  # zero-based
        m.frameselector['stim_id_matrix'] =np.where(~mask_B, base, base + 51)

    # per-trial response over first 100 positions
    per_trial_resp = m.interp_spks[:, :, :100].mean(2)  # (neurons, trials)
    per_trial_resp = per_trial_resp[selection, :]  # filter neurons
    cc_region = cc[selection]
    sig_neurons = per_trial_resp[cc_region > cc_tsh]
    nneurons = sig_neurons.shape[0]
    sig_neurons = zscore(sig_neurons, axis=1, nan_policy='omit') #across textures
    return sig_neurons, nneurons

def get_istim_matrix(m, area: str, celltype: str, cc_tsh: float = 0.1) -> np.ndarray:
    selection, _ = filter_neurons(m, area, celltype)
    # ensure column exists (zero-based, as created above)
    if 'stim_id_matrix' not in m.frameselector.columns:
        mask_B = m.frameselector['trial_type'].isin(['non rewarded', 'non rewarded test'])
        base = m.frameselector['istim'].astype('Int64') - 1  # zero-based
        m.frameselector['stim_id_matrix'] =np.where(~mask_B, base, base + 51)

    # per-trial response over first 100 positions
    per_trial_resp = m.interp_spks[:, :, :100].mean(2)  # (neurons, trials)
    # get stim ids
    stim_ids = []
    for tn in m.frameselector['trial_no'].unique():
        stim_ids.append(m.frameselector.loc[m.frameselector['trial_no'] == tn, 'stim_id_matrix'].values[0])
    stim_ids = np.array(stim_ids)
    cc = sig_variance(per_trial_resp, stim_ids)
    per_trial_resp = per_trial_resp[selection, :]  # filter neurons
    cc_region = cc[selection]
    sig_neurons = per_trial_resp[cc_region > cc_tsh]
    nneurons = sig_neurons.shape[0]
    id_responses = np.full((nneurons, 102, 2), np.nan, dtype=float)
    # build responses by stim id (0..101)
    stim_vals = m.frameselector['stim_id_matrix']
    for idx in range(102):
        mask = (stim_vals == idx)
        if mask.any():
            trials = m.frameselector.loc[mask, 'trial_no'].dropna().unique().astype(int) - 1
            if trials.size >= 2:
                id_responses[:, idx, 0] = sig_neurons[:, trials[0]]
                id_responses[:, idx, 1] = sig_neurons[:, trials[1]]
    print(id_responses.shape)
    id_responses = zscore(id_responses, axis=1, nan_policy='omit') #across textures
    return id_responses

def get_rsm(stim_mtx: np.ndarray) -> np.ndarray:
    from itertools import combinations
    pairs = list(combinations(np.arange(102),2))
    representation_matrix = np.ones((102,102)) * np.nan
    for pair in pairs:
        instance_a = stim_mtx[:,pair[0],:]
        instance_b = stim_mtx[:,pair[1],:]
        if np.isnan(instance_a).all() or np.isnan(instance_b).all():
            continue
        fh_neurons = instance_a.mean(axis=1)
        sh_neurons = instance_b.mean(axis=1)
        representation_matrix[pair[0],pair[1]] = np.corrcoef(fh_neurons, sh_neurons)[0,1]
        representation_matrix[pair[1],pair[0]] = np.corrcoef(fh_neurons, sh_neurons)[0,1]
    return representation_matrix

def compute_iindex(rsm: np.ndarray) -> float:
    # A: 0..50, B: 51..101
    intraA = np.nanmean(rsm[:51, :51])
    intraB = np.nanmean(rsm[51:, 51:])
    intercat = np.nanmean(rsm[:51, 51:])
    return intraA, intraB, intercat 

### TRIAL X TRIAL 

def get_rsm_trial(sig_neurons: np.ndarray) -> np.ndarray:
    n_trials = sig_neurons.shape[1]
    S = np.full((n_trials, n_trials), np.nan)
    #from scipy.stats import pearsonr
    for i in range(n_trials):
        vi = sig_neurons[:, i]
        for j in range(i, n_trials):  # start from i to skip lower triangle
            vj = sig_neurons[:, j]
            r = np.corrcoef(vi, vj)[0, 1]
            S[i, j] = r

    i_upper = np.triu_indices(n_trials, 1)
    S[i_upper[1], i_upper[0]] = S[i_upper]
    return S

def get_trials_per_stim(m):
    trials_per_stim = []
    for tn in pd.Series(m.frameselector['stim_id_matrix'].unique()).sort_values().unique():
        trials_per_stim.append(m.frameselector.loc[m.frameselector['stim_id_matrix'] == tn, 'trial_no'].unique().astype(int)- 1)
    trials_per_stim = np.array(trials_per_stim, dtype=object)
    return trials_per_stim

def clean_rsm_trial(S: np.ndarray, sig_neurons: np.ndarray, trials_per_stim: np.ndarray) -> np.ndarray:
    n_trials = sig_neurons.shape[1]
    S_cleaned = S.copy()
    for stim, trials in enumerate(trials_per_stim):
        if len(trials) == 2:
            # create masks 
            S_cleaned[trials[0], trials[1]] = np.nan
            S_cleaned[trials[1], trials[0]] = np.nan
    # set diagonal to nan
    diagnonal_indices = np.arange(n_trials)
    S_cleaned[diagnonal_indices, diagnonal_indices] = np.nan
    return S_cleaned

def filter_rsm(rsm, trial_cond1, trial_cond2):
    filtered = np.concatenate((trial_cond1, trial_cond2))
    return rsm[np.ix_(filtered, filtered)]

def compute_iindex_cond(rsm_filtered: np.ndarray, cond1) -> float:
    intraA = np.nanmean(rsm_filtered[:len(cond1), :len(cond1)])
    intraB = np.nanmean(rsm_filtered[len(cond1):, len(cond1):])
    intercat = np.nanmean(rsm_filtered[:len(cond1), len(cond1):])
    return intraA, intraB, intercat

def get_sig_variance(m) -> np.ndarray:
    per_trial_resp = m.interp_spks[:, :, :100].mean(2)  # (neurons, trials)
    # get stim ids
    stim_ids = []
    for tn in m.frameselector['trial_no'].unique():
        stim_ids.append(m.frameselector.loc[m.frameselector['trial_no'] == tn, 'stim_id_matrix'].values[0])
    stim_ids = np.array(stim_ids)
    cc = sig_variance(per_trial_resp, stim_ids)
    return cc