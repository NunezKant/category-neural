"""
utility functions for the analysis of the data
"""
import os
import sys
from scipy import io
import pickle
import tkinter as tk
sys.path.insert(0, r"C:\Users\labadmin\Documents\suite2p")
sys.path.insert(0, r"C:\Users\labadmin\Documents\rastermap")
from sklearn.decomposition import PCA
import numpy as np
from suite2p.extraction import dcnv
from rastermap import mapping
from scipy import ndimage
from tqdm import tqdm
from dataclasses import dataclass
from scipy.stats import zscore 
import os
from pathlib import Path
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime, timedelta
from sklearn.decomposition import TruncatedSVD
from src import udcnv

##### SUITE2P FUNCTIONS #####


def deconvolve(root, ops):
    """
    Correct the lags of the dcnv data.

    Parameters
    ----------
    root : str
        Path to the experiment.
    ops : dict
        suite2p pipeline options
    """

    # we initialize empty variables
    spks = np.zeros(
        (0, ops["nframes"]), np.float32
    )  # the neural data will be Nneurons by Nframes.
    stat = np.zeros((0,))  # these are the per-neuron stats returned by suite2p
    xpos, ypos = np.zeros((0,)), np.zeros((0,))  # these are the neurons' 2D coordinates

    # this is for channels / 2-plane mesoscope
    tlags = 0.25 + np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
    tlags = np.hstack((tlags, tlags))

    # loop over planes and concatenate
    iplane = np.zeros((0,))

    th_low, th_high = 0.5, 1.1
    for n in range(ops["nplanes"]):
        ops = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"), allow_pickle=True
        ).item()

        # load and deconvolve
        iscell = np.load(os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"))[
            :, 1
        ]
        iscell = (iscell > th_low) * (iscell < th_high)

        stat0 = np.load(
            os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"), allow_pickle=True
        )[iscell]
        ypos0 = np.array(
            [stat0[n]["med"][0] for n in range(len(stat0))]
        )  # notice the python list comprehension [X(n) for n in range(N)]
        xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

        ypos0 += ops["dy"]  # add the per plane offsets (dy,dx)
        xpos0 += ops["dx"]  # add the per plane offsets (dy,dx)

        f_0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))[iscell]
        f_neu0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))[
            iscell
        ]
        f_0 = f_0 - 0.7 * f_neu0

        # compute spks0 with deconvolution
        if tlags[n] < 0:
            f_0[:, 1:] = (1 + tlags[n]) * f_0[:, 1:] + (-tlags[n]) * f_0[:, :-1]
        else:
            f_0[:, :-1] = (1 - tlags[n]) * f_0[:, :-1] + tlags[n] * f_0[:, 1:]

        f_0 = dcnv.preprocess(
            f_0.copy(),
            ops["baseline"],
            ops["win_baseline"],
            ops["sig_baseline"],
            ops["fs"],
            ops["prctile_baseline"],
        )
        spks0 = dcnv.oasis(f_0, ops["batch_size"], ops["tau"], ops["fs"])

        spks0 = spks0.astype("float32")
        iplane = np.concatenate(
            (
                iplane,
                n
                * np.ones(
                    len(stat0),
                ),
            )
        )
        stat = np.concatenate((stat, stat0), axis=0)
        if spks.shape[1] > spks0.shape[0]:
            spks0 = np.concatenate(
                (
                    spks0,
                    np.zeros(
                        (spks0.shape[0], spks.shape[1] - spks0.shape[1]), "float32"
                    ),
                ),
                axis=1,
            )
        spks = np.concatenate((spks, spks0), axis=0)
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)

        print(f"plane {n}, neurons: {len(xpos0)}")

    print(f"total neurons: {len(spks)}")

    xpos = xpos / 0.75
    ypos = ypos / 0.5

    return spks, stat, xpos, ypos, iplane


def baselining(ops, tlag, F):
    """
    Baseline the neural data before deconvolution

    Parameters:
    ----------
    ops : dict
        Dictionary with the experiment info
    tlag : int
        Time lag for the deconvolution
    F : array
        Deconvolved fluorescence - neurophil corrected 
    Returns:
    ----------
    F : array
        Baselined deconvolved fluorescence
    """
    #F = preprocess(F, Fneu, ops["win_baseline"], ops["sig_baseline"], ops["fs"])
    F = dcnv.preprocess(F, ops['baseline'], ops['win_baseline'], ops['sig_baseline'],
                       ops['fs'], ops['prctile_baseline'])
    if tlag < 0:
        F[:, 1:] = (1 + tlag) * F[:, 1:] + (-tlag) * F[:, :-1]
    else:
        F[:, :-1] = (1 - tlag) * F[:, :-1] + tlag * F[:, 1:]
    return F


def preprocess(F, Fneu, win_baseline, sig_baseline, fs):
    """
    Preprocess the fluorescence data

    Parameters:
    ----------
    F : array
        Deconvolved fluorescence
    Fneu : array
        Neurophil fluorescence
    baseline : int
        Baseline for the fluorescence
    win_baseline : int
        Window for the baseline
    sig_baseline : int
        Sigma for the baseline
    fs : int
        Sampling rate
    Returns:
    ----------
    F : array
        Preprocessed deconvolved fluorescence
    """
    win = int(win_baseline * fs)

    Flow = ndimage.gaussian_filter(F, [0.0, sig_baseline])
    Flow = ndimage.minimum_filter1d(Flow, win)
    Flow = ndimage.maximum_filter1d(Flow, win)
    F = F - 0.7 * Fneu
    Flow2 = ndimage.gaussian_filter(F, [0.0, sig_baseline])
    Flow2 = ndimage.minimum_filter1d(Flow2, win)
    Flow2 = ndimage.maximum_filter1d(Flow2, win)

    Fdiv = np.maximum(10, Flow.mean(1))
    F = (F - Flow2) / Fdiv[:, np.newaxis]

    return F


### BEHAVIOR FUNCTIONS ###


def get_trial_categories(rewarded_trial_structure, new_trial_structure):
    """
    Compute the trial categories for the new trial structure

    Parameters
    ----------
    rewarded_trial_structure : array
        vector of the rewarded trials.
    new_trial_structure : array
        vector with new exemplar trials.

    Returns
    -------
    trial_categories : list
        List of the trial categories.
    trial_counts : dict
        Dictionary with the trial categories counts.

    """
    rewarded_trial_structure = np.array(rewarded_trial_structure)
    new_trial_structure = np.array(new_trial_structure)
    trial_categories = [None] * len(rewarded_trial_structure)
    rewarded_new_counter = 0
    rewarded_counter = 0
    non_rewarded_counter = 0
    non_rewarded_new_counter = 0

    for idx in range(new_trial_structure.shape[0]):
        if np.logical_and(rewarded_trial_structure[idx], new_trial_structure[idx]):
            trial_categories[idx] = "rewarded test"
            rewarded_new_counter += 1
        elif np.logical_and(
            rewarded_trial_structure[idx], np.logical_not(new_trial_structure[idx])
        ):
            trial_categories[idx] = "rewarded"
            rewarded_counter += 1
        elif np.logical_and(
            np.logical_not(rewarded_trial_structure[idx]), new_trial_structure[idx]
        ):
            trial_categories[idx] = "non rewarded test"
            non_rewarded_new_counter += 1
        elif np.logical_and(
            np.logical_not(rewarded_trial_structure[idx]),
            np.logical_not(new_trial_structure[idx]),
        ):
            trial_categories[idx] = "non rewarded"
            non_rewarded_counter += 1

    trial_counts = {
        "rewarded test": rewarded_new_counter,
        "rewarded": rewarded_counter,
        "non rewarded test": non_rewarded_new_counter,
        "non rewarded": non_rewarded_counter,
    }

    return np.array(trial_categories), trial_counts


##### MOUSE DATACLASS #####


def methods(MouseObject):
    """
    List all the methods of a MouseObject
    ----------
    MouseObject : object
    Returns
    -------
    methods : list
    """
    import inspect

    methods = inspect.getmembers(MouseObject, predicate=inspect.ismethod)
    m = [method[0] for method in methods if not method[0].startswith("__")]
    return m


def properties(MouseObject):
    """
    List all the properties of a MouseObject
    ----------
    MouseObject : object
    Returns
    -------
    properties : list
    """
    return MouseObject.__dict__.keys()


def get_trial_per_frame(MouseObject):
    init_frames, _, last_frame = get_init_frames_per_category(MouseObject)
    n_frames = MouseObject._spks.shape[1]
    trial_per_frame = np.empty(n_frames, dtype=float)
    trial_per_frame[:] = np.nan
    frames = np.sort(np.concatenate(init_frames))
    for i, init_frame in enumerate(frames):
        if init_frame == last_frame:
            trial_per_frame[init_frame:] = int(i + 1)
        else:
            trial_per_frame[init_frame : frames[i + 1]] = int(i + 1)
    return trial_per_frame

def get_lick_df(MouseObject, drop_last_trial=True):
    df = pd.DataFrame(MouseObject._timeline["Licks"].T, columns=["trial", "distance","alpha","is_rewarded","time", "flag","is_test"])
    df["datetime"] = pd.to_datetime(
        df["time"].apply(
            lambda x: datetime.fromordinal(int(x))
            + timedelta(days=x % 1)
            - timedelta(days=366)
        )
    )
    df = df.assign(distance = df["distance"]*10)
    df = df.assign(date=df["datetime"].dt.date)
    df = df.assign(hour_min_sec=df["datetime"].dt.time)
    df = df.assign(seconds_in_session=(df["datetime"] - df["datetime"][0]).dt.total_seconds())
    if drop_last_trial:
        n_trials = df.trial.unique()[-2].astype(int)
        df = df.loc[df.trial != df.trial.max()]
    else:
        n_trials = df.trial.unique()[-1].astype(int)
    isrewarded = MouseObject._timeline['TrialRewardStrct'].flatten()[:n_trials]
    isnew = MouseObject._timeline['TrialNewTextureStrct'].flatten()[:n_trials]
    trial_type , _ = get_trial_categories(isrewarded, isnew)
    for ix, ttype in enumerate(trial_type):
        df.loc[df.trial == ix+1, "trial_type"] = ttype
    df.drop(["time","datetime","is_rewarded","alpha"], axis=1, inplace=True)
    return df

def get_movement_df(MouseObject, intertrial_distance=100):
    corridor_length = ((MouseObject._settings['CorridorLength_dm'].item()*10))
    full_length = ((MouseObject._settings['CorridorLength_dm'].item()*10)+intertrial_distance)
    property_set = set(properties(MouseObject))
    assert {"_settings", "_timeline"}.issubset(
        property_set
    ), "self._settings and self._timeline not defined, make sure to use self.load_behav() first"
    data = MouseObject._timeline["Movement"]
    Movementdf = pd.DataFrame(
        {
            "pitch": data[0],
            "roll": data[1],
            "yaw": data[2],
            "distance": data[3] * 10,
            "trial": data[4],
            "time": data[5],
        }
    )
    Movementdf["time"] = pd.to_datetime(
        Movementdf["time"].apply(
            lambda x: datetime.fromordinal(int(x))
            + timedelta(days=x % 1)
            - timedelta(days=366)
        )
    )
    Movementdf["time"] = (Movementdf["time"] - Movementdf["time"][0]).dt.total_seconds()
    alpha_dx = MouseObject._settings["CorridorLength_dm"].item() #previously ContrastSteps
    alpha_dx = 1 / (alpha_dx / 2)
    Movementdf.loc[Movementdf["distance"] < corridor_length/2, "alpha"] = np.minimum(
        1, Movementdf.loc[Movementdf["distance"] < corridor_length/2, "distance"] / 10 * alpha_dx
    )
    Movementdf.loc[Movementdf["distance"] > corridor_length/2, "alpha"] = np.maximum(
        0,
        1
        - np.abs(
            1 - Movementdf.loc[Movementdf["distance"] > corridor_length/2, "distance"] / 10 * alpha_dx
        ),
    )
    Movementdf.loc[Movementdf["distance"] > corridor_length, "alpha"] = 0 
    Movementdf['distance_interp'] = np.nan
    Movementdf['vel_interp'] = np.nan
    for trial in Movementdf.trial.unique():
        distance = Movementdf.query(f'trial == {trial}')['distance']
        pitch = Movementdf.query(f'trial == {trial}')['pitch']
        roll = Movementdf.query(f'trial == {trial}')['roll']
        vel = np.sqrt(pitch**2 + roll**2)
        Movementdf.loc[Movementdf["trial"]==trial,'vel_interp'] = vel.cumsum()
        Movementdf.loc[Movementdf["trial"]==trial,'distance_interp'] = distance + (trial-1)*full_length
    Movementdf = Movementdf.assign(
        dd=np.diff(Movementdf["distance_interp"].shift(1), append=np.nan)
    )
    Movementdf = Movementdf.assign(dd=Movementdf["dd"].shift(-1))
    Movementdf = Movementdf.assign(
        dt=np.diff(Movementdf["time"].shift(1), append=np.nan)
    )
    Movementdf = Movementdf.assign(dt=Movementdf["dt"].shift(-1))
    # Movementdf.dropna(inplace = True)
    Movementdf = Movementdf.assign(speed=Movementdf["dd"] / Movementdf["dt"])
    Movementdf = Movementdf.assign(speed=Movementdf["speed"].ffill())
    return Movementdf


def get_rastermap(MouseObject, n_comp=200):
    S = MouseObject._spks.copy()
    mu = S.mean(1)
    sd = S.std(1)
    S = (S - mu[:, np.newaxis]) / (1e-10 + sd[:, np.newaxis])
    S -= S.mean(axis=0)
    # PCA for rastermap
    U = PCA(n_components=n_comp).fit_transform(S)
    model = mapping.Rastermap(
        n_clusters=100,
        n_PCs=n_comp,
        grid_upsample=10,
        n_splits=0,
        time_lag_window=7,
        ts=0.9,
    ).fit(S, normalize=False, u=U)
    return model

def get_rastermap_spks(spks, n_comp=200):
    S = spks.copy()
    mu = S.mean(1)
    sd = S.std(1)
    S = (S - mu[:, np.newaxis]) / (1e-10 + sd[:, np.newaxis])
    S -= S.mean(axis=0)
    # PCA for rastermap
    U = PCA(n_components=n_comp).fit_transform(S)
    model = mapping.Rastermap(
        n_clusters=100,
        n_PCs=n_comp,
        grid_upsample=10,
        n_splits=0,
        time_lag_window=7,
        ts=0.9,
    ).fit(S, normalize=False, u=U)
    return model

def get_init_frames_per_category(MouseObject):
    """
    This function return the frame index of the first frame of each trial

    Parameters
    ----------
    MouseObject : object
        Object containing the data of a mouse

    Returns
    -------
    init_frames_per_category : np.array
        Array containing the frame index of the first frame of each trial
    first_trialframes : int
        index of the firt trial start
    last_trialframes : int
        index of the last trial start
    """
    init_frames_per_category = np.empty((4), dtype=object)
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    first_trialframes = []
    last_trialframes = []
    categories, trial_counts = get_trial_categories(
        MouseObject._trial_info["isrewarded"], MouseObject._trial_info["istest"]
    )
    for i, cat_color in enumerate(opt_dict.items()):
        ix = np.round(MouseObject._timestamps["trial_frames"]) * (
            categories == cat_color[0]
        )
        ix = ix[ix != 0].astype(int)
        init_frames_per_category[i] = ix
        if trial_counts[cat_color[0]] != 0:
            first_trialframes.append(np.min(init_frames_per_category[i]))
            last_trialframes.append(np.max(init_frames_per_category[i]))
    first_trialframe = np.min(np.array(first_trialframes))
    last_trialframe = np.max(np.array(last_trialframes))
    return init_frames_per_category, first_trialframe, last_trialframe


def get_frametypes(MouseObject, color=True):
    """
    Return a numpy array containing the type of frame for each frame in the recording.
    The type of frame is determined by the trial type (rewarded, non rewarded, rewarded test, non rewarded test)
    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the data of interest
    color : bool, optional
        True, the output array will contain the color of the trial type, by default True
        False, the output array will contain the name of the trial type.
    Returns
    -------
    trial_type_byframe : numpy array
        Numpy array containing the type of frame for each frame in the recording
    """
    ttypebyframes = ["NaN"] * MouseObject._spks.shape[1]
    ttypebyframes = np.array(ttypebyframes, dtype=object)
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    categories, _ = get_trial_categories(
        MouseObject._trial_info["isrewarded"], MouseObject._trial_info["istest"]
    )
    for cat_color in opt_dict.items():
        ix = np.round(MouseObject._timestamps["trial_frames"]) * (
            categories == cat_color[0]
        )
        ix = ix[ix != 0].astype(int)
        if color == True:
            ttypebyframes[ix] = cat_color[1]
        else:
            ttypebyframes[ix] = cat_color[0]
    df = pd.DataFrame(ttypebyframes, columns=["trial_type"])
    df.replace("NaN", np.nan, inplace=True)
    filled = df.fillna(method="ffill")
    trial_type_byframe = filled.values.flatten()
    return trial_type_byframe

#### FRAME SELECTOR #####
def get_frameselector(MouseObject, intertrial_distance = 100, effective_frames = True):
    reward_delivery_frame = np.round(
        MouseObject._timestamps["reward_frames"][
            np.isnan(MouseObject._timestamps["reward_frames"]) == False
        ]
    ).astype(int)
    FrameSelector = pd.DataFrame(
        {
            "trial_no": get_trial_per_frame(MouseObject),
            "trial_type": get_frametypes(MouseObject, color=False),
            "contrast": MouseObject._timestamps["alpha"][: MouseObject._spks.shape[1]],
            "speed": MouseObject._timestamps["run"][: MouseObject._spks.shape[1]],
            "distance": MouseObject._timestamps["distance"][
                : MouseObject._spks.shape[1]
            ],
            "reward_delivery": np.nan,
            "intertrial": False,
            "ordinal_time": MouseObject._timestamps["frame_times"][: MouseObject._spks.shape[1]],
        }
    )
    FrameSelector.loc[reward_delivery_frame, "reward_delivery"] = "delivery"
    rewarded_trials = FrameSelector.loc[FrameSelector["trial_type"] == "rewarded"][
        "trial_no"
    ].unique()
    for trial in rewarded_trials:
        selected_trial = FrameSelector.query(f"trial_no == {trial}")
        selected_frames = selected_trial.index.values
        delivery_frame = selected_trial.query(
            "reward_delivery == 'delivery'"
        ).index.values
        if np.any(delivery_frame):
            before_delivery = selected_frames[selected_frames < delivery_frame]
            after_delivery = selected_frames[selected_frames > delivery_frame]
            FrameSelector.loc[after_delivery, "reward_delivery"] = "after"
            FrameSelector.loc[before_delivery, "reward_delivery"] = "pre"

    FrameSelector["ordinal_time"] = pd.to_datetime(
    FrameSelector["ordinal_time"].apply(
        lambda x: datetime.fromordinal(int(x))
        + timedelta(days=x % 1)
        - timedelta(days=366)
        )
    )
    FrameSelector["time_fromstart"] = (FrameSelector["ordinal_time"] - FrameSelector["ordinal_time"][0]).dt.total_seconds()
    all_trials = FrameSelector["trial_no"].unique()
    trials = all_trials[~np.isnan(all_trials)].astype(int)
    FrameSelector["time_within_trial"] = np.nan
    corridor_length = MouseObject._settings['CorridorLength_dm'].item()*10
    full_length = corridor_length + intertrial_distance
    for trial in trials:
        FrameSelector.loc[FrameSelector["trial_no"] == trial, "time_within_trial"] = FrameSelector.loc[FrameSelector["trial_no"] == trial, "time_fromstart"] - FrameSelector.loc[FrameSelector["trial_no"] == trial, "time_fromstart"].iloc[0]
        FrameSelector.loc[FrameSelector["trial_no"] == trial, "distance"] = FrameSelector.loc[FrameSelector["trial_no"] == trial, "distance"] - (full_length * (FrameSelector.loc[FrameSelector["trial_no"] == trial, "trial_no"]-1))
    FrameSelector.loc[FrameSelector["distance"] > corridor_length, "intertrial"] = True
    FrameSelector = FrameSelector.drop(columns=["ordinal_time"]) 
    if effective_frames == True:
        FrameSelector = FrameSelector.loc[~pd.isnull(FrameSelector)['trial_no']]
        FrameSelector = FrameSelector.loc[FrameSelector["trial_no"] != all_trials[-1]] #drops the last trial (which sometimes might not be complete)
    return FrameSelector


def get_neurons_bytrial(
    MouseObject,
    FrameSelector,
    rwd_condition="reward_delivery == 'pre'",
    nonrwd_condition="distance <= 57",
):
    """
    This function returns a numpy array containing the mean firing rate for the n frames meeting the conditions specified in each trial type, for each neuron.
    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the mouse data
    FrameSelector : pandas DataFrame
        DataFrame containing the the correspondences between frames and many behavorial variables
    rwd_condition : str, optional
        Condition to select the frames of rewarded trials, by default "reward_delivery == 'pre'"
    nonrwd_condition : str, optional
        Condition to select the frames of non rewarded trials, by default "distance <= 57"
    Returns
    -------
    spks_bytrial : numpy array
        Numpy array containing the mean firing rate for the n frames meeting the conditions specified in each trial type, for each neuron.
    trials : numpy array
        Numpy array containing the original (matlab) trial numbers, the spks_bytrial array is indexed from 0, but the trials in FrameSelector are indexed from 1 (from matlab)
    bad_trials : list
        List containing the trials with detected issues
    """
    all_trials = FrameSelector["trial_no"].unique()
    trials = all_trials[~np.isnan(all_trials)].astype(int)
    n_neurons = MouseObject._spks.shape[0]
    spks = MouseObject._spks.copy()
    spks = zscore(spks, axis=1)
    n_trials = len(trials)
    spks_bytrial = np.empty([n_neurons, n_trials])
    bad_trials = []
    conditions = {"rewarded": f"{rwd_condition}", "non rewarded": f"{nonrwd_condition}",  "non rewarded test": f"{nonrwd_condition}" , "rewarded test": f"{nonrwd_condition}"}
    for trial in trials:
        trial_type = (
            FrameSelector.query(f"trial_no == {trial}")["trial_type"].unique().item()
        )
        full_query = conditions[trial_type] + f" & trial_no == {trial}"
        frame_num = FrameSelector.query(full_query).index.values
        if len(frame_num) == 0:
            print(
                f"{trial_type} trial: {trial}, has no frames meeting the condition: {conditions[trial_type]}"
            )
            print("check that trial!!, filling with nan")
            spks_bytrial[:, trial - 1] = np.nan
            bad_trials.append(trial-1)
        else:
            selected_spks = spks[:, frame_num]
            spks_save = selected_spks.mean(axis=1)
            spks_bytrial[:, trial - 1] = spks_save
    return spks_bytrial, np.array(bad_trials)

def get_trialno_bytype(FrameSelector):
    seq = ("rewarded","non rewarded","rewarded test","non rewarded test")
    trial_type_dict = dict.fromkeys(seq, np.nan)
    ttypes = FrameSelector["trial_type"].unique()
    nan_mask = pd.isnull(FrameSelector["trial_type"].unique())
    ttypes = ttypes[~nan_mask]
    for trial_type in ttypes:
        trialno = FrameSelector.query(f"trial_type == '{trial_type}'")["trial_no"].unique().astype(int)
        trialno = trialno - 1
        trial_type_dict[trial_type] = trialno
    return trial_type_dict

def superneuron_toneurons(isort_vect,clust_idxs,spn_binsize):
    """
    convert superneuron index to individual neuron index

    Parameters:
    isort_vect: isort vector from rastermap
    clust_idxs: tuple of (start, end) cluster index
    spn_binsize: number of neurons per superneuron

    Returns: 
    selected_neurons: list of neuron indices
    """

    nsuper = len(isort_vect)//spn_binsize
    assert clust_idxs[1] <= nsuper, "clustidx[1] should be smaller than number of superneurons"
    assert clust_idxs[0] >= 0, "clustidx[0] should be larger than 0"
    selected_neurons = isort_vect[clust_idxs[0]*spn_binsize:(clust_idxs[1]+1)*spn_binsize]
    return selected_neurons

def interp_spks_by_corridorlength(Mouse_spks, FrameSelector, z = True, corridor_length=150):
    """
    Interpolate the spks by corridor length

    Parameters
    ----------
    Mouse_spks : Mouse object or ndarray (neurons, frames)
        Mouse object containing the spks and other information or the spks ndarray
    FrameSelector : DataFrame
        DataFrame containing the frame selector information
    z : bool
        If True, zscores the spks, by default True
    corridor_length : int
        The length of the corridor in cm

    Returns
    -------
    spks_interp_reshaped : ndarray
        The interpolated spks reshaped to the shape of (nneurons, corridor_length, ntrials)
    """

    if len(FrameSelector.query("trial_no==1 & distance == 0"))>1: #checks if there are more than one frame with distance 0 in the first trial, to choose either 
        aparent_first_frame = FrameSelector.index[0] #the first frame of the first trial
        last_frame_onzero_idx = FrameSelector.query("trial_no==1 & distance == 0").index[-1] #the first effective frame if there are more than one on the first position
        first_effective_frame_idx = last_frame_onzero_idx - aparent_first_frame #the first effective frame index
        last_effective_frame_idx = FrameSelector.index[-1]+1
        selector = FrameSelector.iloc[first_effective_frame_idx:last_effective_frame_idx]
        first_effective_frame_idx = selector.index[0] # or just the first if in fact there is only one
        last_effective_frame_idx = selector.index[-1]+1 # the last effective frame index
        selector = selector.reset_index(drop=True) #reset the index of the frame selector to match the spks, now index 0 is the first frame of the first trial
    else:
        first_effective_frame_idx = FrameSelector.index[0] # or just the first if in fact there is only one
        last_effective_frame_idx = FrameSelector.index[-1]+1 # the last effective frame index 
        selector = FrameSelector.reset_index(drop=True) #reset the index of the frame selector to match the spks, now index 0 is the first frame of the first trial
    if isinstance(Mouse_spks, np.ndarray):
        spks = Mouse_spks[:,first_effective_frame_idx:last_effective_frame_idx] #remove the frames before the first effective frame i.e, the first frame of the first trial
        if z:
            spks = zscore(spks, axis=1) #zscore the spks
    else:
        spks = Mouse_spks._spks.copy()
        spks = spks[:,first_effective_frame_idx:last_effective_frame_idx] #remove the frames before the first effective frame i.e, the first frame of the first trial
        print(first_effective_frame_idx, last_effective_frame_idx)
        if z:
            spks = zscore(spks, axis=1) #zscore the spks
    distance = selector["distance"].values
    trials = selector["trial_no"].astype(int).values
    ntrials = trials[-1]
    if distance[0] > 0:
        spks = np.insert(spks,0,spks[:,0],axis=1) #duplicate the first frame for interpolation purposes
        distance = np.insert(distance,0,0) #insert the first distance as 0
        trials = np.insert(trials,0,1) #insert the first trial as 1
    elif distance[-1] < corridor_length:
        spks = np.insert(spks,0,spks[:,-1],axis=1) #duplicate the last frame for interpolation purposes
        distance = np.insert(distance,distance.shape[0],corridor_length) #insert the last distance as corridor_length
        trials = np.insert(trials,trials.shape[0],ntrials) #insert the last trial 
    dist_interp = (distance + (corridor_length * (trials-1)))
    print(f"interpolating {spks.shape[0]} neurons, {spks.shape[1]} frames to")
    print(f"the vector of distance with shape: {dist_interp.shape}")
    x_new = np.linspace(0, corridor_length*ntrials, (corridor_length*ntrials))
    spks_interp = interp1d(dist_interp, spks, kind='linear', fill_value="extrapolate")(x_new)
    spks_interp_reshaped = spks_interp.reshape(spks.shape[0],ntrials,corridor_length)
    print(f"neurons: {spks_interp_reshaped.shape[0]}, trials: {spks_interp_reshaped.shape[1]}, corridor length: {spks_interp_reshaped.shape[2]}")
    if len(np.where(np.isnan(spks_interp_reshaped))[0]) != 0:
        print("Warning: There are NaNs in the interpolated spks")
        print(np.where(np.isnan(spks_interp_reshaped)))
    return spks_interp_reshaped 

@dataclass
class Mouse:

    name: str
    datexp: str
    blk: str
    data_path : str = "Z:/data/PROC"


    def load_behav(self, timeline_block=None, verbose=False):
        import h5py
        """
        Loads the experiment info from the database
        Parameters:
        ----------
        timeline_block : int
            Specifies the timeline block to choose
        verbose : bool
            If True, prints the timeline info
        Returns:
        ----------
        Timeline : dict
            Dictionary with the experiment info
        """

        ### This function can be ad hoc for different experiments, this is just an example for mine ###
        ## at the end, this functions should always return a timeline dict, and the data_var to sync the behav and imaging data

        if timeline_block is not None:
            blk = str(timeline_block)
            root = os.path.join(self.data_path, self.name, self.datexp, blk)
            fname = "Timeline_%s_%s_%s.mat" % (self.name, self.datexp, blk)
        else:
            root = os.path.join(self.data_path, self.name, self.datexp, self.blk)
            fname = "Timeline_%s_%s_%s.mat" % (self.name, self.datexp, self.blk)

        fnamepath = Path(os.path.join(root, fname))
        if not fnamepath.exists():
            print(f"Timeline with fname: {fname} not found, trying with fname: {self.name}_{self.datexp}_{self.blk}.mat ")
            fname = f"{self.name}_{self.datexp}_{self.blk}.mat"
            fnamepath = Path(os.path.join(root, fname))
        # old matlab file format
        try:
            matfile = io.loadmat(fnamepath, squeeze_me=True)
            self._timeline = matfile["Timeline"]
            if "data" in matfile.keys():
                self._data_var = matfile["data"]
        except NotImplementedError:
            print("Timeline file is in v7.3 format, loading with h5py")
            ## Syntax to load a .mat v7.3 file
            with h5py.File(fnamepath, "r") as f:
                timeline_group = f.get("Timeline").get("Results")
                settings_group = f.get("Timeline").get("Settings")
                data_var = np.array(
                    f.get("Timeline").get("data")
                )  # variable that syncs the timeline with the imaging data
                timeline = {k: np.array(v) for k, v in timeline_group.items()}
                settings = {k: np.array(v) for k, v in settings_group.items()}
                if 'Notes' in settings.keys():
                    settings.pop('Notes')
                self._timeline = timeline
                self._data_var = data_var
                self._settings = settings
        if verbose:
            print("###### Behavior loaded ######")
            print("Loaded timeline from: %s" % fnamepath)
            print("----------------------------------")
            print("Timeline dict created: into self._timeline")
            print(f"Timeline keys: {self._timeline.keys()}")
            print("----------------------------------")
            print("Settings dict created: into self._settings")
            print(f"Timeline keys: {self._settings.keys()}")
            print("----------------------------------")
            print("Data_var loaded:  into self._data_var")
            print("This variable is used to sync the timeline with the imaging data")


    def load_neurons_VG(self, dual_plane=True, Fs=3, return_F = False, return_iscell = False, verbose=False):
        """
        Loads the neural data from the database

        Parameters:
        ----------

        dual_plane : Boolean
            Dual plane flag indicates whether the data is from the dual plane or not
        Baseline : Boolean
            Baseline flag indicates whether the data is preproceded or not
        verbose : bool
            If True, prints the loaded info
        return_F : bool
            If True, loads the F matrix
        return_iscell : bool
            If True, loads the iscell matrix
        Returns:
        ----------
        self : Mouse
            Returns the Mouse object with the loaded data
        """

        root = os.path.join(self.data_path, self.name, self.datexp, self.blk)
        ops = np.load(
            os.path.join(root, "suite2p", "plane0", "ops.npy"), allow_pickle=True
        ).item()

        if return_iscell == True:
            is_cell = np.zeros((0,2))
            ops["nplanes"] =  20
        if dual_plane:
            tlags = np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
            tlags = np.hstack((tlags, tlags))
            tlags = tlags.flatten()
        else:
            tlags = np.linspace(0.2, -0.8, ops["nplanes"] + 1)[:-1]
        print(f"planes: {tlags.shape[0]}")

        spks = np.zeros((0, ops["nframes"]), np.float32)
        F_ret = np.zeros((0, ops["nframes"]), np.float32)
        snr_ret = np.zeros((0,))
        stat = np.zeros((0,))
        iplane = np.zeros((0,))
        xpos, ypos = np.zeros((0,)), np.zeros((0,))

        for n in tqdm(range(ops["nplanes"])):
            ops = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"),
                allow_pickle=True,
            ).item()

            stat0 = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"),
                allow_pickle=True,
            )

            ypos0 = np.array([stat0[n]["med"][0] for n in range(len(stat0))])
            xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

            if "dy" in ops:
                ypos0 += ops["dy"]
                xpos0 += ops["dx"]

            F0 = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))
            Fneu = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))
            F0 = F0 - 1*Fneu
            snr = 1  - .5 * np.var(np.diff(F0, axis=1), axis=1) / np.var(F0, axis=1)
            F0 = baselining(ops, tlags[n], F0)
            spks0 = udcnv.apply(F0, Fs, "C:/Users/labadmin/Documents/category-neural/data/sim_right_flex.th", batch_size=1)
            #spks0 = dcnv.oasis(F0, ops["batch_size"], ops["tau"], ops["fs"])
            F_ret = np.concatenate((F_ret, F0.astype("float32")), axis=0)


            spks0 = spks0.astype("float32")
            if spks.shape[1] > spks0.shape[0]:
                spks0 = np.concatenate(
                    (spks0, np.zeros((spks0.shape[0], spks.shape[1] - spks0.shape[1]))),
                    axis=1,
                )
            if return_iscell:
                is_cell0 = np.load(
                    os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"),
                    allow_pickle=True,
                )
                is_cell = np.concatenate((is_cell, is_cell0), axis=0)
            snr_ret = np.concatenate((snr_ret, snr.astype("float32")), axis=0)
            spks = np.concatenate((spks, spks0), axis=0)
            ypos = np.concatenate((ypos, ypos0), axis=0)
            xpos = np.concatenate((xpos, xpos0), axis=0)
            iplane = np.concatenate(
                (
                    iplane,
                    n
                    * np.ones(
                        len(stat0),
                    ),
                )
            )
            stat = np.concatenate((stat, stat0), axis=0)
        self._spks = spks
        self._ypos = ypos
        self._xpos = xpos
        self._iplane = iplane
        self._stat = stat
        self._ops = ops
        self._snr = snr_ret
        if return_F == True:
            self._F = F_ret
        if return_iscell == True:
            self._is_cell = is_cell
        if verbose:
            print("###### Neurons loaded ######")
            print(f"Total neurons loaded: {len(spks)}")
            print("---------------------------")
            print(
                f"Spikes created at: self.spks, with shape: {spks.shape} : (neurons, frames)"
            )
            print("Neurons plane information created at: self.iplane")
            print("Neurons positions created at: self.xpos, self.ypos")
            print("Neurons snr created at: self.snr")
            print("Suite2p stats created at: self.stat")
            print("Suite2p options created at: self.ops")
            print("---------------------------")

    def load_neurons(self, dual_plane=True, baseline=True, return_F = False, return_iscell = False, verbose=False):
        """
        Loads the neural data from the database

        Parameters:
        ----------

        dual_plane : Boolean
            Dual plane flag indicates whether the data is from the dual plane or not
        Baseline : Boolean
            Baseline flag indicates whether the data is preproceded or not
        verbose : bool
            If True, prints the loaded info
        return_F : bool
            If True, loads the F matrix
        return_iscell : bool
            If True, loads the iscell matrix
        Returns:
        ----------
        self : Mouse
            Returns the Mouse object with the loaded data
        """

        root = os.path.join(self.data_path, self.name, self.datexp, self.blk)
        ops = np.load(
            os.path.join(root, "suite2p", "plane0", "ops.npy"), allow_pickle=True
        ).item()

        if return_iscell == True:
            is_cell = np.zeros((0,2))
            ops["nplanes"] =  20
        if dual_plane:
            tlags = np.linspace(0.2, -0.8, ops["nplanes"] // 2 + 1)[:-1]
            tlags = np.hstack((tlags, tlags))
            tlags = tlags.flatten()
        else:
            tlags = np.linspace(0.2, -0.8, ops["nplanes"] + 1)[:-1]
        print(f"planes: {tlags.shape[0]}")

        spks = np.zeros((0, ops["nframes"]), np.float32)
        F_ret = np.zeros((0, ops["nframes"]), np.float32)
        stat = np.zeros((0,))
        iplane = np.zeros((0,))
        xpos, ypos = np.zeros((0,)), np.zeros((0,))

        for n in tqdm(range(ops["nplanes"])):
            ops = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"),
                allow_pickle=True,
            ).item()

            stat0 = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "stat.npy"),
                allow_pickle=True,
            )
            ypos0 = np.array([stat0[n]["med"][0] for n in range(len(stat0))])
            xpos0 = np.array([stat0[n]["med"][1] for n in range(len(stat0))])

            if "dy" in ops:
                ypos0 += ops["dy"]
                xpos0 += ops["dx"]

            if baseline:
                F = np.load(os.path.join(root, "suite2p", "plane%d" % n, "F.npy"))
                Fneu = np.load(os.path.join(root, "suite2p", "plane%d" % n, "Fneu.npy"))
                F = baselining(ops, tlags[n], F, Fneu)
                spks0 = dcnv.oasis(F, ops["batch_size"], ops["tau"], ops["fs"])
                if return_F:
                    F_ret = np.concatenate((F_ret, F.astype("float32")), axis=0)
            else:
                spks0 = np.load(
                    os.path.join(root, "suite2p", "plane%d" % n, "spks.npy"),
                    allow_pickle=True,
                )

            spks0 = spks0.astype("float32")
            if spks.shape[1] > spks0.shape[0]:
                spks0 = np.concatenate(
                    (spks0, np.zeros((spks0.shape[0], spks.shape[1] - spks0.shape[1]))),
                    axis=1,
                )
            if return_iscell:
                is_cell0 = np.load(
                    os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"),
                    allow_pickle=True,
                )
                is_cell = np.concatenate((is_cell, is_cell0), axis=0)
            spks = np.concatenate((spks, spks0.astype("float32")), axis=0)
            ypos = np.concatenate((ypos, ypos0), axis=0)
            xpos = np.concatenate((xpos, xpos0), axis=0)
            iplane = np.concatenate(
                (
                    iplane,
                    n
                    * np.ones(
                        len(stat0),
                    ),
                )
            )
            stat = np.concatenate((stat, stat0), axis=0)
        self._spks = spks
        self._ypos = ypos
        self._xpos = xpos
        self._iplane = iplane
        self._stat = stat
        self._ops = ops
        if return_F == True:
            self._F = F_ret
        if return_iscell == True:
            self._is_cell = is_cell
        if verbose:
            print("###### Neurons loaded ######")
            print(f"Total neurons loaded: {len(spks)}")
            print("---------------------------")
            print(
                f"Spikes created at: self.spks, with shape: {spks.shape} : (neurons, frames)"
            )
            print("Neurons plane information created at: self.iplane")
            print("Neurons positions created at: self.xpos, self.ypos")
            print("Suite2p stats created at: self.stat")
            print("Suite2p options created at: self.ops")
            print("---------------------------")

    def get_timestamps(self, intertrial_distance=100, verbose=False):
        """
        Creates the timestamps of behavior in terms of frames

        ******
        the syntax of this function will change for diferent experiments, since the timeline variable might have different named variables,
        but in escence it is the same process.
        ******
        Parameters:
        verbose : Boolean
            Verbose flag indicates whether to print timestamps info or not
        -------
        Returns:
        -------
        timestamps : dict
            Dictionary with the timestamps of the behavior in terms of frames
        """
        try:
            # Gets the time for each neural frame
            frames_time = self._data_var[0]
            ixx = (frames_time[:-1] > 2.5) * (frames_time[1:] < 2.5)
            iframes = np.argwhere(ixx).flatten()
            isamp = np.argwhere(self._data_var[1] > 1).flatten()
            ts = self._data_var[1][isamp]
            tframes = interp1d(isamp, ts)(iframes)
            nframes = len(tframes)
        except NameError:
            print(
                "data_var or _timeline not loaded, make sure to load them first by self.load_behav()"
            )

        # what frame number does each trial start on? 
        ttrial = self._timeline["Movement"][4]
        istart = np.argwhere(np.diff(ttrial) > 0.5).flatten() + 1
        frameidx_first_trial = np.where(self._timeline["Movement"][5] > tframes.min())[0][0]
        istart = np.insert(istart,0,frameidx_first_trial) # insert frameidx_first_trial in first position
        tstart = self._timeline["Movement"][5][istart]

        # get trial start frames
        trial_frames = interp1d(tframes, np.arange(1, nframes + 1))(
            tstart
        )  
        df = get_movement_df(self, intertrial_distance = intertrial_distance)
        # interpolate running speed for each neural frame
        #runsp = df['speed'].values
        runsp = (df['pitch'] ** 2 + df['roll'] ** 2) ** 0.5
        trun = self._timeline["Movement"][5]
        run = interp1d(trun, runsp, fill_value="extrapolate")(tframes)

        # interpolate running speed for each neural frame
        distance = df["distance_interp"].values
        trun = self._timeline["Movement"][5]
        dist = interp1d(trun, distance, fill_value="extrapolate")(tframes)

        # interpolate alpha values for each neural frame
        alpha = df.alpha.values
        trun = self._timeline["Movement"][5]
        alpha_interp = interp1d(trun, alpha, fill_value="extrapolate")(tframes)

        # get lick times as frame numbers
        tlick = self._timeline["Licks"][4]
        ix = self._timeline["Licks"][5] < 0.5
        tlick = tlick[ix]
        frame_lick = interp1d(tframes, np.arange(1, nframes + 1), fill_value="extrapolate")(tlick)
        # get the reward delivery as frame numbers
        treward = self._timeline["RewardInfo"][1]
        frame_reward = interp1d(tframes, np.arange(1, nframes + 1), fill_value="extrapolate")(treward)

        timestamps = {
            "trial_frames": trial_frames,
            "run": run,
            "distance": dist,
            "alpha": alpha_interp,
            "lick_frames": frame_lick,
            "reward_frames": frame_reward,
            "frame_times": tframes,
        }
        self._timestamps = timestamps
        if verbose:
            print("###### Timestamps created ######")
            print("---------------------------")
            print('Trial frames created at: self._timestamps["trial_frames"]')
            print('Interpolated run variable created at: self._timestamps["run"]')
            print(
                'Interpolated distance variable created at: self._timestamps["distance"]'
            )
            print('Interpolated alpha variable created at: self._timestamps["alpha"]')
            print(
                'Licks in terms of frames created at: self._timestamps["lick_frames"]'
            )
            print(
                'Rewards in terms of frames created at: self._timestamps["reward_frames"]'
            )
            print('Frame times created at: self._timestamps["frame_times"]')
            print("---------------------------")

    def get_trial_info(self, verbose=False):
        """
        Creates the trial info information vectors, indicating the trial type
        Parameters:
        -------
        verbose : Boolean
            Verbose flag indicates whether to print trial type info or not
        Returns:
        -------
        trial_info : dict
            Dictionary with the trial type information
        """
        try:
            ttrial = self._timeline["Movement"][4]
            istart = np.argwhere(np.diff(ttrial) > 0.5).flatten() + 1
            frameidx_first_trial = np.where(self._timeline["Movement"][5] > self._timestamps["frame_times"].min())[0][0]
            istart = np.insert(istart,0,frameidx_first_trial) # insert frameidx_first_trial in first position
            tstart = self._timeline["Movement"][5][istart]
        except NameError:
            print(
                "data_var, _timeline or _timestamps not loaded, make sure to load them first by self.load_behav() and self.get_timestamps()"
            )

        # get the trial type for each trial
        ntrials = len(tstart)
        trial_type = self._timeline["TrialRewardStrct"].flatten()[:ntrials]
        trial_new = self._timeline["TrialNewTextureStrct"].flatten()[:ntrials]
        trial_info = {
            "isrewarded": trial_type,
            "istest": trial_new,
        }

        self._trial_info = trial_info
        if verbose:
            print("###### Trial informartion dict created ######")
            print("---------------------------")
            print(
                'Boolean array describing if the trial is rewarded created at: self._trial_info["isrewarded"]'
            )
            print(
                'Boolean array describing if the trial was a test trial created at: self._trial_info["istest"]'
            )
            print("---------------------------")

def objectify(objdict):
    """
    create object from a dictionary
    
    Parameters:
    -------
    objdict : dict
        Dictionary with the object attributes
    """
    name = objdict['name']
    date = objdict['datexp']
    blk = objdict['blk']
    mouse = Mouse(name,date,blk)
    for key in objdict.keys():
        setattr(mouse, key, objdict[key])
    return mouse


def load_mouse(name: str, date: str, block: str, data_path: str = "Z:/data/PROC",  mdl_path: str = "C:/Users/labadmin/Documents/models/mouseobj", ret_path = "D:/retinotopy/aligned_xy", **kwargs):
    """
    Checks if a local copy of the mouse object exists, if not it creates it, if it exists, it asks if you want to load it or create a new one.
    Parameters:
    -------
    name : str
        Name of the mouse
    date : str
        Date of the experiment
    block : str
        Block of the experiment
    data_path : str
        Path to the data (usually is a path to DM11)
    mdl_path : str
        Path to the models folder (usually is a local folder in your computer)
    return_iscell: Boolean
        if True loads the iscell matrix
    return_F: Boolean
        if True loads the F matrix
    kwargs: dict
        dictionary with the arguments of the low level upload functions:
        - load_neurons: Boolean, if True loads the raw spike data
        - interp_behav: Boolean, if True interpolates the behavior data to neural frames
        - intertrial_distance: float, distance in cm between trials
        - timeline_block: int, block of the timeline to load, this needs to be specified if the timeline is not in the block defined in the object
    Returns:
    -------
    mouse : Mouse object
        Mouse object with the data loaded
    """
    
    def build_mouse(name: str, date: str, block: str, data_path: str, **kwargs):
        """
        Builds a mouse object from the data path, by default it only loads the behavior data, 
        other attributes can be loaded by passing kwargs to the function.

        Parameters:
        -------
        name : str
            Name of the mouse
        date : str
            Date of the experiment
        block : str
            Block of the experiment
        data_path : str
            Path to the data (usually is a path to DM11)
        kwargs: dict
            dictionary with the arguments of the low level upload functions:
            - load_neurons: Boolean, if True loads the raw spike data
            - interp_behav: Boolean, if True interpolates the behavior data to neural frames
            - intertrial_distance: float, distance in cm between trials
            - return_F: Boolean, if True loads the F matrix to the mouse object
            - timeline_block: int, block of the timeline to load, this needs to be specified if the timeline is not in the block defined in the object
            - load_retinotopy: Boolena, if True loads the retinotopy data
            - ret_path: str, path to the retinotopy data
            - dual_plane: Boolean, if True loads the dual plane data
        Returns:
        -------
        mouse : Mouse
            Mouse object with the data loaded
        """
        mouse = Mouse(name, date, block, data_path)
        if 'timeline_block' in kwargs.keys():
            mouse.load_behav(timeline_block=kwargs['timeline_block'])
        else:
            mouse.load_behav()
        if ('interp_behav' in kwargs):
            if kwargs['interp_behav'] == True:
                if 'intertrial_distance' in kwargs:
                    mouse.get_timestamps(intertrial_distance=kwargs['intertrial_distance'])
                    mouse.get_trial_info()
                else:
                    mouse.get_timestamps() # default intertrial distance is 100 cm
                    mouse.get_trial_info()
        if ('load_neurons' in kwargs):
            if kwargs['load_neurons'] == True:
                if 'return_F' in kwargs:
                    if 'dual_plane' in kwargs:
                        if 'return_iscell' in kwargs:
                            mouse.load_neurons_VG(dual_plane=kwargs['dual_plane'], 
                                               return_F=kwargs['return_F'], return_iscell=kwargs['return_iscell'])
                        else:
                            mouse.load_neurons_VG(dual_plane=kwargs['dual_plane'],
                                               return_F=kwargs['return_F'])
                    else:
                        if 'return_iscell' in kwargs:
                            mouse.load_neurons_VG(dual_plane=True, return_F=kwargs['return_F'],
                                                return_iscell=kwargs['return_iscell'])
                        else:
                            mouse.load_neurons_VG(dual_plane=True, return_F=kwargs['return_F'])
                else:
                    if 'dual_plane' in kwargs:
                        if 'return_iscell' in kwargs:
                            mouse.load_neurons_VG(dual_plane=kwargs['dual_plane'],
                                               return_iscell=kwargs['return_iscell'])
                        else:
                            mouse.load_neurons_VG(dual_plane=kwargs['dual_plane'])
                    else:
                        if 'return_iscell' in kwargs:
                            mouse.load_neurons_VG(dual_plane=True, return_iscell=kwargs['return_iscell'])
                        else:
                            mouse.load_neurons_VG(dual_plane=True)
        if ('load_retinotopy' in kwargs):
            if kwargs['load_retinotopy'] == True:
                p = Path(ret_path).glob('**/*')
                n = "%s_%s" % (name, date)
                rtpy_file = [x for x in p if x.is_file() if n in x.name][0]
                #rtpy_file = Path(rtpy_file)
                print(f"Loading retinotopy data from {rtpy_file}")
                for file in np.load(rtpy_file, allow_pickle = True).files:
                    exec(f"mouse.{file} = np.load(rtpy_file, allow_pickle = True)['{file}']")
        return mouse


    print("Checking if model object exists ...")
    directory = f"{name}/{date}/{block}"
    full_pth = Path(
        os.path.join(Path(mdl_path), directory)
    )
    file = f"{name}_{date}_{block}.pkl"
    file_pth = full_pth.joinpath(file)
    if file_pth.is_file() == False:
        print("*************************************")
        print("Model object does not exist in path:")
        print(f"{full_pth}")
        print("Creating new mouse object ...")
        mouse = build_mouse(name , date, block, data_path, **kwargs)
        print("*************************************")
        print("mouse object created with the following attributes:")
        print(properties(mouse))
        return mouse
    else:
        print(f"Loading mouse object from {full_pth}")
        with open(file_pth, "rb") as file:
            mouse = pickle.load(file)
            #mouse = objectify(mouse)
            print("Existing mouse object has the following attributes:")
            print(properties(mouse))
            print("*************************************")
            inp = input("You want to load the saved object? (Y/N)")
            if inp.lower() == "y":
                print("Mouse object loaded from local path")
                return mouse
            else:
                print("Creating new mouse object ...")
                mouse = build_mouse(name , date, block, data_path, **kwargs)
                print("*************************************")
                print("mouse object created with the following attributes:")
                print(properties(mouse))
                return mouse

def save_mouse(MouseObject: object, compressed = True, n_comps: int = 1000, mdl_path: str = "C:/Users/labadmin/Documents/models/mouseobj"):
    name = MouseObject.name
    date = MouseObject.datexp
    block = MouseObject.blk
    directory = f"{name}/{date}/{block}"
    full_pth = Path(
        os.path.join(Path(mdl_path), directory)
    )
    file = f"{name}_{date}_{block}.pkl"
    file_pth = full_pth.joinpath(file)
    if full_pth.is_dir() == False:
        print(f"Creating directory {full_pth}")
        full_pth.mkdir(parents=True)
    with open(file_pth, "wb") as file:
        if ('_spks' in properties(MouseObject)) & (compressed == True):
                spks = MouseObject._spks.copy()
                spks = zscore(spks, axis=1)
                print("Compressing spike data ...")
                SVD_model = TruncatedSVD(n_components=n_comps).fit(spks.T)
                print("Spike data compressed")
                print("**********************")
                MouseObject._V = SVD_model.components_ @ spks
                MouseObject._U = SVD_model.components_
                delattr(MouseObject, '_spks')
                #objdct = MouseObject.serialize()
                pickle.dump(MouseObject, file, pickle.HIGHEST_PROTOCOL)
        else:
            #objdct = MouseObject.serialize()
            pickle.dump(MouseObject, file, pickle.HIGHEST_PROTOCOL)
    print(f"Mouse object saved to {full_pth}")

def get_dprime_trial(MouseObject, trial_dict, trialtypes: tuple = ('rewarded','non rewarded'), tsh: float = 95, discrimination_region: tuple = None, corridor_length: int = None, plane: int= 1, iarea: np.array = None ):
    """
    Calculate dprime for a given mouse object, trial dictionary and trial types, the discrimination region is the boundary of the corridor to be used for the calculation

    Args:
        MouseObject (MouseObject): MouseObject
        trial_dict (dict): trial dictionary
        trialtypes (tuple, optional): trial types. Defaults to ('rewarded','non rewarded').
        tsh (float, optional): if lower tan 1, it sets a treshold to the dprime distribution, if bigger than 1 sets the tsh to that percentile. Defaults to 95.
        discrimination_region (tuple of ints, optional): corridor region boundaries . Defaults to None.
        corridor_length (int, optional): corridor length. Defaults to None.
        plane (int, optional): plane to use for the calculation. Defaults to 1.
        iarea (np.array, optional): area mask. Defaults to None.

    Returns:
    --------
        dp (np.array): dprime values
        prefer_r (np.array): rewarded trials prefering neurons for the given plane - area
        prefer_nr (np.array): non-rewarded trials prefering neurons for the given plane - area
        selected_neurons (np.array): all neurons for the given plane - area

    """
    if ('interp_spks' not in properties(MouseObject)):
        if ('frameselector' not in properties(MouseObject)):
            MouseObject.frameselector = get_frameselector(MouseObject, effective_frames=True)
            MouseObject.interp_spks = interp_spks_by_corridorlength(MouseObject, MouseObject.frameselector, corridor_length=corridor_length)
        else:
            MouseObject.interp_spks = interp_spks_by_corridorlength(MouseObject, MouseObject.frameselector, corridor_length=corridor_length)
    if discrimination_region is None:
        resp = MouseObject.interp_spks.mean(-1) # mean all positions
    else:
        resp = MouseObject.interp_spks[:,:,discrimination_region[0]:discrimination_region[1]].mean(-1)

    train_trial_r = trial_dict[trialtypes[0]][::2]
    train_trial_nr = trial_dict[trialtypes[1]][::2] # only even trials for training
    # filtering for train trials
    resp = MouseObject.interp_spks # added
    r1 = resp[:, train_trial_r]
    r2 = resp[:, train_trial_nr]
    if plane == 1:
        if iarea is not None:
            selected_neurons = (MouseObject._iplane >= 10) * iarea
            print(f"total neurons in given area: {iarea.sum()}")
            print(f"total neurons in given area and plane: {selected_neurons.sum()}")
            r1 = r1[selected_neurons]
            r2 = r2[selected_neurons]
            dp = 2 * (r1.mean(1) - r2.mean(1)) / (r1.std(1) + r2.std(1))
            #dp_tsh = dp[:,discrimination_region[0]:discrimination_region[1]].mean(1)
            if tsh > 1:
                pstv_tsh = np.percentile(dp, tsh) #positive threshold
                ngtv_tsh = np.percentile(-dp, tsh)
            else:
                pstv_tsh = tsh
                ngtv_tsh = tsh
            prefer_r = (dp>pstv_tsh) 
            prefer_nr = (-dp>ngtv_tsh) 
            print(f"total neurons in given area and plane prefering rewarded trials: {prefer_r.sum()}")
            print(f"total neurons in given area and plane prefering non-rewarded trials: {prefer_nr.sum()}")
        else:
            selected_neurons = (MouseObject._iplane >= 10)
            print(f"total neurons in given area and plane: {selected_neurons.sum()}")
            r1 = r1[selected_neurons]
            r2 = r2[selected_neurons]
            dp = 2 * (r1.mean(1) - r2.mean(1)) / (r1.std(1) + r2.std(1))
            if tsh > 1:
                pstv_tsh = np.percentile(dp, tsh) #positive threshold
                ngtv_tsh = np.percentile(-dp, tsh)
            else:
                pstv_tsh = tsh
                ngtv_tsh = tsh
            prefer_r = (dp>pstv_tsh) 
            prefer_nr = (-dp>ngtv_tsh) 
            print(f"total neurons in given area and plane prefering rewarded trials: {prefer_r.sum()}")
            print(f"total neurons in given area and plane prefering non-rewarded trials: {prefer_nr.sum()}")
    elif plane == 2:
        if iarea is not None:
            selected_neurons = (MouseObject._iplane < 10) * iarea
            print(f"total neurons in given area: {iarea.sum()}")
            print(f"total neurons in given area and plane: {selected_neurons.sum()}")
            r1 = r1[selected_neurons]
            r2 = r2[selected_neurons]
            dp = 2 * (r1.mean(1) - r2.mean(1)) / (r1.std(1) + r2.std(1))
            if tsh > 1:
                pstv_tsh = np.percentile(dp, tsh) #positive threshold
                ngtv_tsh = np.percentile(-dp, tsh)
            else:
                pstv_tsh = tsh
                ngtv_tsh = tsh
            prefer_r = (dp>pstv_tsh) 
            prefer_nr = (-dp>ngtv_tsh)
            print(f"total neurons in given area and plane prefering rewarded trials: {prefer_r.sum()}")
            print(f"total neurons in given area and plane prefering non-rewarded trials: {prefer_nr.sum()}") 
        else:
            selected_neurons = (MouseObject._iplane < 10)
            print(f"total neurons in given area and plane: {selected_neurons.sum()}")
            r1 = r1[selected_neurons]
            r2 = r2[selected_neurons]
            dp = 2 * (r1.mean(1) - r2.mean(1)) / (r1.std(1) + r2.std(1))
            if tsh > 1:
                pstv_tsh = np.percentile(dp, tsh) #positive threshold
                ngtv_tsh = np.percentile(-dp, tsh)
            else:
                pstv_tsh = tsh
                ngtv_tsh = tsh
            prefer_r = (dp>pstv_tsh) 
            prefer_nr = (-dp>ngtv_tsh)
            print(f"total neurons in given area and plane prefering rewarded trials: {prefer_r.sum()}")
            print(f"total neurons in given area and plane prefering non-rewarded trials: {prefer_nr.sum()}") 
    else:
        raise ValueError('plane must be 1 or 2')
    return dp, prefer_r, prefer_nr, selected_neurons

def get_region_idx(iarea, region):
    """ 
    returns the index of the neurons in the specified region

    Args:
        iarea (np.array): array with the region index of each neuron
        region (str): region name
    
    Returns:
        ix (np.array): index of the neurons in the specified region
    """
    if region == 'all':
        ix = np.isin(iarea, np.arange(10)) 
    if region == 'V1':
        ix = np.isin(iarea, [8]) 
    if region == 'medial':
        ix = np.isin(iarea, [0,1,2,9]) 
    if region == 'anterior':
        ix = np.isin(iarea, [3,4]) 
    if region == 'lateral':
        ix = np.isin(iarea, [5,6,7]) 
    return ix

def sortbypeak(interp_spks):
    """ 
    sorts neurons by peak position

    args:
        interp_spks (np.array): interpolated spikes (neurons x trials x positions)
    returns:
        sorted_idx (np.array): sorted index
    """
    n_neurons = interp_spks.shape[0]
    response = interp_spks.mean(1) # mean all trials
    maxval = np.max(response, axis=1, keepdims=True)
    maxpos = np.argmax(response, axis=1)
    neu_ord = np.argsort(maxpos)
    return neu_ord, maxpos, maxval

def filter_byplanes(MouseObject):
    """
    Returns the idx of the deep and superficial layer neurons

    Args:
    MouseObject (object): Object of class MouseObject from src.utils
    
    Returns:
    --------
    deep_layer (np.array): idx of deep layer neurons
    superficial_layer (np.array): idx of superficial layer neurons
    """
    iplane = MouseObject._iplane
    deep_layer = iplane<10
    superficial_layer = iplane>=10
    return deep_layer, superficial_layer

def load_behaviour_retinotopy(MouseObject, RETINOTOPY_PATH = "D:/retinotopy/aligned_xy"):
    """ 
    Loads the aligned xy data of the task to the retinotopy data

    Args:
    MouseObject (object): Object of class MouseObject from src.utils
    RETINOTOPY_PATH (str): path to the retinotopy data

    Returns:
    --------
    xy_t (np.array): aligned xy data of the task
    iarea (np.array): brain area of each neuron
    iregion (np.array): brain region of each neuron
    """
    p = Path(RETINOTOPY_PATH).glob('**/*')
    name = MouseObject.name
    date = MouseObject.datexp
    n = "%s_%s" % (name, date)
    rtpy_file = [x for x in p if x.is_file() if n in x.name][0]
    #print(f"Loading retinotopy data from {rtpy_file}")
    ret_files = np.load(rtpy_file, allow_pickle = True)
    xy_t = ret_files['xy_t']
    iarea = ret_files['iarea']
    iregion = ret_files['iregion']
    return xy_t, iarea, iregion

def filterneurons(MouseObject: object, layer: int = 1, trial_type: bool = False, region: str = None,  **kwargs):
    """
    Filters neurons based on trial type, region and layer

    Args:
    MouseObject (object): Object of class MouseObject from src.utils
    region (str): region to filter neurons
    trial_type (bool): flag to filter neurons based on trial type
    layerwise (bool): flag to filter neurons based on layer
    **kwargs: arguments to be passed to get_dprime_trial function and get_region function
        types (tuple): trial types to filter neurons
        corridor_region (tuple): corridor region to filter neurons
        percentile (int): percentile for dprime calculation
        corridor_length (int): corridor length
        iarea (np.array): brain area array of each neuron
    Returns:
    --------
    trial_type_idxs (np.array): idx of neurons that prefer each trial type
    layer_idxs (np.array): idx of neurons in each layer
    region_idxs (np.array): idx of neurons in the selected region
    """
    if region is not None:
        assert region in ['all', 'V1', 'medial', 'anterior', 'lateral'], "Please provide a valid region"
        assert 'iarea' in kwargs, "Please provide iarea"
        region_idxs = get_region_idx(kwargs['iarea'], region)
    else:
        region_idxs = np.nan
    if trial_type:
        n_keys = 0
        for key in kwargs.keys():
            if key in ['types','corridor_region']:
                n_keys += 1
        assert n_keys == 2, "Please provide trial types and corridor region to descriminite neurons"
        if ('frameselector' not in properties(MouseObject)):
            frame_selector = get_frameselector(MouseObject)
            trial_no = get_trialno_bytype(frame_selector)
        else:
            trial_no = get_trialno_bytype(MouseObject.frameselector)
        _, prefer_r, prefer_nr = get_dprime_trial(MouseObject, trial_no, trialtypes = kwargs['types'], percentile=kwargs['percentile'], 
                                                  discrimination_region=kwargs['corridor_region'], corridor_length=kwargs['corridor_length'], plane = layer, iarea = region_idxs)
        trial_type_idxs = [prefer_r, prefer_nr]
    else:
        prefer_nr = np.nan
        prefer_r = np.nan
    return trial_type_idxs, region_idxs


#def compute_dprime(MouseObject, corridor_length: int = 400, nogray: bool = False):

#    """
#    Calculate dprime for a given mouse object

#    Args:
#    -------
#        MouseObject (MouseObject): MouseObject
#        corridor_length (int, optional): corridor length. Defaults to None
#        nogray (bool, optional): if True, subtracts the gray response from the interpolated response. Defaults to False
#    Returns:
#    --------
#        Mouse.Object.train_dp (np.array): dprime values (neurons x positions)
#        MouseObject.projected_response (np.array): interpolated response divided by the pool of stds of the odd rew and nrew trials (neurons x trials x positions)
#    """

#    # checks if the object contains the spks interpolated and the frameselector properties, if don't creates them
#    if ('interp_spks' not in properties(MouseObject)):
#        if ('frameselector' not in properties(MouseObject)):
#            MouseObject.frameselector = get_frameselector(MouseObject, effective_frames=True)
#            MouseObject.interp_spks = interp_spks_by_corridorlength(MouseObject, MouseObject.frameselector, corridor_length=corridor_length)
#        else:
#            MouseObject.interp_spks = interp_spks_by_corridorlength(MouseObject, MouseObject.frameselector, corridor_length=corridor_length)
    # get trial numbers by trial type
#    trial_dict = get_trialno_bytype(MouseObject.frameselector)
    # create the "train" trial numbers for rew and nrew trials
#    train_trial_r = trial_dict["rewarded"][::2]
#    train_trial_nr = trial_dict["non rewarded"][::2]
    #get the responses from those trials
#    if nogray:
#        gray_response = MouseObject.interp_spks[:, :, 300:400] # neurons, trials, positions (interp_spks is not zscored)
#        avg_gray_response = gray_response.mean(2, keepdims=True) # avg over gray
#        no_gray = MouseObject.interp_spks - avg_gray_response # subtract avg gray response
#        r1 = no_gray[:, train_trial_r, :]
#        r2 = no_gray[:, train_trial_nr, :]
#    else:
#        r1 = MouseObject.interp_spks[:, train_trial_r, :]
#        r2 = MouseObject.interp_spks[:, train_trial_nr, :]

    # collect means and stds
#    mu1 = r1.mean(1, keepdims=True)
#    mu2 = r2.mean(1, keepdims=True)
#    std1 = r1.std(1, keepdims=True) + np.finfo(np.float64).tiny
#    std2 = r2.std(1, keepdims=True) + np.finfo(np.float64).tiny
    #compute the train dprime
#    MouseObject.train_dp = 2 * ((mu1 - mu2) / (std1 + std2))
#    MouseObject.train_dp = np.squeeze(MouseObject.train_dp)
#    print("dprime saved in MouseObject.train_dp (neurons x positions) using even trials")

#def get_projected_response(MouseObject, area_layer, discrimination_region: tuple = (150,250), usetrain: bool = False):
#    # get trial numbers by trial type
#    trial_dict = get_trialno_bytype(MouseObject.frameselector)
#    # create the "train" trial numbers for rew and nrew trials
#    if usetrain:
#        t = "even"
#        r_trial = trial_dict["rewarded"][::2]
#        nr_trial = trial_dict["non rewarded"][::2]
#    else:
#        t = "odd"
#        r_trial = trial_dict["rewarded"][1::2]
#        nr_trial = trial_dict["non rewarded"][1::2]
#    #get the responses from those trials
#    r1 = MouseObject.interp_spks[:, r_trial, :]
#    r2 = MouseObject.interp_spks[:, nr_trial, :]
#    r1 = r1[area_layer]
#    r2 = r2[area_layer]
#    r1 = r1[:, :, discrimination_region[0]:discrimination_region[1]].mean(-1, keepdims=True)
#    r2 = r2[:, :, discrimination_region[0]:discrimination_region[1]].mean(-1, keepdims=True)
#    std1 = r1.std(1, keepdims=True) + np.finfo(np.float64).tiny
#    std2 = r2.std(1, keepdims=True) + np.finfo(np.float64).tiny
#    MouseObject.projected_response = 2 * (MouseObject.interp_spks[area_layer] / (std1 + std2))
#    print(f"projected response is the interpolated response divided by the pool of stds of the {t} rew and nrew trials")
#    print("saved in MouseObject.projected_response (neurons x trials x positions)")



#def filter_neurons(MouseObject, discrimination_region: tuple = (150,250), corridor_length: int = None, area: str = 'all', plane: int = 1, tsh: float = 95):
    """
    Calculate dprime for a given mouse object, trial dictionary and trial types, the discrimination region is the boundary of the corridor to be used for the calculation

    Args:
        MouseObject (MouseObject): MouseObject
        tsh (float, optional): sets the tsh to that percentile. Defaults to 95.
        discrimination_region (tuple of ints, optional): corridor region boundaries . Defaults to None.
        corridor_length (int, optional): corridor length. Defaults to None.
        plane (int, optional): plane to use for the calculation. Defaults to 1.
        iarea (str, optional): area mask. Defaults to None.

    Returns:
    --------
        dp (np.array): dprime values
        prefer_r (np.array): rewarded trials prefering neurons for the given plane - area
        prefer_nr (np.array): non-rewarded trials prefering neurons for the given plane - area
        selected_neurons (np.array): all neurons for the given plane - area

    """
    # checks if the object contains the spks interpolated and the frameselector properties, if don't creates them
    if ('train_dp' not in properties(MouseObject)):
        compute_dprime(MouseObject, corridor_length = corridor_length)   
    # filtering neurons per area / plane
    region = get_region_idx(MouseObject.iarea, area) # get neurons in the area 
    assert plane == 1 or plane == 2, "plane must be 1 or 2"
    p = (MouseObject._iplane < 10) if plane == 2 else (MouseObject._iplane >= 10) # get neurons in the plane
    area_layer_neurons = p * region #gets neurons in the area-plane
    # get the avg dprime values for the given discrimination region
    dp_position = MouseObject.train_dp[:,discrimination_region[0]:discrimination_region[1]].mean(1)
    # get the neurons the dprime distribution for the target area-layer population
    dp_position = dp_position[area_layer_neurons]
    # get the threshold for the dprime distribution
    if tsh > 1: # in the case we want to use a percentile
        pstv_tsh = np.percentile(dp_position, tsh) #positive threshold
        ngtv_tsh = np.percentile(dp_position, 100-tsh)
    else: # in the case we want to use a fixed value (needs to be a number lower than 1)
        pstv_tsh = tsh
        ngtv_tsh = tsh
    # collect the neurons in that area-layer that are above and below the tresh
    prefer_r = (dp_position>=pstv_tsh)
    prefer_nr = (dp_position<=ngtv_tsh)
    print(f"area {area} - plane {plane}")
    print(f"NN area: {region.sum()}, NN plane: {p.sum()}, NN area-plane: {area_layer_neurons.sum()}")
    print(f"NN prefering rewarded trials: {prefer_r.sum()}, NN prefering non-rewarded trials: {prefer_nr.sum()}")
    return prefer_r, prefer_nr, area_layer_neurons

def compute_dprime(MouseObject, discrimination_region: tuple = (150,250), corridor_length: int = 400, nogray: bool = False):

    """
    Calculate dprime for a given mouse object

    Args:
    -------
        MouseObject (MouseObject): MouseObject
        corridor_length (int, optional): corridor length. Defaults to None
        nogray (bool, optional): if True, subtracts the gray response from the interpolated response. Defaults to False
        usetrain (bool, optional): if True, uses the even trials for the norm, odd trials otherwise. Defaults to False
    Returns:
    --------
        Mouse.Object.train_dp (np.array): dprime values (neurons x positions)
        MouseObject.projected_response (np.array): interpolated response divided by the pool of stds of the odd rew and nrew trials (neurons x trials x positions)
    """

    # checks if the object contains the spks interpolated and the frameselector properties, if don't creates them
    if ('interp_spks' not in properties(MouseObject)):
        if ('frameselector' not in properties(MouseObject)):
            MouseObject.frameselector = get_frameselector(MouseObject, effective_frames=True)
            MouseObject.interp_spks = interp_spks_by_corridorlength(MouseObject, MouseObject.frameselector, corridor_length=corridor_length)
        else:
            MouseObject.interp_spks = interp_spks_by_corridorlength(MouseObject, MouseObject.frameselector, corridor_length=corridor_length)
    # get trial numbers by trial type
    trial_dict = get_trialno_bytype(MouseObject.frameselector)
    # create the "train" trial numbers for rew and nrew trials
    train_trial_r = trial_dict["rewarded"][::2]
    train_trial_nr = trial_dict["non rewarded"][::2]
    #get the responses from those trials
    if nogray:
        gray_response = MouseObject.interp_spks[:, :, 300:400] # neurons, trials, positions (interp_spks is not zscored)
        avg_gray_response = gray_response.mean(2, keepdims=True) # avg over gray
        no_gray = MouseObject.interp_spks - avg_gray_response # subtract avg gray response
        no_gray = no_gray[:,:,discrimination_region[0]:discrimination_region[1]].mean(2, keepdims=True)
        r1 = no_gray[:, train_trial_r, :]
        r2 = no_gray[:, train_trial_nr, :]
    else:
        response = MouseObject.interp_spks[:,:,discrimination_region[0]:discrimination_region[1]].mean(2, keepdims=True)
        r1 = response[:, train_trial_r]
        r2 = response[:, train_trial_nr]

    # collect means and stds
    mu1 = r1.mean(1, keepdims=True)
    mu2 = r2.mean(1, keepdims=True)
    std1 = r1.std(1, keepdims=True) + np.finfo(np.float64).tiny
    std2 = r2.std(1, keepdims=True) + np.finfo(np.float64).tiny
    #compute the train dprime
    MouseObject.train_dp = 2 * ((mu1 - mu2) / (std1 + std2))
    MouseObject.train_dp = np.squeeze(MouseObject.train_dp)
    MouseObject.projected_response = 2 * (MouseObject.interp_spks / (std1 + std2))
    print("dprime saved in MouseObject.train_dp (neurons) using even trials")
    print("projected_response saved in MouseObject.projected_response (neurons x trials x positions)")
    

def filter_neurons(MouseObject, area: str = 'all', plane: int = 1, tsh: float = 95, dendrites: bool = False):
    """
    Calculate dprime for a given mouse object, trial dictionary and trial types, the discrimination region is the boundary of the corridor to be used for the calculation

    Args:
        MouseObject (MouseObject): MouseObject
        tsh (float, optional): sets the tsh to that percentile. Defaults to 95.
        discrimination_region (tuple of ints, optional): corridor region boundaries . Defaults to None.
        corridor_length (int, optional): corridor length. Defaults to None.
        plane (int, optional): plane to use for the calculation. Defaults to 1.
        iarea (str, optional): area mask. Defaults to None.

    Returns:
    --------
        dp (np.array): dprime values
        prefer_r (np.array): rewarded trials prefering neurons for the given plane - area
        prefer_nr (np.array): non-rewarded trials prefering neurons for the given plane - area
        selected_neurons (np.array): all neurons for the given plane - area

    """
    # checks if the object contains the spks interpolated and the frameselector properties, if don't creates them
    #if ('train_dp' not in properties(MouseObject)):
    #    compute_dprime(MouseObject, corridor_length = corridor_length)   
    # filtering neurons per area / plane
    region = get_region_idx(MouseObject.iarea, area) # get neurons in the area 
    assert plane == 1 or plane == 2, "plane must be 1 or 2"
    p = (MouseObject._iplane < 10) if plane == 2 else (MouseObject._iplane >= 10) # get neurons in the plane
    area_layer_neurons = p * region #gets neurons in the area-plane
    # get the neurons the dprime distribution for the target area-layer population
    if dendrites == True:
        area_layer_neurons = area_layer_neurons * MouseObject.dendrites
        dp_al = MouseObject.train_dp[area_layer_neurons]
    else:
        if ('dendrites' in properties(MouseObject)):
            area_layer_neurons = area_layer_neurons * ~MouseObject.dendrites
            dp_al = MouseObject.train_dp[area_layer_neurons]
        else:
            dp_al = MouseObject.train_dp[area_layer_neurons]
    # get the threshold for the dprime distribution
    if tsh > 1: # in the case we want to use a percentile
        pstv_tsh = np.percentile(dp_al, tsh) #positive threshold
        ngtv_tsh = np.percentile(dp_al, 100-tsh)
    else: # in the case we want to use a fixed value (needs to be a number lower than 1)
        pstv_tsh = tsh
        ngtv_tsh = -tsh
    # collect the neurons in that area-layer that are above and below the tresh
    prefer_r = (dp_al>=pstv_tsh)
    prefer_nr = (dp_al<=ngtv_tsh)
    print(f"area {area} - plane {plane}")
    print(f"NN area: {region.sum()}, NN plane: {p.sum()}, NN area-plane: {area_layer_neurons.sum()}")
    print(f"NN prefering rewarded neurons: {prefer_r.sum()}, NN prefering non-rewarded neurons: {prefer_nr.sum()}")
    return prefer_r, prefer_nr, area_layer_neurons

def get_trials_with_licks(MouseObject, lick_window=(150,250), trialtype: str = 'rewarded'):
    licksdf = get_lick_df(MouseObject, drop_last_trial=True)
    trial_no = get_trialno_bytype(MouseObject.frameselector)
    trial_no = trial_no[trialtype]
    licks = licksdf.query(f'trial_type == "{trialtype}"')
    licks = licks[licks['flag']!=1]
    trials_w_licks = licks.query(f'distance >= {lick_window[0]} and distance < {lick_window[1]}')['trial'].unique().astype(int)
    trials_w_licks = trials_w_licks - 1
    trials_wo_licks = trial_no[np.isin(trial_no, trials_w_licks, invert=True)]
    return trials_w_licks, trials_wo_licks