import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from src import utils
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
#sns.set_context("notebook")

from matplotlib import rcParams
default_font = 20
rcParams["font.family"] = "Arial"
rcParams["font.size"] = default_font

def rastermap_plot(
    MouseObject,
    neuron_embedding,
    frame_selection=0,
    frame_num=5000,
    svefig=False,
    savepath=None,
    format="png",
    clustidx = None,
):
    """
    plot the rastermap embedding with behavioral annotations for a given mouse object and embedding

    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the data
    neuron_embedding : np.array
        rastermap embedding of the spks
    frame_selection : int
        which frame_num frames to plot, i.e. 3 means the frames from 1500 to 2000 if frame_num = 500
    frame_num : int
        number of frames to plot
    svefig : bool
        whether to save the figure or not
    format : str
        format of the saved figure
    savepath : str
        path to save the figure
    clustidx : tuple, optional
        tuple of two idx containing the start and end of a desired cluster, by default None
    """
    ## data unpacking ##
    xmin = frame_selection * frame_num
    xmax = (frame_selection + 1) * frame_num

    run = MouseObject._timestamps["run"]
    tframe = MouseObject._timestamps["trial_frames"]
    isrewarded = MouseObject._trial_info["isrewarded"]
    istest = MouseObject._trial_info["istest"]
    alpha = MouseObject._timestamps["alpha"]
    nsuper = neuron_embedding.shape[0]

    ## figure creation ##
    fig = plt.figure(figsize=(16, 9), dpi=300)
    grid = plt.GridSpec(11, 5, hspace=0.2, wspace=0.2)
    raster_ax = fig.add_subplot(grid[2:, :5], facecolor="w")
    alpha_ax = fig.add_subplot(grid[:1, :5], sharex=raster_ax)
    vel_ax = fig.add_subplot(grid[1:2, :5], sharex=raster_ax)
    # the embedding is zscored for this particular time range so the contrast is better.
    raster_ax.imshow(
        zscore(neuron_embedding[:, xmin:xmax], 1),
        cmap="gray_r",
        aspect="auto",
        vmin=0,
        vmax=2,
    )
    if clustidx is not None:
        raster_ax.fill_between(np.arange(0,frame_num),clustidx[1],clustidx[0], color='tab:purple', alpha=0.2)
    vel_ax.plot(run[xmin:xmax], linewidth=0.5, color="k")
    alpha_ax.plot(alpha[xmin:xmax], linewidth=0.5, color="k")
    for label_vel, label_alpha in zip(
        vel_ax.get_xticklabels(), alpha_ax.get_xticklabels()
    ):
        label_vel.set_visible(False)
        label_alpha.set_visible(False)
    lweight = 3
    ## behavioral annotations ##
    for i, annot in enumerate(["lick_frames", "reward_frames"]):
        ranges = (MouseObject._timestamps[annot] > xmin) * (
            MouseObject._timestamps[annot] < xmax
        )
        pos = MouseObject._timestamps[annot][ranges] - xmin
        if i == 0:
            vel_ax.plot(
                pos,
                -np.min(run[xmin:xmax]) * np.ones(len(pos)),
                "|b",
                markersize=5,
                label="licks",
                alpha=0.2,
            )
        else:
            for p in pos:
                vel_ax.axvline(
                    p,
                    ymin=0,
                    ymax=nsuper,
                    label="reward delivery",
                    linestyle="dashed",
                    color="m",
                    alpha=0.5,
                    lw=lweight,
                )
                alpha_ax.axvline(
                    p,
                    ymin=0,
                    ymax=nsuper,
                    label="reward delivery",
                    linestyle="dashed",
                    color="m",
                    alpha=0.5,
                    lw=lweight,
                )
                raster_ax.axvline(
                    p,
                    ymin=0,
                    ymax=nsuper,
                    label="reward delivery",
                    linestyle="dashed",
                    color="m",
                    alpha=0.5,
                    lw=lweight,
                )
    vel_ax.text(
        1.01,
        0.4,
        "reward delivery",
        c="m",
        va="center",
        transform=vel_ax.transAxes,
        fontsize=20,
        alpha=0.8,
    )
    vel_ax.text(
        1.01,
        0.1,
        "licks",
        c="b",
        va="center",
        transform=vel_ax.transAxes,
        fontsize=20,
        alpha=0.8,
    )

    # trial type annotations #
    frame_ranges = (tframe > xmin) * (tframe <= xmax)
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    categories, _ = utils.get_trial_categories(isrewarded, istest)
    for cat_color in opt_dict.items():
        ix = frame_ranges * (categories == cat_color[0])
        ix = ix.nonzero()[0]
        for i in ix:
            raster_ax.axvline(
                x=tframe[i] - xmin,
                ymin=0,
                ymax=nsuper,
                color=cat_color[1],
                lw=lweight,
                alpha=0.5,
            )
            vel_ax.axvline(
                x=tframe[i] - xmin,
                ymin=0,
                ymax=nsuper,
                color=cat_color[1],
                lw=lweight,
                alpha=0.5,
            )
            alpha_ax.axvline(
                x=tframe[i] - xmin,
                ymin=0,
                ymax=nsuper,
                color=cat_color[1],
                lw=lweight,
                alpha=0.5,
            )

    text_offset = 0
    for cat_color in opt_dict.items():
        text_offset -= 0.05
        if np.sum((categories == cat_color[0])) > 0:
            raster_ax.text(
                1.01,
                0.9 + text_offset,
                cat_color[0],
                c=cat_color[1],
                va="center",
                transform=raster_ax.transAxes,
                fontsize=20,
                alpha=0.8,
            )

    raster_ax.set_ylabel("Superneuron #")
    if frame_num < 100:
        raster_ax.set_xticks([0, frame_num], [str(xmin), str(xmax)], fontsize=20)
    else:
        raster_ax.set_xticks(
            np.arange(0, frame_num, int(frame_num/5)), np.arange(xmin, xmax, int(frame_num/5)).astype(str)
        , fontsize=20)
    alpha_ax.set_ylabel("Contrast", fontsize=20)
    vel_ax.set_ylabel("Velocity", fontsize=20)
    raster_ax.set_xlabel("Frame # (at 3Hz)", fontsize=20)
    sns.despine()
    if svefig:
        if savepath is None:
            plt.savefig(
                f"rastermap_embedding_{str(frame_selection)}.{format}",
                bbox_inches="tight",
            )
            plt.close("all")
        else:
            plt.savefig(
                os.path.join(
                    savepath, f"rastermap_embedding_{str(frame_selection)}.{format}"
                ),
                bbox_inches="tight",
            )
            plt.close("all")


def neuron_distribution(MouseObject, conditions_dict = None, color_list = None):
    """
    plot the neuron distribution of each condition

    Parameters:
    MouseObject: Mouse object
    conditions_dict: dictionary of conditions (keys) and neuron indices (values), optional
    color_list: list of colors for each condition, optional

    Returns:
    plotly figure
    """
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=MouseObject._xpos, y=-MouseObject._ypos, mode = 'markers', marker= dict(size=3, color='gray',opacity = 0.2)), row=1, col=1)
    fig.data[0].name = "population"
    i = 1
    if conditions_dict is not None:
        assert color_list is not None, "color_list must be provided if conditions_dict is provided"
        for (name, idx), color in zip(conditions_dict.items(), color_list):
            fig.add_trace(go.Scatter(x=MouseObject._xpos[idx], y=-MouseObject._ypos[idx], mode = 'markers', marker= dict(size=6, color= color ,opacity = 0.5)), row=1, col=1)
            fig.data[i].name = name
            i += 1
    fig.update_layout(height=800, width=800, template="simple_white")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

def comparisonplot(data, figsize = (10,10), x="ID", y="Margin", hue="Layer"):

    def overall_margins_fromdf(data,x,y,hue):
        overall_margins = []
        for id in data[x].unique():
            for layer in data[hue].unique():
                selection = (data[x]==id) & (data[hue]==layer)
                overall_margin = data[selection][y].mean()
                overall_margins.append(overall_margin)
        return np.array(overall_margins)

    def get_text_xcordinates(data,x):
        xcoord = []
        n_ids = len(data[x].unique())
        for id in range(n_ids):
            x1 = id-0.2
            x2 = id+0.2
            xcoord.append(x1)
            xcoord.append(x2)
        return xcoord

    def annotate_means(ax, data):
        xcoord = get_text_xcordinates(data,x)
        means = overall_margins_fromdf(data,x,y,hue)
        for xc, mean in zip(xcoord,means):
            ax.annotate(str(np.round(mean,2)),xy=(xc,mean), ha="left", fontsize=15)

    plt.figure(figsize=figsize)
    ax = sns.violinplot(x=x,y=y,hue=hue,data=data);
    plt.setp(ax.collections, alpha=.5)
    ax = sns.stripplot(x=x,y=y,hue=hue,data=data, dodge=True, palette="tab10");
    annotate_means(ax, data)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2],labels[:2],title='Layer')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.despine()

def dprime_trial_responses(MouseObject, to_plot: str = 'neurons', layer: int = 1, region: str = None, types = ('rewarded','non rewarded'), 
                           corridor_region=(25,275), corridor_length=400, percentile = 97.5):
    trialno = utils.get_trialno_bytype(MouseObject.frameselector)
    _, iarea, _ = utils.load_behaviour_retinotopy(MouseObject, RETINOTOPY_PATH = "D:/retinotopy/aligned_xy")
    trial_type_idxs, region_idxs = utils.filterneurons(MouseObject, layer=layer, trial_type = True,
                                                         region = region, types = types, corridor_region=corridor_region, 
                                                         corridor_length=corridor_length, iarea = iarea, percentile = percentile)
    if layer == 1:
        selected_neurons = np.where(((MouseObject._iplane >= 10) * region_idxs) == 1)[0] 
        prefer_r_region =  selected_neurons[trial_type_idxs[0]]
        prefer_nr_region = selected_neurons[trial_type_idxs[1]]
        l = 'superficial (100$\mu m$)'
    else:
        selected_neurons = np.where(((MouseObject._iplane < 10) * region_idxs) == 1)[0] 
        prefer_r_region =  selected_neurons[trial_type_idxs[0]]
        prefer_nr_region = selected_neurons[trial_type_idxs[1]]
        l = 'deep (250$\mu m$)'
    if to_plot == 'neurons':
        to_sort = True # sort neurons by peak position
    elif to_plot == 'trials':
        to_sort = False # we are plotting trials, avg population response
    
    discrimation_pops = [prefer_r_region, prefer_nr_region]
    fig, ax = plt.subplots(2,5, figsize = (16,8))
    for i_pop, pop in enumerate(discrimation_pops):
        response = MouseObject.interp_spks[pop]
        if i_pop==0:
            resp_pop = 'rewarded'
            vr = np.percentile(np.abs(response[:,trialno['rewarded'][1::2]].flatten()), 95)
        else:
            resp_pop = 'non rewarded'
            vr = np.percentile(np.abs(response[:,trialno['non rewarded'][1::2]].flatten()), 95)
        for i_t, (key, value) in enumerate(trialno.items()):
            if to_sort:
                sorted_idx = utils.sortbypeak(response[:,value[::2]].mean(1)) # grab sort idx using only even trials
                if key in ['rewarded', 'non rewarded']:    
                    value = value[1::2] # test on odd trials only
                r = response[:,value].mean(1)
                r = r[sorted_idx]
            else:
                if key in ['rewarded', 'non rewarded']:    
                    value = value[1::2] 
                r = response[:,value].mean(0)
            colors = ["tab:green", "tab:red","tab:cyan", "tab:orange"]
            ax[i_pop,i_t].imshow(r, aspect = 'auto', cmap = 'RdBu_r', vmax = vr, vmin = -vr)
            ax[i_pop,i_t].axvline(150, color='k', linestyle='--')
            ax[i_pop,i_t].axvline(300, color='b', linestyle='--')
            ax[i_pop,4].plot(r.mean(0), color = colors[i_t], label = key)
            ax[i_pop,i_t].set_title(f'response to  {key}', size='12')
            if to_plot == 'neurons':
                ax[i_pop,0].set_ylabel(f'{resp_pop} \n prefering neurons', rotation=0, size='medium' , labelpad=60)
            else:
                ax[i_pop,0].set_ylabel(f'{resp_pop} \n avg response \n across trials', rotation=0, size='medium' , labelpad=60)
            if i_pop==1:
                ax[i_pop,i_t].set_xlabel('position')
        ax[i_pop,4].axvline(150, color='k', linestyle='--', label = 'middle of corridor')
        ax[i_pop,4].axvline(300, color='b', linestyle='--', label = 'start of intertrial')      
    ax[0,4].set_title('avg response')
    ax[1,4].set_xlabel('position')
    ax[0,4].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.suptitle(f'{l} {region}')
    sns.despine()    


def licksraster(MouseObject, ax):
    import seaborn as sns
    """
    Plot the lick data.

    Parameters
    ----------
    MouseObject : Mouse object
        Mouse object containing the data
    first_lick : bool
        If True, plot the first lick distribution over trials.
    fsize : tuple
        Figure size.
    lick_counter_lim : tuple
        Limits for the lick counter.

    Returns
    -------
    fig : figure
        Figure of lick data.
    """
    opt_dict = {
        "rewarded": "tab:green",
        "non rewarded": "tab:red",
        "rewarded test": "tab:cyan",
        "non rewarded test": "tab:orange",
    }
    #pct_axis = fig.add_subplot(grid[-2:, 5:])
    lick = get_lick_df(MouseObject, drop_last_trial=True)
    n_trials = int(lick.trial.max())
    isrewarded = MouseObject._timeline['TrialRewardStrct'].flatten()[:n_trials]
    isnew = MouseObject._timeline['TrialNewTextureStrct'].flatten()[:n_trials]
    trial_type , counts = utils.get_trial_categories(isrewarded, isnew)
    for key, value in counts.items():
        lick.loc[np.where(lick["trial_type"] == key)[0], "Weight"] = value
    lick = lick[lick["flag"] != 1]
    lick = lick.iloc[::2]
    category_number = len(np.unique(trial_type))
    if category_number == 4:
        categories = ["rewarded", "non rewarded", "rewarded test", "non rewarded test"]
    elif category_number == 2:
        categories = ["rewarded", "non rewarded"]
        

    for category in categories:
        position = lick[lick["trial_type"] == category]["distance"]
        trial = lick[lick["trial_type"] == category]["trial"]
        if category in ["rewarded", "non rewarded"]:
            ax.scatter(
                position,
                trial,
                marker="o",
                label=category,
                alpha=0.5,
                c = opt_dict[category],
                s=5,
            )
        elif category == "rewarded test":
            ax.scatter(
                position,
                trial,
                marker="X",
                alpha=0.2,
                label=category,
                c = opt_dict[category],
                s=5,
            )
        else:
            ax.scatter(
                position,
                trial,
                marker="X",
                alpha=0.2,
                label=category,
                c = opt_dict[category],
                s=5,
            )

    ax.set_xlabel("lick position (cm)")
    ax.set_ylabel("trial")
    ax.set_xlim(0, lick["distance"].max() + 10)
    #init_text = lick["distance"].max() -30
    i = 0
    change_dict = {'rewarded': 'train A', 'non rewarded': 'train B', 'rewarded test': 'test A', 'non rewarded test': 'test B'}
    for category in categories:
        ax.text(310,320-i,change_dict[category],color=opt_dict[category], size=12)
        i+=27
    ax.set_ylim(0, n_trials+30)
    ax.set_xticks([0, 150, 250, 400])
    ax.fill_betweenx([0, n_trials+19], [150],[250], color='tab:green', alpha=0.2)
    ax.vlines(300,0, n_trials+19, color='k', linestyle='--', alpha = 0.3)
    #add text for reward region
    ax.text(145, n_trials+30, "reward \n region", color='tab:green', size=12)
    ax.tick_params(axis='both', which='major')
    sns.despine()

def get_lick_df(MouseObject, drop_last_trial=True):
    if "TX" in MouseObject.name:
        df = pd.DataFrame(MouseObject._timeline["Licks"].T, columns=["trial", "distance","alpha","is_rewarded","time", "flag"])
    else:
        df = pd.DataFrame(MouseObject._timeline["Licks"].T, columns=["trial", "distance","alpha","is_rewarded","time", "flag", "istest"])
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
    trial_type , _ = utils.get_trial_categories(isrewarded, isnew)
    for ix, ttype in enumerate(trial_type):
        df.loc[df.trial == ix+1, "trial_type"] = ttype
    df.drop(["time","datetime","is_rewarded","alpha"], axis=1, inplace=True)
    return df