import numpy as np
from scipy.stats import sem
from src import utils
import matplotlib.pyplot as plt

def dprime_cell(response, condition1, condition2, discrimination_region=(0,125), subpop=None):
    """
    Compute the d-prime for a single cell.
    """
    response = response[:,:,discrimination_region[0]:discrimination_region[1]].mean(2)
    r1 = response[:, condition1]
    r2 = response[:, condition2]
    if subpop is not None:
        r1 = r1[subpop]
        r2 = r2[subpop]
    # collect means and stds
    mu1 = r1.mean(1)
    mu2 = r2.mean(1)
    std1 = r1.std(1) + np.finfo(np.float64).tiny
    std2 = r2.std(1) + np.finfo(np.float64).tiny
    #compute the train dprime
    dp = 2 * ((mu1 - mu2) / (std1 + std2))
    return dp

def select_neurons(m1, area: str, celltype:str, dprime = None, dptsh=95):
    ia = utils.get_region_idx(m1.iarea, area)
    assert celltype in ['exc', 'inh'], "celltype must be either 'exc' or 'inh'"
    selected_type = np.logical_not(m1.isred[:,0]).astype(bool) if celltype == 'exc' else m1.isred[:,0].astype(bool)
    if dprime is None:
        pstv_tsh, ngtv_tsh = utils.get_dp_thresholds(m1.train_dp[ia*selected_type], tsh=dptsh) #tresh based on the area
    else:
        pstv_tsh, ngtv_tsh = utils.get_dp_thresholds(dprime[ia*selected_type], tsh=dptsh)
    prefer_r = (m1.train_dp>=pstv_tsh)
    prefer_nr = (m1.train_dp<=ngtv_tsh)
    area_prefer_r = prefer_r * ia * selected_type
    area_prefer_nr = prefer_nr * ia * selected_type
    return area_prefer_r, area_prefer_nr, selected_type, ia



def plot_cds(day_response, ttype, area, ctype, ax, references=True):
    from scipy.stats import sem
    """ 
    Plot the mean and SEM of the coding direction for a given trial type, area, and cell type.
    Parameters
    ----------
    day_response : np.ndarray
        The coding direction data for the day. Shape: (n_mice, n_trial_types, n_areas, n_cell_types, corridor_length)
    ttype : int
        The trial type index (0: rewarded, 1: non-rewarded, 2: rewarded test, 3: non-rewarded test).
    area : int
        The area index (0: V1, 1: medial, 2: lateral, 3: anterior).
    ctype : int
        The cell type index (0: excitatory, 1: inhibitory).
    ax : matplotlib.axes.Axes
        The axes on which to plot the data.
    """
    trial_type_palette = ['#0072B2', '#D55E00', '#56B4E9', '#E69F00']
    nmice, ntrial_types, nareas, ncelltypes, corridor_length = day_response.shape
    mean_response = np.mean(day_response, axis=0)
    sem_response = sem(day_response, axis=0)
    ax.plot(mean_response[ttype, area, ctype], color=trial_type_palette[ttype], linewidth=1)
    ax.fill_between(np.arange(corridor_length), mean_response[ttype, area, ctype] - sem_response[ttype, area, ctype],
                                mean_response[ttype, area, ctype] + sem_response[ttype, area, ctype],
                                color=trial_type_palette[ttype], alpha=0.2)
    if references:
        ax.axvline(x=150, color='gray', linestyle='--', alpha=0.2)
        ax.axvline(x=300, color='gray', linestyle='--', alpha=0.2)

def attention_effect(correct, incorrect, area, ctype, pos=(0,100)):
    """
    Calculate the attention effect for a given area and cell type.
    Parameters
    ----------
    correct : np.ndarray
        The coding direction data for correct trials.
    incorrect : np.ndarray
        The coding direction data for incorrect trials.
    area : int
        The area index (0: V1, 1: medial, 2: lateral, 3: anterior).
    ctype : int
        The cell type index (0: excitatory, 1: inhibitory).
    Returns
    -------
    np.ndarray
        The attention effect for the specified area and cell type.
    """
    correct = correct[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    incorrect = incorrect[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    incorrect_sep = incorrect[:,0] - incorrect[:,1]
    correct_sep = correct[:,0] - correct[:,1]
    attention_eff = 1 - (np.abs(incorrect_sep) / np.abs(correct_sep))
    return attention_eff


def prot_di(correct, incorrect, area, ctype, pos=(0,100)):
    """
    Calculate the attention effect for a given area and cell type.
    Parameters
    ----------
    correct : np.ndarray
        The coding direction data for correct trials.
    incorrect : np.ndarray
        The coding direction data for incorrect trials.
    area : int
        The area index (0: V1, 1: medial, 2: lateral, 3: anterior).
    ctype : int
        The cell type index (0: excitatory, 1: inhibitory).
    Returns
    -------
    np.ndarray
        The attention effect for the specified area and cell type.
    """
    correct = correct[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    incorrect = incorrect[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    incorrect_sep = incorrect[:,0] - incorrect[:,1]
    correct_sep = correct[:,0] - correct[:,1]
    return correct_sep, incorrect_sep

def get_di(subsetA, subsetB, area, ctype, type="prot", pos=(0,100)):
    """
    Get the coding direction for a given area and cell type.
    Parameters
    ----------
    subsetA : np.ndarray
        The coding direction data for the first subset.
    subsetB : np.ndarray
        The coding direction data for the second subset.
    type : str
        The type of coding direction to compute ('prot' or 'non prot').
    area : int
        The area index (0: V1, 1: medial, 2: lateral, 3: anterior).
    ctype : int
        The cell type index (0: excitatory, 1: inhibitory).
    Returns
    -------
    np.ndarray
        The coding direction for the specified area and cell type.
    """
    if type == "prot":
        ttypes = (0, 1)
    elif type == "non prot":
        ttypes = (2, 3)
    else:
        raise ValueError("type must be either 'prot' or 'non prot'")
    subsetA = subsetA[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    subsetB = subsetB[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    subsetA_di = subsetA[:, ttypes[0]] - subsetA[:, ttypes[1]]
    subsetB_di = subsetB[:, ttypes[0]] - subsetB[:, ttypes[1]]
    return subsetA_di,  subsetB_di

def rest_di(correct, incorrect, area, ctype, pos=(0,100)):
    """
    Calculate the attention effect for a given area and cell type.
    Parameters
    ----------
    correct : np.ndarray
        The coding direction data for correct trials.
    incorrect : np.ndarray
        The coding direction data for incorrect trials.
    area : int
        The area index (0: V1, 1: medial, 2: lateral, 3: anterior).
    ctype : int
        The cell type index (0: excitatory, 1: inhibitory).
    Returns
    -------
    np.ndarray
        The attention effect for the specified area and cell type.
    """
    correct = correct[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    incorrect = incorrect[:, :, area, ctype, pos[0]:pos[1]].mean(axis=-1)
    incorrect_sep = incorrect[:,2] - incorrect[:,3]
    correct_sep = correct[:,2] - correct[:,3]
    return correct_sep, incorrect_sep

def compute_gi(avgs_coding_dirs, pos):
    """
    Compute the GI for each cell type and area.
    GI = |cyan - orange| / |green - red|

    avgs_coding_dirs: array of shape (mice, ttype, area, cell type, positions)
    """
    gis = np.zeros((avgs_coding_dirs.shape[0], 4, 2))
    dis = np.zeros((avgs_coding_dirs.shape[0], 4, 2))
    gen = np.zeros((avgs_coding_dirs.shape[0], 4, 2))
    avgs_coding_dirs = avgs_coding_dirs[:,:,:,:,:pos].mean(-1) #mouse, ttype, area, cell type, positions
    for area in range(4):
        for cell_type in range(2):
            gen[:, area, cell_type] = avgs_coding_dirs[:,2,area,cell_type] - avgs_coding_dirs[:,3,area,cell_type]
            dis[:, area, cell_type] = avgs_coding_dirs[:,0,area,cell_type] - avgs_coding_dirs[:,1,area,cell_type]
            for m in range(avgs_coding_dirs.shape[0]):
                if (dis[m, area, cell_type] < 0) and (gen[m, area, cell_type] < 0):
                    gis[m, area, cell_type] = np.abs(gen[m, area, cell_type]) / np.abs(dis[m, area, cell_type]) * -1
                else:
                    gis[m, area, cell_type] = gen[m, area, cell_type] / dis[m, area, cell_type]
    return gis, dis, gen

def significance(pval):
    if  pval >= .05:
        sig = ''
    elif (pval < .05) and (pval >= .01):
        sig = '*'
    elif (pval < .01) and (pval >= .001):
        sig = '**'
    elif (pval < .001) and (pval >= .0001):
        sig = '***'
    else:
        sig = '****'
    return sig

def plot_gi_comparison(gis_sess1, gis_sess2, labels, ax):
    """
    Plot the GI comparison between the first and last training sessions.
    Parameters
    ----------
    gis_sess1 : np.ndarray
        The GI data for the first training session. Shape: (sess*day, areas, ctypes)
    gis_sess2 : np.ndarray
        The GI data for the last training session. Shape: (sess*day, areas, ctypes)
    ax : matplotlib.axes.Axes
        The axes on which to plot the data.
    """
    from scipy.stats import ttest_rel, sem
    for a in range(4):
        for ctp in range(2):
            day_one_r = gis_sess1[:, a, ctp]
            day_two_r = gis_sess2[:, a, ctp]
            mean_day_one = np.mean(day_one_r, axis=0)
            mean_day_two = np.mean(day_two_r, axis=0)
            sem_day_one = sem(day_one_r, axis=0)
            sem_day_two = sem(day_two_r, axis=0)
            ax[ctp].scatter(a-.1, mean_day_one, color='gray', alpha=1, s=20)
            ax[ctp].scatter(a+.1, mean_day_two, color='k', alpha=1, s=20)
            ax[ctp].errorbar(a-.1, mean_day_one, yerr=sem_day_one, color='gray', alpha=1)
            ax[ctp].errorbar(a+.1, mean_day_two, yerr=sem_day_two, color='k', alpha=1)
            ax[ctp].set_xticks(np.arange(4), ['V1', 'medial', 'lateral', 'anterior'])
            
            ax[ctp].axhline(y=0, color='gray', linestyle='--', alpha=0.2)
            if ctp == 0:
                ax[ctp].set_ylabel('Invariance Index $(a.u.)$')
                #ax[ctp].set_yticks([-1,-.5, 0,.5, 1])
                ax[ctp].set_title("excitatory")
            else:
                ax[ctp].set_title("inhibitory")
            t, p = ttest_rel(day_two_r, day_one_r, alternative='greater')
            p_t = significance(p)
            ax[ctp].text(a, .45, p_t, ha='center', va='center', color='k', fontsize=20, transform=ax[ctp].transData)

            # connect lines between every sample point and each day
            #for m in range(4):
            #    ax[ctp].plot([a-.1, a+.1], [day_one_r[m], day_two_r[m]], color='k', linewidth=.5, alpha=0.4)

    ax[-1].text(1.1, .85, labels[0], ha='left', color='gray', transform=ax[-1].transAxes)
    ax[-1].text(1.1, .75, labels[1], ha='left', color='k',transform=ax[-1].transAxes)

def move_axis(ax, hdx=0, vdx=0, widthdx=1, heightdx=1):
    poss = ax.get_position().bounds
    new_left = poss[0] + hdx
    new_bottom = poss[1] + vdx
    new_width = poss[2] * widthdx
    new_height = poss[3] * heightdx
    ax.set_position([new_left, new_bottom, new_width, new_height])

def show_image(ax, img_path):
    ax.imshow(plt.imread(img_path))
    ax.axis('off')

def add_panel_label(ax, label, x=0.01, y=0.99, **kwargs):
    """
    Add a label (e.g., 'A', 'B', ...) to the top left corner of an axis.
    Parameters:
        ax : matplotlib.axes.Axes
            The axis to label.
        label : str
            The label text.
        x, y : float
            Position in axis coordinates (default: top left).
        **kwargs : dict
            Additional arguments passed to ax.text (e.g., fontsize, fontweight).
    """
    ax.text(x, y, label, transform=ax.transAxes, ha='center', va='center', **kwargs)
    text = ax.text(x, y, label, transform=ax.transAxes, ha='center', va='center', **kwargs)
    x_axes, y_axes = text.get_position()
    display_coord = ax.transAxes.transform((x_axes, y_axes))
    curr_fig = plt.gcf()
    figure_coord = curr_fig.transFigure.inverted().transform(display_coord)
    print("Figure-relative coordinates:", figure_coord)

def plot_gi_comparison_wcontrol(first_gi, second_gi, control_gi, ax):
    """
    Plot the GI comparison between the first and last training sessions.
    Parameters
    ----------
    gis_sess : np.ndarray
        The GI data for the first and last training sessions. Shape: (sess*day, areas, ctypes)
    ax : matplotlib.axes.Axes
        The axes on which to plot the data.
    """
    from scipy.stats import ttest_rel, sem
    colors = ["#a8d7bc",'#2ca25f',"#77b9aa"]
    offset = [-0.1, 0.1, 0.3] # offset for the first, second and control GI
    labels = ['all water day (before)', 'test day', 'all water day (after)']
    for i_gi, gi in enumerate([first_gi, second_gi, control_gi]):
        for a in range(4):
            for ctp in range(2):
                mean = np.mean(gi[:, a, ctp], axis=0)
                sem_ = sem(gi[:, a, ctp], axis=0)
                ax[ctp].scatter(a+offset[i_gi], mean, color=colors[i_gi], alpha=1, s=20)
                ax[ctp].errorbar(a+offset[i_gi], mean, yerr=sem_, color=colors[i_gi], alpha=1)
                ax[ctp].set_xticks(np.arange(4), ['V1', 'medial', 'lateral', 'anterior'])
                #ax[ctp].axhline(y=0, color='gray', linestyle='--', alpha=0.2)
                if ctp == 0:
                    ax[ctp].set_ylabel('Invariance Index $(a.u.)$')
                    ax[ctp].set_yticks([0,.25,.5,.75, 1, 1.25])
                    ax[ctp].set_ylim(0, 1.3)
    ax[0].text(0.05, .92, "all water day (before)", ha='left', color=colors[0], transform=ax[0].transAxes)
    ax[0].text(0.05, .86, "test day (matched)", ha='left', color=colors[1],transform=ax[0].transAxes)
    ax[0].text(0.05, .8, "all water day (after)", ha='left', color=colors[2], transform=ax[0].transAxes)
    from matplotlib.lines import Line2D
    for a in range(4):
        for ctp in range(2):
            day_one_r = first_gi[:, a, ctp]
            day_two_r = second_gi[:, a, ctp]
            control_r = control_gi[:, a, ctp]
            t, p = ttest_rel(day_two_r, day_one_r)
            p_t = significance(p)
            if p<.05:
                ax[ctp].text(a, .75, p_t, ha='center', va='center', color='k', fontsize=15, transform=ax[ctp].transData)
                # a line between one category and the other
                
                line = Line2D([a-.1, a+.1], [.75, .75], color='k', linewidth=1, alpha=1)
                ax[ctp].add_line(line)
            t, p = ttest_rel(control_r, day_one_r)
            p_t = significance(p)
            if p<.05:
                ax[ctp].text(a+.1, .9, p_t, ha='center', va='center', color='k', fontsize=15, transform=ax[ctp].transData)
                # a line between one category and the other
                line = Line2D([a-.1, a+.3], [.9, .9], color='k', linewidth=1, alpha=1)
                ax[ctp].add_line(line)
            for m in range(first_gi.shape[0]):
                ax[ctp].plot([a-.1, a+.1], [day_one_r[m], day_two_r[m]], color='k', linewidth=.5, alpha=0.4)