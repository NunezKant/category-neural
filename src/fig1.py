import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel, sem


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

def compute_gi_bins(first_day_cds, last_day_matched_cds, binsize=25, corridor_lim=400):
    bins = np.arange(binsize, corridor_lim+1, binsize)
    nbins = len(bins)
    gis_first_bin = np.empty((nbins, first_day_cds.shape[0], 4, 2))
    gis_last_bin = np.empty((nbins, last_day_matched_cds.shape[0], 4, 2))
    for i, pos in enumerate(bins):
        gis_first, dis_first, gen_first = compute_gi(first_day_cds, pos)
        gis_last, dis_last, gen_last = compute_gi(last_day_matched_cds, pos)
        gis_first_bin[i] = gis_first
        gis_last_bin[i] = gis_last
    return gis_first_bin, gis_last_bin, bins

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


def plot_gi_comparison(gis_sess, n_sess, labels, ax):
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
    for a in range(4):
        for ctp in range(2):
            day_one_r = gis_sess[:n_sess, a, ctp]
            day_two_r = gis_sess[n_sess:, a, ctp]
            mean_day_one = np.mean(day_one_r, axis=0)
            mean_day_two = np.mean(day_two_r, axis=0)
            sem_day_one = sem(day_one_r, axis=0)
            sem_day_two = sem(day_two_r, axis=0)
            ax[ctp].scatter(a-.1, mean_day_one, color='gray', alpha=1, s=10)
            ax[ctp].scatter(a+.1, mean_day_two, color='k', alpha=1, s=20)
            ax[ctp].errorbar(a-.1, mean_day_one, yerr=sem_day_one, color='gray', alpha=1)
            ax[ctp].errorbar(a+.1, mean_day_two, yerr=sem_day_two, color='k', alpha=1)
            ax[ctp].set_xticks(np.arange(4), ['V1', 'medial', 'lateral', 'anterior'])
            
            ax[ctp].axhline(y=0, color='gray', linestyle='--', alpha=0.2)
            if ctp == 0:
                ax[ctp].set_ylabel('Invariance Index $(a.u.)$')
                ax[ctp].set_yticks([0,.25,.5,.75, 1, 1.25])
                ax[ctp].set_title("excitatory")
            else:
                ax[ctp].set_title("inhibitory")
            t, p = ttest_rel(day_two_r, day_one_r, alternative='greater')
            p_t = significance(p)
            ax[ctp].text(a, 1, p_t, ha='center', va='center', color='k', fontsize=20, transform=ax[ctp].transData)

            # connect lines between every sample point and each day
            for m in range(4):
                ax[ctp].plot([a-.1, a+.1], [day_one_r[m], day_two_r[m]], color='k', linewidth=.5, alpha=0.4)

    ax[-1].text(1.1, .85, labels[0], ha='left', color='gray', transform=ax[-1].transAxes)
    ax[-1].text(1.1, .75, labels[1], ha='left', color='k',transform=ax[-1].transAxes)
    plt.tight_layout()

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
    colors = ["#b1afcf",'#756bb1',"#9691c7"]
    offset = [-0.1, 0.1, 0.3] # offset for the first, second and control GI
    labels = ['first day', 'last day (behavior matched)', 'last day (all trials)']
    for i_gi, gi in enumerate([first_gi, second_gi, control_gi]):
        for a in range(4):
            for ctp in range(2):
                mean = np.mean(gi[:, a, ctp], axis=0)
                sem_ = sem(gi[:, a, ctp], axis=0)
                ax[ctp].scatter(a+offset[i_gi], mean, color=colors[i_gi], alpha=1, s=20)
                ax[ctp].errorbar(a+offset[i_gi], mean, yerr=sem_, color=colors[i_gi], alpha=1)
                ax[ctp].set_xticks(np.arange(4), ['V1', 'medial', 'lateral', 'anterior'])
                ax[ctp].axhline(y=0, color='gray', linestyle='--', alpha=0.2)
                if ctp == 0:
                    ax[ctp].set_ylabel('Invariance Index $(a.u.)$')
                    ax[ctp].set_yticks([0,.25,.5,.75, 1, 1.25])
    ax[0].text(0.05, .92, "first day", ha='left', color=colors[0], transform=ax[0].transAxes, fontsize=10)
    ax[0].text(0.05, .86, "last day (matched)", ha='left', color=colors[1],transform=ax[0].transAxes, fontsize=10)
    ax[0].text(0.05, .8, "last day (all trials)", ha='left', color=colors[2], transform=ax[0].transAxes, fontsize=10)

    for a in range(4):
        for ctp in range(2):
            day_one_r = first_gi[:, a, ctp]
            day_two_r = second_gi[:, a, ctp]
            control_r = control_gi[:, a, ctp]
            t, p = ttest_rel(day_two_r, day_one_r, alternative='greater')
            p_t = significance(p)
            if p<.05:
                ax[ctp].text(a, 1, p_t, ha='center', va='center', color='k', fontsize=15, transform=ax[ctp].transData)
                # a line between one category and the other
                from matplotlib.lines import Line2D
                line = Line2D([a-.1, a+.1], [1, 1], color='k', linewidth=1, alpha=1)
                ax[ctp].add_line(line)
            t, p = ttest_rel(control_r, day_one_r, alternative='greater')
            p_t = significance(p)
            if p<.05:
                ax[ctp].text(a+.1, 1.15, p_t, ha='center', va='center', color='k', fontsize=15, transform=ax[ctp].transData)
                # a line between one category and the other
                from matplotlib.lines import Line2D
                line = Line2D([a-.1, a+.3], [1.15, 1.15], color='k', linewidth=1, alpha=1)
                ax[ctp].add_line(line)
            for m in range(first_gi.shape[0]):
                ax[ctp].plot([a-.1, a+.1], [day_one_r[m], day_two_r[m]], color='k', linewidth=.5, alpha=0.4)

def plot_cumulative_gi(ax, gis_first_bin, gis_last_bin, a, i, errorbars=True, xlabel=None, ylabel=None, title=None, legend=False):
    """
    Plot the cumulative GI for a given area and cell type.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the data.
    gis_last_bin : np.ndarray
        The GI data for the last training session. Shape: (nbins, areas, cell types).
    gis_first_bin : np.ndarray  
        The GI data for the first training session. Shape: (nbins, areas, cell types).
    a : int
        Index of the area to plot.
    i : int
        Index of the cell type to plot.
    errorbars : bool
        Whether to plot error bars or not.
    xlabel : str or None
        Label for the x-axis. If None, no label is set.
    ylabel : str or None
        Label for the y-axis. If None, no label is set.
    """
    gis_first_bin_mean = gis_first_bin.mean(1)
    gis_last_bin_mean = gis_last_bin.mean(1)
    gis_first_bin_sem = sem(gis_first_bin, axis=1)
    gis_last_bin_sem = sem(gis_last_bin, axis=1)
    nbins = gis_last_bin_mean.shape[0]
    xrange = np.arange(nbins)
    cmap = ['#1b9e77','#d95f02','#7570b3','#e7298a']
    areas = ["V1", "medial", "lateral", "anterior"]
    if errorbars:
        ax.errorbar(xrange, gis_last_bin_mean[:,a,i], yerr=gis_last_bin_sem[:,a,i], linewidth=1, capsize=1, color=cmap[a], linestyle="None")
        ax.scatter(xrange, gis_last_bin_mean[:,a,i], linewidth=1, label="last day (matched)", color=cmap[a], marker='x')
        ax.scatter(xrange, gis_first_bin_mean[:,a,i], color=cmap[a], linewidth=1, label=f"first day")
        ax.errorbar(xrange, gis_first_bin_mean[:,a,i], yerr=gis_first_bin_sem[:,a,i], linewidth=1, capsize=1, color=cmap[a], linestyle="None")
    else:
        ax.plot(xrange, gis_last_bin_mean[:,a,i], linewidth=1, label=f"last day (matched)", color=cmap[a])
        ax.fill_between(xrange, 
                        gis_last_bin_mean[:,a,i] - gis_last_bin_sem[:,a,i], 
                        gis_last_bin_mean[:,a,i] + gis_last_bin_sem[:,a,i], 
                        color=cmap[a], alpha=0.2)
        ax.plot(xrange, gis_first_bin_mean[:,a,i], color=cmap[a], linewidth=1, label=f"first day", linestyle="--")
        ax.fill_between(xrange, 
                        gis_first_bin_mean[:,a,i] - gis_first_bin_sem[:,a,i], 
                        gis_first_bin_mean[:,a,i] + gis_first_bin_sem[:,a,i], 
                        color=cmap[a], alpha=0.2)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(areas[a], loc='center')
    #if legend:
    #    ax.text(nbins-1 + .2, gis_last_bin_mean[-1,a,i], "last day (matched)", ha='left', va='center')
    #    ax.text(nbins-1 + .2, gis_first_bin_mean[-1,a,i], "first day", ha='left', va='center')
    ax.text(0.05, 0.9, areas[a], ha='left', va='center', color=cmap[a], transform=ax.transAxes)
    ax.set_ylim(0,.9)
    ax.set_yticks([0,.25,.5,.75], ["0", ".25", ".50", ".75"])

def lick_averages(df, ax, lines=True, alpha=1, offset=0, ylabel = True, **kwargs):
    from scipy import stats
    data = df[["rewarded", "non rewarded", "rewarded test", "non rewarded test"]]
    lick_rate = pd.melt(data, var_name="trial_type", value_name="lick_rate")
    lick_rate["trial_type"] = lick_rate["trial_type"].map({"rewarded": "Prototype A", "non rewarded": "Prototype B", "rewarded test": "Rest of A", "non rewarded test": "Rest of B"})
    import scikit_posthocs as sp
    res = sp.posthoc_ttest(lick_rate, val_col='lick_rate', group_col='trial_type', p_adjust='holm')
        #print(np.round(res,4))
    if lines:
        for i, row in data.iterrows():
            ax.plot([1, 2], [row['rewarded'], row['non rewarded']], '-', alpha=0.1, color='gray')
            ax.plot([3, 4], [row['rewarded test'], row['non rewarded test']], '-', alpha=0.1, color='gray')
    cmap = {"rewarded": 'tab:green', "non rewarded": 'tab:red', "rewarded test": 'tab:cyan', "non rewarded test": 'tab:orange'}
    for i, column in enumerate(['rewarded', 'non rewarded', 'rewarded test', 'non rewarded test'], start=1):
        mean = data[column].mean()
        median = data[column].median()
        c = cmap[column]
        ax.errorbar(i+offset, mean, yerr=stats.sem(data[column]), color=c, alpha=alpha, zorder=0)
        ax.plot(i+offset, median, '_', color=c, markersize=13, markeredgewidth=2, alpha=alpha, zorder=1)
        ax.plot(i+offset, mean, '8', color=c, markersize=5, markerfacecolor='white',  alpha=alpha, zorder=0)
    from matplotlib.lines import Line2D
    comparisons = [(0,1), (2,3)]
    xcoor = []
    ycoor = []
    pvals = []
    yval = 1
    for comp in comparisons:
        if res.iloc[comp[0], comp[1]] < 0.05:
            xcoor.append([comp[0]+1, comp[1]+1])
            ycoor.append(yval)
            yval += 0.03
            pvals.append(significance(res.iloc[comp[0], comp[1]]))
    #xcoor = [[1,4], [1,3], [1,2], [2,3], [3,4]]
    #ycoor = [1.14, 1.09, 1.04, 1.02, 1]
    fig = plt.gcf()
    lines = [Line2D(x, [y,y], color='k', linewidth=.5, transform=ax.transData, figure=fig) for x,y in zip(xcoor, ycoor)]
    for x,y,text in zip(xcoor, ycoor, pvals):
        ax.text(np.mean(x), y, text, ha='center', va='center', transform=ax.transData, fontsize=12)
    for line in lines:
        fig.add_artist(line)
    if ylabel:
        ax.set_ylabel("% trials with licks")
    ax.set_xticks([1, 2, 3, 4])
    if "xtickslabels" in kwargs:
        ax.set_xticklabels(kwargs["xtickslabels"])
    else:
        ax.set_xticklabels(["Prot A", "Prot B", "Rest A", "Rest B"])
    ax.set_yticks([0,.25, .5, .75, 1], [0, 25, 50, 75, 100])

def plot_behav_dis(ax, behavior_df, first100df):
    """
    Plot the behavioral DI and GI_num for two tasks.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the data.
    behavior_df : pd.DataFrame
        The DataFrame containing the behavioral data for the second task.
    first100df : pd.DataFrame
        The DataFrame containing the behavioral data for the first task.
    """
    task1_xs = [-.1, .9]
    task2_xs = [.1, 1.1]
    xs = [task1_xs, task2_xs]
    colors = ["#939090","#383737"]
    labels = ["Task 1", "Task 2"]
    for it, task_df in enumerate([first100df, behavior_df]):
        di_mean = task_df['DI'].mean()
        di_sem = sem(task_df['DI'])
        gi_num_mean = task_df['GI_num'].mean()
        gi_num_sem = sem(task_df['GI_num'])
        xvals = xs[it]
        c = colors[it]
        ax.errorbar(x=xvals[0], y=di_mean, yerr=di_sem, color=c, alpha=1, capsize=1)
        ax.errorbar(x=xvals[1], y=gi_num_mean, yerr=gi_num_sem, color=c, alpha=1, capsize=1)
        ax.scatter(x=xvals[0], y=di_mean, color=c, alpha=1, s=20)
        ax.scatter(x=xvals[1], y=gi_num_mean, color=c, alpha=1, s=20)
        ax.text(.1, .3-(it*.1) -.05, labels[it], ha='left', va='center', color=c, transform=ax.transAxes, fontsize=10)
    ax.set_ylabel("Behavioral DI");
    ax.set_xticks([0,1], ["Prototypes", "Non Prototypes"]);
    ax.set_ylim(0,1)
    
def accuracy_plot(overall_acc, ax):
    """
    """
    corridor_length = 400
    bsize = 25
    n_bins = corridor_length // bsize
    avg_acc = np.mean(overall_acc, axis=0)
    sem_acc = sem(overall_acc, axis=0)
    xtickslabels = [":100", ":250", ":400"]
    ax.plot(range(0, n_bins), avg_acc, color='k', marker='o', markersize=1, linewidth=1)
    ax.fill_between(range(0, n_bins), avg_acc - sem_acc, avg_acc + sem_acc, color='k', alpha=0.2)
    ax.set_ylabel("Accuracy")
    ax.set_yticks([.5,.7,.9], [.5, .7, .9,])
    ax.set_xticks([3, 9, 15], xtickslabels)
    ax.set_xlabel("Position bin ($cm$)")
    ymin, ymax = ax.get_ylim()
    ax.fill_between(np.arange(4), ymin, ymax, color='gray', alpha=0.2)

def betas_plot(overall_betas, ax, legend=False):
    # get 6 colors from Dark2 color palette
    from matplotlib import cm
    cmap = cm.get_cmap('Set2', 6)
    reg_names = ["Intercept", "Lick rate", "Speed", "Acceleration", "$\Delta$ Pupil", "$\Delta$ Motion"]
    corridor_length = 400
    bsize = 25
    n_bins = corridor_length // bsize
    n_cov = 6
    colors = [cmap(i) for i in range(6)]
    avg_b = np.mean(overall_betas, axis=0)
    sem_b = sem(overall_betas, axis=0)
    xtickslabels = [":100", ":250", ":400"]
    for i in range(n_cov):
        ax.plot(range(0, n_bins), avg_b[i], label=reg_names[i], marker='o', markersize=1, color=colors[i], linewidth=1)
        ax.fill_between(range(0, n_bins), avg_b[i]-sem_b[i], avg_b[i]+sem_b[i], color=colors[i], alpha=0.2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=1)
    ax.set_xticks([3, 9, 15], xtickslabels)
    ax.set_ylabel(r"$\beta$", rotation=0, labelpad=10)
    ax.set_xlabel("Position bin ($cm$)")
    ymin, ymax = ax.get_ylim()
    ax.fill_between(np.arange(4), ymin, ymax, color='gray', alpha=0.2)
    if legend:
        for i in range(n_cov):
            ax.text(1, .9-(i*.1), reg_names[i], ha='left', va='center', fontsize=8, color=colors[i], transform=ax.transAxes)
        #ax.legend(loc='upper left', fontsize=default_font, bbox_to_anchor=(1.05, 2), frameon=False)

def plot_catvsbehav(behav_cov_catA, behav_cov_catB, ylabel, xlabel, ax, legend=False):
    cat_a = behav_cov_catA
    cat_b = behav_cov_catB
    ax.plot(cat_a.mean(0), color='tab:green', linestyle='-', label='Category A')
    ax.fill_between(np.arange(400), cat_a.mean(0)-sem(cat_a, axis=0), cat_a.mean(0)+sem(cat_a, axis=0), alpha=0.3, color='tab:green')
    ax.plot(cat_b.mean(0), color='tab:red', linestyle='-', label='Category B')
    ax.fill_between(np.arange(400), cat_b.mean(0)-sem(cat_b, axis=0), cat_b.mean(0)+sem(cat_b, axis=0), alpha=0.3, color='tab:red')
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xticks([0, 150, 300, 400], ['0', '150', '300', '400'])    
    if legend:
        ax.legend(bbox_to_anchor=(1.05, .9), frameon=False)
    if xlabel:
        ax.set_xlabel("Position ($cm$)")
    #fill between 0 and 100 with gray
    ymin, ymax = ax.get_ylim()
    ax.fill_between(np.arange(100), ymin, ymax, color='gray', alpha=0.2)
    ax.axvline(150, color='k', linestyle='--', alpha=0.3)
    ax.axvline(300, color='k', linestyle='--', alpha=0.3)

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

def plot_cds(day_response, ttype, area, ctype, ax, references=True):
    
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
    trial_type_palette = ['tab:green', 'tab:red', 'tab:cyan', 'tab:orange']
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

def plot_matched_trials(overall_prob, catA_trials, catB_trials, ax, ax2):
    bins = np.linspace(0, 1, 11)
    histA, _ = np.histogram(overall_prob[catA_trials, 3], bins=bins, density=False)
    histB, _ = np.histogram(overall_prob[catB_trials, 3], bins=bins, density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.bar(bin_centers, histA, width=np.diff(bins), alpha=0.4, color='tab:green', label='Category A')
    ax.bar(bin_centers, histB, width=np.diff(bins), alpha=0.4, color='tab:red', label='Category B')

    # Highlight overlap with hatching
    overlap = np.minimum(histA, histB)
    bars = ax.bar(bin_centers, overlap, width=np.diff(bins), alpha=0, label='Overlap', hatch='///')
    ymin, ymax = ax.get_ylim()
    # Set the y-limits to match the first panel
    ax2.set_ylim(ymin, ymax)
    # Plot overlap in right panel
    ax2.bar(bin_centers, overlap, width=np.diff(bins), color='gray', label='Overlap')

    # Remove left axis of right panel
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_ticks([])
    ax2.tick_params(labelleft=False)

    fig = plt.gcf()
    import matplotlib.patches as mpatches
    arrow = mpatches.FancyArrowPatch(
        (.71, .72), (.84, .72),
        transform=fig.transFigure,
        arrowstyle="->", color='k', lw=1, mutation_scale=10
    )
    fig.patches.append(arrow)
    ax2.text(0.5, 0.7, "Behaviorally matched", ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='k')
    ax.set_xlabel('Prob (Cat A)')
    ax2.set_xlabel('Prob (Cat A)')
    ax.set_ylabel('Trials')