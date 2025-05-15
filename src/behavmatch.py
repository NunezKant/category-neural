import numpy as np
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_behav_data(mouse):
    name, date, blk = mouse.name, mouse.datexp, mouse.blk
    speed = np.load(f"../data/{name}/{date}/{blk}/speed_interp.npy")
    motion = np.load(f"../data/{name}/{date}/{blk}/motion_energy_corridor.npy")
    pupil = np.load(f"../data/{name}/{date}/{blk}/pupil_area_corridor.npy")
    lick_rate = np.load(f"../data/notz/{name}/{date}/{blk}/lick_rate.npy")
    delta_motion = ((np.expand_dims(motion[:,0],axis=1) - motion) / (np.expand_dims(motion[:,0],axis=1))) * 100
    delta_pupil = ((np.expand_dims(pupil[:,0],axis=1) - pupil) / (np.expand_dims(pupil[:,0],axis=1))) * 100
    return speed, lick_rate, delta_motion, delta_pupil

def causal_exponential_filter(x, ew=0.1):
    """Causal exponential filter with smoothing factor alpha (0 < alpha <= 1)"""
    x_filtered = np.zeros_like(x)
    x_filtered[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        x_filtered[:, t] = ew * x[:, t] + (1 - ew) * x_filtered[:, t - 1]
    return x_filtered

def compute_acceleration(speed, ew=.1):
    """Compute acceleration from speed using a causal exponential filter"""
    smooth_speed = causal_exponential_filter(speed, ew=ew)
    acceleration = np.diff(smooth_speed, axis=1)
    acceleration = np.pad(acceleration, ((0,0),(1, 0)), mode='edge') # pad with last value
    return acceleration

def build_covariates(lick_rate, speed, acc, delta_pupil, delta_motion):
    """Build covariates for regression"""
    reg_names = ['Intercept','Lick rate', 'Speed', 'Acc', 'Pupil', 'Motion']
    features = [lick_rate, speed, acc, delta_pupil, delta_motion]
    covariates = np.stack(features, axis=2)
    print(covariates.shape, "trials, positions, features")
    return reg_names, covariates

def reg_cat_frombehav(category, covariates, bin_size=25, n_splits=10, cumulative=False):
    n_trials, n_positions, n_features = covariates.shape
    n_bins = n_positions // bin_size
    prob = np.zeros((n_trials, n_bins))
    betas = np.zeros((n_features+1, n_bins))
    models = []

    #sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=333)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

    for b in range(n_bins):
        # Average over bin
        if cumulative:
            bin_covs = covariates[:, :(b+1)*bin_size, :].mean(axis=1)
        else:
            bin_covs = covariates[:, b*bin_size:(b+1)*bin_size, :].mean(axis=1)
        X_bin = bin_covs  
        y = category

        fold_probs = np.zeros(n_trials)
        fold_betas = []

        for train_idx, test_idx in skf.split(X_bin, y):
            X_train, X_test = X_bin[train_idx], X_bin[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(max_iter=1000))
            ])
            pipe.fit(X_train, y_train)

            fold_probs[test_idx] = pipe.predict_proba(X_test)[:, 1]
            # Get coefficients from the pipeline
            coefs = pipe.named_steps['logreg'].coef_[0]
            intercept = pipe.named_steps['logreg'].intercept_[0]
            fold_betas.append(np.concatenate([[intercept], coefs]))
            models.append(pipe)

        prob[:, b] = fold_probs
        betas[:, b] = np.mean(fold_betas, axis=0)
        
    return models, prob, betas 

def compute_performance(prob, observed):
    n_bins = prob.shape[1]
    auc_scores = []
    accuracies = []
    for b in range(n_bins):
        auc = roc_auc_score(observed, prob[:, b])
        acc = accuracy_score(observed, (prob[:, b] > 0.5).astype(int))
        auc_scores.append(auc)
        accuracies.append(acc)
    return auc_scores, accuracies

def plot_performance_and_betas(auc_scores, accuracies, betas, names, ax, cumulative=False, bin_size=50):
    n_bins = 400//bin_size
    if cumulative:
        xtickslabels = [f":{int(i*bin_size)}" for i in range(1, n_bins+1)]
    else:
        xtickslabels = [f"{int(i*bin_size)}:{int((i+1)*bin_size)}" for i in range(n_bins)]

    n_bins = len(auc_scores)
    # Plot Accuracy
    ax[0].plot(range(n_bins), accuracies, label="Accuracy", color="k")
    # fill between the reward bins
    #ax[0].fill_between(range(6, 9), 0.5, 1, color='green', alpha=0.2)
    ax[0].set_ylabel("Performance")
    ax[0].set_yticks(np.arange(0.5, 1, 0.2))
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[0].set_title("Decoder Performance Across Bins")
    #add text with the avg accuracy of the first 4 bins
    ax[0].text(2, 0.8, f"{np.mean(accuracies[:5]):.2f}", color='k')
    ax[0].axvline(x=4, color='gray', linestyle='--', alpha=0.5)

    # Plot betas
    for i, name in enumerate(names):
        ax[1].plot(range(n_bins), betas[i], label=name)
    ax[1].set_ylabel("Beta Coefficients")
    ax[1].set_xlabel("Bins")
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1].set_title("Change in Betas Across Bins")
    ax[1].axvline(x=4, color='gray', linestyle='--', alpha=0.5)
    # get ylim of the second plot
    y_min, y_max = ax[1].get_ylim()
    #ax[1].fill_between(range(6, 9), y_min, y_max, color='green', alpha=0.2)
    ax[1].set_xticks(range(n_bins), xtickslabels, rotation=45)
    
    plt.tight_layout()
    plt.show()

def match_trials_by_prob_bins(prob, b, cond1, cond2, bins=np.linspace(0, 1, 11), random_state=0):
    """
    Match trials from each category within probability bins

    Parameters:
        prob: ndarray (trials, bins) – probability per trial and bin
        b: int – position bin index to match trials 
        cond1: list of indices – trials from condition 1
        cond2: list of indices – trials from condition 2
        bins: array-like – bin edges for digitizing probabilities

    Returns:
        matched_indices: list of lists [(indices_catA, indices_catB), ...] for each prob bin
    """
    n_trials, n_pos_bins = prob.shape
    cnd1 = np.zeros(n_trials)
    cnd1[cond1] = 1
    cnd2 = np.zeros(n_trials)
    cnd2[cond2] = 1
    matched_indices = []
    rng = np.random.default_rng(random_state)
    #for b in range(n_pos_bins):
    p = prob[:, b]
    bin_ids = np.digitize(p, bins) - 1 
    bin_matches = []

    for i in range(len(bins) - 1):
        idx_A = np.where((bin_ids == i) & (cnd1 == 1))[0]
        idx_B = np.where((bin_ids == i) & (cnd2 == 1))[0]

        n_match = min(len(idx_A), len(idx_B))
        if n_match > 0:
            #rng.shuffle(idx_A)
            #rng.shuffle(idx_B)
            bin_matches.append((idx_A[:n_match], idx_B[:n_match]))
        else:
            bin_matches.append((np.array([]), np.array([])))

    matched_indices.append(bin_matches)

    m_indices = np.array(matched_indices[0], dtype=object)

    return m_indices

def plot_prob_bins(prob, category, pbins=np.linspace(0, 1, 11), cbins=5):
    """
    Plot probability distributions for each category in each bin

    Parameters:
        prob: ndarray (trials, bins) – probability per trial and bin
        category: list of indices – trials from condition 1
        pbins: array-like – bin edges for digitizing probabilities
        cbins: int – number of corridor bins to plot
    """
    fig, ax = plt.subplots(1, cbins, figsize=(14, 3), sharex=True, sharey=True)
    for b in range(cbins):
        sns.histplot(prob[category==1,b], ax=ax[b], label=f"Cat A", alpha=0.5, color='green', bins=pbins, zorder=2)
        sns.histplot(prob[category==0,b], ax=ax[b], label=f"Cat B", alpha=0.5, color='red', bins=pbins)
        ax[b].set_ylabel("Counts")
        ax[b].set_title(f"Corridor Bin: {b}")
        ax[b].set_xlim(0, 1.009)
        ax[b].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()