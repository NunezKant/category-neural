from pathlib import Path
import os
from tqdm import tqdm
import sys

sys.path.insert(0, r"C:\Users\labadmin\Documents\suite2p")
sys.path.insert(0, r"C:\Users\labadmin\Documents\rastermap")
from scipy.stats import zscore
from src import utils  # this is our own library of functions
from src.utils import Mouse
from src.plots import rastermap_plot
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def bin1d(X, bin_size, axis=0):
    """bin over axis of data with bin bin_size"""
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = (
            Xb[: size[axis] // bin_size * bin_size]
            .reshape((size[axis] // bin_size, bin_size, -1))
            .mean(axis=1)
        )
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X


def autoraster(filename, neuron_bin_size=50, dual_plane=True, format="png"):
    splitted = filename.split("/")
    mname = splitted[3]
    mdate = splitted[4]
    mblock = splitted[5]

    parentpath = "Z:/data/autoraster/"
    directory = f"{mname}/{mdate}/{mblock}"

    plots_pth = Path(os.path.join(parentpath, directory))
    if os.path.isdir(plots_pth) == False:
        print(f"Creating directory {plots_pth}")
        os.makedirs(plots_pth)
    Mouse1 = Mouse(mname, mdate, mblock)
    Mouse1.load_behav(timeline_block=None)
    Mouse1.get_timestamps()
    Mouse1.get_trial_info()
    print("loading neurons ...")
    Mouse1.load_neurons_VG(dual_plane=True)
    nframes = Mouse1._spks.shape[1]
    chunksize = 500
    nchunks = int(nframes / chunksize) + 1
    last_chunk_size = nframes - int(nframes / chunksize) * chunksize
    mdl_pth = Path(
        os.path.join(Path("C:/Users/labadmin/Documents/models/rastermaps"), directory)
    )
    if os.path.isdir(mdl_pth) == False:
        print(f"Creating directory {mdl_pth}")
        os.makedirs(mdl_pth)
    mdl_pth = Path(os.path.join(mdl_pth, "rastermap_model.npy"))
    if mdl_pth.is_file() == True:
        print(f"Loading rastermap model from {mdl_pth}")
        with open(mdl_pth, "rb") as file:
            model = pickle.load(file)
    else:
        print("fitting rastermap ... ")
        model = utils.get_rastermap(Mouse1, n_comp=200)
        print(f"Rastermap fitted, saving model to {mdl_pth}")
        with open(mdl_pth, "wb") as file:
            pickle.dump(model, file)
    if  dual_plane == True:
        for layer in [1, 2]:
            dr = f"Layer{layer}"
            layer_pth = Path(os.path.join(plots_pth, dr))
            if os.path.isdir(layer_pth) == False:
                print(f"Creating directory {layer_pth}")
                os.mkdir(layer_pth)
            print(f"Creating plots for {nchunks} rastermaps in layer {layer}:")
            print(
                f"each with {chunksize} frames (except last one with {last_chunk_size} frames)"
            )
            print(f"Plots will be saved to {layer_pth}")
            if layer == 1:
                layer_isort = model.isort[Mouse1._iplane[model.isort] >= 10]
                neuron_embedding = zscore(
                    bin1d(Mouse1._spks[layer_isort], neuron_bin_size), axis=1
                )
            elif layer == 2:
                layer_isort = model.isort[Mouse1._iplane[model.isort] < 10]
                neuron_embedding = zscore(
                    bin1d(Mouse1._spks[layer_isort], neuron_bin_size), axis=1
                )
            for chunk in tqdm(range(nchunks)):
                if chunk == (nchunks - 1):
                    rastermap_plot(
                        Mouse1,
                        neuron_embedding,
                        frame_selection=chunk,
                        frame_num=last_chunk_size,
                        svefig=True,
                        savepath=layer_pth,
                        format=format,
                    )
                else:
                    rastermap_plot(
                        Mouse1,
                        neuron_embedding,
                        frame_selection=chunk,
                        frame_num=chunksize,
                        svefig=True,
                        savepath=layer_pth,
                        format=format,
                    )
        print("Creating full rastermap")
        dr = "Full"
        full_pth = Path(os.path.join(plots_pth, dr))
        if os.path.isdir(full_pth) == False:
            print(f"Creating directory {full_pth}")
            os.mkdir(full_pth)
        print(f"Plots will be saved to {full_pth}")
        for chunk in tqdm(range(nchunks)):
            if chunk == (nchunks - 1):
                rastermap_plot(
                    Mouse1,
                    model.X_embedding,
                    frame_selection=chunk,
                    frame_num=last_chunk_size,
                    svefig=True,
                    savepath=full_pth,
                    format=format,
                )
            else:
                rastermap_plot(
                    Mouse1,
                    model.X_embedding,
                    frame_selection=chunk,
                    frame_num=chunksize,
                    svefig=True,
                    savepath=full_pth,
                    format=format,
                )
    else:
        print("Creating full rastermap")
        dr = "Full"
        full_pth = Path(os.path.join(plots_pth, dr))
        if os.path.isdir(full_pth) == False:
            print(f"Creating directory {full_pth}")
            os.mkdir(full_pth)
        print(f"Plots will be saved to {full_pth}")
        for chunk in tqdm(range(nchunks)):
            if chunk == (nchunks - 1):
                rastermap_plot(
                    Mouse1,
                    model.X_embedding,
                    frame_selection=chunk,
                    frame_num=last_chunk_size,
                    svefig=True,
                    savepath=full_pth,
                    format=format,
                )
            else:
                rastermap_plot(
                    Mouse1,
                    model.X_embedding,
                    frame_selection=chunk,
                    frame_num=chunksize,
                    svefig=True,
                    savepath=full_pth,
                    format=format,
                )


if __name__ == "__main__":

    Tk().withdraw()
    filename = askopenfilename()
    autoraster(filename)
