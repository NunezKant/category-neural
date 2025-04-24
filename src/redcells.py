import numpy as np
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from suite2p import default_ops
from suite2p.io import tiff
import imp
imp.reload(tiff)
import json
from suite2p.registration import register
import contextlib 
from suite2p import io
from natsort import natsorted 
import imp
imp.reload(register)

def choose_path(title):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

def path_two_channels():
    red_path = choose_path("Choose the red channel TIF folder")
    print("RED CHANNEL: selected Directory:", red_path)
    green_path = choose_path("Choose the green channel SUITE2P folder")
    print("GREEN CHANNEL: selected Directory:", green_path)
    s2p_green = Path(green_path)
    s2p_red = Path(red_path)
    return s2p_green, s2p_red

def tiffs_to_binary(s2p_red):
    ### convert red/green tiffs to binary files
    ops = default_ops()
    ops["data_path"] = [s2p_red]
    ops["nchannels"] = 2
    ops["save_path0"] = str(ops["data_path"][0])

    #ops = tiff.tiff_to_binary(ops)
    ops = tiff.mesoscan_to_binary(ops)
    return ops

### align red/green recording to green recording
def align_to_green(ops, s2p_green):
    # get plane folders
    ops["save_folder"] = "suite2p"
    save_folder = os.path.join(ops["save_path0"], ops["save_folder"])
    plane_folders = natsorted(
        [
            f.path
            for f in os.scandir(save_folder)
            if f.is_dir() and f.name[:5] == "plane"
        ]
    )
    ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    nplanes = len(ops_paths)

    # get reference images from long green recording
    ops_paths_green = [str(s2p_green / f"plane{ipl}" / "ops.npy") 
                    for ipl in range(nplanes)]
    refImgs = [np.load(ops_path, allow_pickle=True).item()["meanImg"] 
            for ops_path in ops_paths_green]

    # loop over planes
    align_by_chan2 = False
    for ipl, ops_path in enumerate(ops_paths):
        print(ops_path)
        if ipl in ops["ignore_flyback"]:
            print(">>>> skipping flyback PLANE", ipl)
            continue
        else:
            print(">>>> registering PLANE", ipl)
        ops = np.load(ops_path, allow_pickle=True).item()
        # get binary file paths
        raw = ops.get("keep_movie_raw") and "raw_file" in ops and os.path.isfile(
            ops["raw_file"])
        reg_file = ops["reg_file"]
        raw_file = ops.get("raw_file", 0) if raw else reg_file
        # get number of frames in binary file to use to initialize files if needed
        if ops["nchannels"] > 1:
            reg_file_chan2 = ops["reg_file_chan2"]
            raw_file_chan2 = ops.get("raw_file_chan2", 0) if raw else reg_file_chan2
        else:
            reg_file_chan2 = reg_file
            raw_file_chan2 = reg_file

        # shape of binary files
        n_frames, Ly, Lx = ops["nframes"], ops["Ly"], ops["Lx"]

        null = contextlib.nullcontext()
        twoc = ops["nchannels"] > 1

        with io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file, n_frames=n_frames) \
            if raw else null as f_raw, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file, n_frames=n_frames) as f_reg, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2, n_frames=n_frames) \
            if raw and twoc else null as f_raw_chan2,\
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2, n_frames=n_frames) \
            if twoc else null as f_reg_chan2:
                f_alt_in, f_align_out, f_alt_out = None, None, None
                print(ops["ops_path"], f_reg.filename)
                registration_outputs = register.registration_wrapper(
                    f_reg, f_raw=f_raw, f_reg_chan2=f_reg_chan2, f_raw_chan2=f_raw_chan2,
                    refImg=refImgs[ipl], align_by_chan2=align_by_chan2, ops=ops)

                ops = register.save_registration_outputs_to_ops(registration_outputs, ops)
                
                meanImgE = register.compute_enhanced_mean_image(
                ops["meanImg"].astype(np.float32), ops)
                ops["meanImgE"] = meanImgE
        np.save(ops["ops_path"], ops)
        return ops_paths, ops_paths_green, nplanes

def check_alignment(ops_paths, ops_paths_green, nplanes):
    ### check alignment
    import matplotlib.pyplot as plt
    from cellpose.transforms import normalize99

    fig = plt.figure(figsize=(12,12))

    for ipl in range(nplanes):
        ops = np.load(ops_paths[ipl], allow_pickle=True).item()
        ops_green = np.load(ops_paths_green[ipl], allow_pickle=True).item()
        plt.subplot(nplanes, 4, 1 + ipl*4)
        plt.imshow(normalize99(ops_green["meanImg"]), vmin=0, vmax=1, cmap="gray")
        plt.title(f"plane{ipl}, meanImg")
        plt.axis("off")
        
        plt.subplot(nplanes, 4, 2 + ipl*4)
        plt.imshow(normalize99(ops["meanImg"]), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title(f"from green/red, meanImg")
        
        rgb = np.zeros((*ops["meanImg"].shape, 3))
        rgb[:,:,1] = np.clip(normalize99(ops["meanImg"]), 0, 1)
        rgb[:,:,2] = np.clip(normalize99(ops_green["meanImg"]), 0, 1)
        plt.subplot(nplanes, 4, 3 + ipl*4)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(f"overlaid meanImg")
        
        plt.subplot(nplanes, 4, 4 + ipl*4)
        plt.imshow(normalize99(ops["meanImg_chan2"]), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        plt.title(f"from green/red, meanImg_chan2")
    plt.tight_layout()
    plt.show()

def overlap_with_green(s2p_green, ops_paths, ops_paths_green, nplanes):
    from suite2p.io.save import combined
    from suite2p.detection import chan2detect, anatomical
    imp.reload(anatomical)
    imp.reload(chan2detect)

    stat_paths_green = [str(s2p_green / f"plane{ipl}" / "stat.npy") 
                    for ipl in range(nplanes)]
    redcell_paths_green = [str(s2p_green / f"plane{ipl}" / "redcell.npy") 
                    for ipl in range(nplanes)]

    for ipl, ops_path in enumerate(ops_paths):
        print(ops_path)
        ops = np.load(ops_path, allow_pickle=True).item()
        stat = np.load(stat_paths_green[ipl], allow_pickle=True)

        ops, redstats = chan2detect.detect(ops, stat)

        np.save(ops_path, ops)
        
        opsg = np.load(ops_paths_green[ipl], allow_pickle=True).item()
        opsg["meanImg_chan2"] = ops["meanImg_chan2"]
        opsg["meanImg_chan2_corrected"] = ops["meanImg_chan2_corrected"]
        opsg["nchannels"] = 2

        np.save(ops_paths_green[ipl], opsg)
        np.save(redcell_paths_green[ipl], redstats)
    #combined(str(s2p_green));

def get_redcells(s2p_green):
    root = s2p_green
    isredcell = np.zeros((0,2))
    ops = np.load(
        os.path.join(root, "plane0", "ops.npy"), allow_pickle=True
    ).item()
    for n in range(ops["nplanes"]):
        redcell0 = np.load(os.path.join(root, "plane%d" % n, "redcells.npy"), allow_pickle=True
    ) #redcells.npy is the output from red intensity ratio, redcell is the output from nn detection
        isredcell = np.concatenate((isredcell, redcell0), axis=0)
    print(isredcell.shape)
    return isredcell
    


# Example usage
if __name__ == "__main__":
    s2p_green, s2p_red = path_two_channels()
    ops = tiffs_to_binary(s2p_red)
    ops_paths, ops_paths_green, nplanes = align_to_green(ops, s2p_green)
    overlap_with_green(s2p_green, ops_paths, ops_paths_green, nplanes)