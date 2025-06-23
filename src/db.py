import numpy as np
import pandas as pd

def get_sessions():
    """
    Get all sessions from the database
    """
    VG11 = []
    VG11.append({'mname': 'VG11', 'datexp': '2024_10_15', 'blk':'4','session': 'all rewarded before'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_10_16', 'blk':'2','session': 'first training'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_10_31', 'blk':'2', 'session': 'last training'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_01', 'blk':'2','session': 'all rewarded after'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_04', 'blk':'2','session': 'all rewarded before'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_05', 'blk':'3','session': 'first training'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_14', 'blk':'2','session': 'last training'})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_15', 'blk':'2','session': 'all rewarded after'})
    VG14 = []
    VG14.append({'mname': 'VG14', 'datexp': '2024_10_15', 'blk':'2','session': 'all rewarded before'})
    VG14.append({'mname': 'VG14', 'datexp': '2024_10_16', 'blk':'2','session': 'first training'})
    VG14.append({'mname': 'VG14', 'datexp': '2024_11_21', 'blk':'2','session': 'last training'})
    VG14.append({'mname': 'VG14', 'datexp': '2024_11_23', 'blk':'2','session': 'all rewarded after'})
    VG15 = []
    VG15.append({'mname': 'VG15', 'datexp': '2024_10_15', 'blk':'3','session': 'all rewarded before'})
    VG15.append({'mname': 'VG15', 'datexp': '2024_10_16', 'blk':'2','session': 'first training'})
    VG15.append({'mname': 'VG15', 'datexp': '2024_10_31', 'blk':'2','session': 'last training'})
    VG15.append({'mname': 'VG15', 'datexp': '2024_11_01', 'blk':'3','session': 'all rewarded after'})
    all_sessions = VG11 + VG14 + VG15
    all_sessions
    return pd.DataFrame(all_sessions)