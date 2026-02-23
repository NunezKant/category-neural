import numpy as np
import pandas as pd

def get_sessions():
    """
    Get all sessions from the database
    """
    VG11 = []
    VG11.append({'mname': 'VG11', 'datexp': '2024_10_15', 'blk':'4','session': 'all rewarded before', "round_id": 1, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_10_16', 'blk':'2','session': 'first training', "round_id": 1, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_10_31', 'blk':'2','session': 'last training', "round_id": 1, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_01', 'blk':'2','session': 'all rewarded after', "round_id": 1, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_04', 'blk':'2','session': 'all rewarded before', "round_id": 2, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_05', 'blk':'3','session': 'first training', "round_id": 2, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_14', 'blk':'2','session': 'last training', "round_id": 2, 'recording': True, 'obj': "old"})
    VG11.append({'mname': 'VG11', 'datexp': '2024_11_15', 'blk':'2','session': 'all rewarded after', "round_id": 2, 'recording': True, 'obj': "old"})
    VG14 = []
    VG14.append({'mname': 'VG14', 'datexp': '2024_10_15', 'blk':'2','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG14.append({'mname': 'VG14', 'datexp': '2024_10_16', 'blk':'2','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG14.append({'mname': 'VG14', 'datexp': '2024_11_21', 'blk':'2','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG14.append({'mname': 'VG14', 'datexp': '2024_11_23', 'blk':'2','session': 'all rewarded after', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG15 = []
    VG15.append({'mname': 'VG15', 'datexp': '2024_10_15', 'blk':'3','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG15.append({'mname': 'VG15', 'datexp': '2024_10_16', 'blk':'2','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG15.append({'mname': 'VG15', 'datexp': '2024_10_31', 'blk':'2','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG15.append({'mname': 'VG15', 'datexp': '2024_11_01', 'blk':'3','session': 'all rewarded after', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG21 = []
    VG21.append({'mname': 'VG21', 'datexp': '2025_06_24', 'blk':'3','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_06_25', 'blk':'2','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_07_17', 'blk':'3','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_07_18', 'blk':'2','session': 'all rewarded after', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_07_21', 'blk':'2','session': 'all rewarded before', 'round_id': 2, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_07_22', 'blk':'2','session': 'first training', 'round_id': 2, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_08_07', 'blk':'2','session': 'last training', 'round_id': 2, 'recording': True, 'obj': "old"})
    VG21.append({'mname': 'VG21', 'datexp': '2025_08_08', 'blk':'2','session': 'all rewarded after', 'round_id': 2, 'recording': True, 'obj': "old"})
    VG24 = []
    VG24.append({'mname': 'VG24', 'datexp': '2025_06_25', 'blk':'2','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG24.append({'mname': 'VG24', 'datexp': '2025_06_26', 'blk':'2','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG24.append({'mname': 'VG24', 'datexp': '2025_07_10', 'blk':'2','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG24.append({'mname': 'VG24', 'datexp': '2025_07_14', 'blk':'2','session': 'all rewarded after', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG33 = []
    VG33.append({'mname': 'VG33', 'datexp': '2025_09_30', 'blk':'2','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG33.append({'mname': 'VG33', 'datexp': '2025_10_01', 'blk':'2','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG33.append({'mname': 'VG33', 'datexp': '2025_10_09', 'blk':'3','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG33.append({'mname': 'VG33', 'datexp': '2025_10_10', 'blk':'2','session': 'all rewarded after', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG34 = []
    VG34.append({'mname': 'VG34', 'datexp': '2025_09_30', 'blk':'3','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG34.append({'mname': 'VG34', 'datexp': '2025_10_01', 'blk':'2','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG34.append({'mname': 'VG34', 'datexp': '2025_10_10', 'blk':'3','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "old"})
    VG34.append({'mname': 'VG34', 'datexp': '2025_10_13', 'blk':'4','session': 'all rewarded after', 'round_id': 1, 'recording': False, 'obj': "old"})
    VGa9 = []
    VGa9.append({'mname': 'VGa9', 'datexp': '2025_11_06', 'blk':'4','session': 'all rewarded before', 'round_id': 1, 'recording': True, 'obj': "new"})
    VGa9.append({'mname': 'VGa9', 'datexp': '2025_11_07', 'blk':'3','session': 'first training', 'round_id': 1, 'recording': True, 'obj': "new"})
    VGa9.append({'mname': 'VGa9', 'datexp': '2025_11_20', 'blk':'2','session': 'last training', 'round_id': 1, 'recording': True, 'obj': "new"})
    VGa9.append({'mname': 'VGa9', 'datexp': '2025_11_21', 'blk':'2','session': 'all rewarded after', 'round_id': 1, 'recording': True, 'obj': "new"})
    all_sessions = VG11 + VG14 + VG15 + VG21 + VG24 + VG33 + VG34 + VGa9
    all_sessions = pd.DataFrame(all_sessions)
    return all_sessions