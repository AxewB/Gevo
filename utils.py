import math
import numpy as np
import shutil
import os
import re

from config import PARAM_FILENAME, RESULT_FILENAME_PATTERN, OUTPUT_PATH
from typing import List, Union


def sigma(state_histories: np.ndarray) -> np.ndarray:
    sig = state_histories.sum(axis=1) / state_histories.shape[1]
    return sig


def sigmas(state_histories: List[np.ndarray]) -> List[np.ndarray]:
    return [sigma(sh) for sh in state_histories]


def get_outcome(state_history: np.ndarray) -> bool:
    if not state_history.any():
        return False
    final_state_mean = np.round(state_history[-1].sum() / state_history.shape[1], decimals=0)
    for outcome in [True, False]:
        if np.isclose(final_state_mean, int(outcome)):
            return outcome
    else:
        raise RuntimeError("final state is non-homogenous")


def get_outcomes(state_histories: List[np.ndarray]) -> np.ndarray[bool]:
    return np.array([get_outcome(sh) for sh in state_histories])


def pad(arrays: List[np.ndarray], pad_to: Union[int, None] = None) -> np.ndarray:
    outcomes = [np.round(a[-1], decimals=0) for a in arrays]
    assert all(
        np.isclose(outcome, 0.) or
        np.isclose(outcome, 1.)
        for outcome in outcomes
    ), "found outcome neither 0 nor 1"

    num_arrays = len(arrays)
    sizes = [a.shape[0] for a in arrays]
    if pad_to is None:
        pad_to = max(sizes)
    else:
        assert pad_to >= max(sizes), "`pad_to` should surpass max array size"

    arr = np.zeros(shape=(num_arrays, pad_to), dtype=np.float64)
    for i, (size, array, outcome) in enumerate(zip(sizes, arrays, outcomes)):
        arr[i, :size] = array
        arr[i, size:] = outcome

    return arr


def sigma_mean(state_histories: List[np.ndarray], pad_to: Union[int, None] = None) -> np.ndarray:
    return pad(sigmas(state_histories), pad_to=pad_to).mean(axis=0)


def pad_2d(arrays: List[np.ndarray]) -> np.ndarray:
    """
    pads 2D numpy arrays to common number of rows.
    """
    assert len({a.shape[1] for a in arrays}) == 1, "arrays in `arr_list` should have equal number of columns"
    outcomes = [int(np.round(a[-1].sum() / a.shape[1], decimals=0)) for a in arrays]
    assert all(outcome in [0, 1] for outcome in outcomes), "some arrays are not completed"

    nums_rows = [a.shape[0] for a in arrays]
    num_rows = max(nums_rows)
    num_cols = arrays[0].shape[1]
    num_arrays = len(arrays)

    arr = np.zeros(shape=(num_arrays, num_rows, num_cols), dtype=bool)
    for i, (num_rows, array, outcome) in enumerate(zip(nums_rows, arrays, outcomes)):
        arr[i, :num_rows, :] = array
        arr[i, num_rows:, :] = outcome

    return arr


def digits(num: Union[int, float]):
    if num < 0:
        raise ValueError("`num` should be positive.")
    elif num == 0:
        return 1
    else:
        return math.floor(math.log10(num)) + 1

''' Author: Aksenov IV
Функция для слияния нескольких симуляций из разных папок
Если есть несколько ранов, в которых по несколько симуляций, эта
функция их объединит в общие папки
'''
def merge_sim(cur_paths: List[str], type_path: str, outpath: str = OUTPUT_PATH):
    out_dir = outpath + "\\merge"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    if len(cur_paths) == 0:
        raise NotImplementedError("Empty paths!!!")
    
    for cur_path in cur_paths:
        all_paths = [
            cur_path + "\\" + run_path + "\\" + exp_path + "\\" + edge_path + "\\" + graph_path + "\\" + sim_path
            for run_path in os.listdir(cur_path)
            if run_path != "merged"
            for exp_path in os.listdir(cur_path + "\\" + run_path)
            for edge_path in os.listdir(cur_path + "\\" + run_path + "\\" + exp_path)
            for graph_path in os.listdir(cur_path + "\\" + run_path + "\\" + exp_path + "\\" + edge_path)
            for sim_path in os.listdir(cur_path + "\\" + run_path + "\\" + exp_path + "\\" + edge_path + "\\" + graph_path)
        ]

    suitable_paths = [
        path 
        for path in all_paths
        if (type_path in path)
    ]

    divided_path = type_path.split('\\')[:-1]
    
    for i, path in enumerate(divided_path):
        out_dir += "\\" + path
        if not os.path.isdir(out_dir):
            print(out_dir)
            os.mkdir(out_dir)
    
    size = 0
    for path in suitable_paths:
        for fname in os.listdir(path):
            if re.match(RESULT_FILENAME_PATTERN, fname):
                size += 1
    
    
    fmt_str = f'0{digits(size - 1)}d'
    
    
    cnt = 0
    for path in suitable_paths:
        sim_fname = '_'.join(path.split('\\')[-1].split("_")[2:])
        if not os.path.isdir(out_dir + "\\" + sim_fname):
            os.mkdir(out_dir + "\\" + sim_fname)
            shutil.copyfile(suitable_paths[0] + "\\" + PARAM_FILENAME, out_dir + "\\" + sim_fname + "\\" + PARAM_FILENAME)  
        fnames = [
            fname
            for fname in os.listdir(path)
            if re.match(RESULT_FILENAME_PATTERN, fname)
        ]
        for fname in fnames:
            shutil.copyfile(path + '\\' + fname, out_dir + '\\' + sim_fname + '\\' + f"{cnt:{fmt_str}}" + ".pickle")
            cnt += 1
