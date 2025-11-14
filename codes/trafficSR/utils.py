import time
# print("torch.import_start_time", time.time())
import torch
# print(torch.__version__)
# print("torch.import_end_time", time.time())
import random
import numpy as np
import os
from datetime import datetime
import re
import concurrent.futures

colors = ['#C34A36', '#bcbd22', '#0081CF', '#845EC2', ]  # '#ff9999' pink
title_font = 13
lable_font = 13#11
axis_font = 12#10
legend_font = 11#9
myseed = 9

# Function to set timeout mechanism
def run_with_timeout_thread(func, args=(), kwargs={}, timeout_duration=5):
    # Set timeout signal handler
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit task
        future = executor.submit(func, *args, **kwargs)
        try:
            # Try to get the result within the specified time
            result = future.result(timeout=timeout_duration)
        except concurrent.futures.TimeoutError:
            print(f"Function {func.__name__} timed out after {timeout_duration} seconds") #还是会跳不出来，导致非常久的时间
            # Try to cancel the running function
            # Note: The cancellation of the function depends on whether the function checks the interrupt signal and exits accordingly. Some operations (such as IO operations, certain loops, etc.) may not immediately respond to the cancel request.
            future.cancel() #本质原因：Python 线程没有提供强制终止的 API
            return None  # You can return a specific value as needed

# 发现：调用 future.cancel() 对已经开始执行的任务实际上不会产生任何效果。cancel() 方法主要用于取消尚未开始执行的任务
def run_with_timeout_process(func, args=(), kwargs={}, timeout_duration=5):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_duration)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
            # future.cancel()  # 强制终止进程:这步太久
            return None

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def rebase_token_args(token_args):
    token_types = ["variable", "operator", "free_const", "fixed_const", "combination"]
    new_token_args = {}
    for token_type in token_types:

        for token_id in range(len(token_args[f"{token_type}_tokens"])):
            token_name = token_args[f"{token_type}_tokens"][token_id]
            if token_type == "combination":
                if (token_name not in token_args[f"{token_type}_description"].keys() or \
                        token_name not in token_args[f"{token_type}_units"].keys() or \
                        token_name not in token_args[f"{token_type}_prefix_expression"].keys()):
                    token_args[f"{token_type}_tokens"].remove(token_name)
                    break
            token_description = token_args[f"{token_type}_description"][token_name]
            token_unit = token_args[f"{token_type}_units"][token_name][:2] if token_type != "operator" else None
            token_prefix_expression = token_args[f"{token_type}_prefix_expression"][
                token_name] if token_type == "combination" else [token_name]
            new_token_args[token_name] = {
                "name": token_name,
                "description": token_description,
                "units": token_unit,
                "prefix_expression": token_prefix_expression,
                "type": token_type,
            }
    return new_token_args


def data_process_Helly_GHR(filter_dt=0.5,
                           file_folder="NGSIM_multiVehicle",
                           filename="MultiVehicle1.npy",
                           preceeding_ID='0',
                           following_ID='1',
                           last_ngsim_args=None,
                           last_X_array=None,
                           last_y_array=None,
                           ):
    data_source = file_folder + "/" + filename
    data = np.load(data_source, allow_pickle=True).item()
    car0_data = data[preceeding_ID]
    car1_data = data[following_ID]
    preceeding_length = car0_data['veh']
    # car1-following, car0-leading----->following distance
    delta_x = (car0_data['x'] - car1_data['x'])  # need to be car0-car1, because car0 before car1
    v = car1_data['v']  # car1-following
    delta_v = (car0_data['v'] - car1_data['v'])  # leading-following

    filter_length = int(filter_dt / 0.1)
    length = int(delta_x.shape[0] / filter_length)
    delta_x = delta_x[:length * filter_length].reshape(-1, filter_length)
    delta_x = np.mean(delta_x, axis=1)
    v = v[:length * filter_length].reshape(-1, filter_length)
    v = np.mean(v, axis=1)
    delta_v = delta_v[:length * filter_length].reshape(-1, filter_length)
    delta_v = np.mean(delta_v, axis=1)
    a_follow = (v[1:] - v[:-1]) / filter_dt

    X_array = np.stack((delta_x[:-1], v[:-1], delta_v[:-1]), axis=1)
    # X_array = np.stack((delta_x[:-1], v[:-1]), axis=1)
    y_array = a_follow
    X_array = torch.tensor(X_array).to("cpu")
    y_array = torch.tensor(y_array).to("cpu")

    x_lead = car0_data['x']
    x_lead = x_lead[:length * filter_length].reshape(-1, filter_length)
    x_lead = np.mean(x_lead, axis=1)[:-1]
    v_lead = car0_data['v']
    v_lead = v_lead[:length * filter_length].reshape(-1, filter_length)
    v_lead = np.mean(v_lead, axis=1)[:-1]
    x_obs = car1_data['x']
    x_obs = x_obs[:length * filter_length].reshape(-1, filter_length)
    x_obs = np.mean(x_obs, axis=1)[:-1]
    v_obs = car1_data['v']
    v_obs = v_obs[:length * filter_length].reshape(-1, filter_length)
    v_obs = np.mean(v_obs, axis=1)[:-1]

    ngsim_args = {}
    ngsim_args["stop_positions"] = [len(x_obs) - 1]
    ngsim_args["x_lead"] = torch.tensor(x_lead).to("cpu")
    ngsim_args["v_lead"] = torch.tensor(v_lead).to("cpu")
    ngsim_args["x_obs"] = torch.tensor(x_obs).to("cpu")
    ngsim_args["v_obs"] = torch.tensor(v_obs).to("cpu")
    ngsim_args["a_obs"] = torch.tensor(a_follow).to("cpu")
    ngsim_args["x_0"] = x_obs[0]
    ngsim_args["v_0"] = v_obs[0]
    ngsim_args["dt"] = filter_dt
    ngsim_args["preceeding_length"] = [preceeding_length[0]]

    if last_ngsim_args is not None:
        ngsim_args["x_lead"] = torch.cat(
            (last_ngsim_args["x_lead"], ngsim_args["x_lead"]), dim=0)
        ngsim_args["v_lead"] = torch.cat(
            (last_ngsim_args["v_lead"], ngsim_args["v_lead"]), dim=0)
        ngsim_args["x_obs"] = torch.cat(
            (last_ngsim_args["x_obs"], ngsim_args["x_obs"]), dim=0)
        ngsim_args["v_obs"] = torch.cat(
            (last_ngsim_args["v_obs"], ngsim_args["v_obs"]), dim=0)
        ngsim_args["a_obs"] = torch.cat((last_ngsim_args["a_obs"], ngsim_args["a_obs"]), dim=0)
        ngsim_args["x_0"] = ngsim_args["x_obs"][0]
        ngsim_args["v_0"] = ngsim_args["v_obs"][0]
        last_ngsim_args["stop_positions"].append(len(ngsim_args["x_obs"]) - 1)
        ngsim_args["stop_positions"] = last_ngsim_args["stop_positions"]
        last_ngsim_args["preceeding_length"].append(preceeding_length[0])
        ngsim_args["preceeding_length"] = last_ngsim_args["preceeding_length"]
        X_array = torch.cat((last_X_array, X_array), dim=0)
        y_array = torch.cat((last_y_array, y_array), dim=0)
    return X_array, y_array, ngsim_args, data_source


def data_process_IDM(filter_dt=0.5,
                     file_folder="NGSIM_multiVehicle",
                     filename="MultiVehicle1.npy",
                     preceeding_ID='0',
                     following_ID='1',
                     last_ngsim_args=None,
                     last_X_array=None,
                     last_y_array=None,
                     ):
    data_source = file_folder + "/" + filename
    data = np.load(data_source, allow_pickle=True).item()
    car0_data = data[preceeding_ID]
    car1_data = data[following_ID]
    preceeding_length = car0_data['veh']
    s = (car0_data['x'] - car1_data['x'] - preceeding_length)  # car1-following, car0-leading----->following distance
    # need to be car0-car1, because car0 before car1
    v = car1_data['v']  # car1-following
    delta_v = (car1_data['v'] - car0_data['v'])

    filter_length = int(filter_dt / 0.1)
    length = int(s.shape[0] / filter_length)
    s = s[:length * filter_length].reshape(-1, filter_length)
    s = np.mean(s, axis=1)
    v = v[:length * filter_length].reshape(-1, filter_length)
    v = np.mean(v, axis=1)
    delta_v = delta_v[:length * filter_length].reshape(-1, filter_length)
    delta_v = np.mean(delta_v, axis=1)
    a_follow = (v[1:] - v[:-1]) / filter_dt

    X_array = np.stack((s[:-1], v[:-1], delta_v[:-1]), axis=1)
    # X_array = np.stack((s[:-1], v[:-1]), axis=1)
    y_array = a_follow
    X_array = torch.tensor(X_array).to("cpu")
    y_array = torch.tensor(y_array).to("cpu")

    x_lead = car0_data['x']
    x_lead = x_lead[:length * filter_length].reshape(-1, filter_length)
    x_lead = np.mean(x_lead, axis=1)[:-1]
    v_lead = car0_data['v']
    v_lead = v_lead[:length * filter_length].reshape(-1, filter_length)
    v_lead = np.mean(v_lead, axis=1)[:-1]
    x_obs = car1_data['x']
    x_obs = x_obs[:length * filter_length].reshape(-1, filter_length)
    x_obs = np.mean(x_obs, axis=1)[:-1]
    v_obs = car1_data['v']
    v_obs = v_obs[:length * filter_length].reshape(-1, filter_length)
    v_obs = np.mean(v_obs, axis=1)[:-1]

    ngsim_args = {}
    ngsim_args["stop_positions"] = [len(x_obs) - 1]
    ngsim_args["x_lead"] = torch.tensor(x_lead).to("cpu")
    ngsim_args["v_lead"] = torch.tensor(v_lead).to("cpu")
    ngsim_args["x_obs"] = torch.tensor(x_obs).to("cpu")
    ngsim_args["v_obs"] = torch.tensor(v_obs).to("cpu")
    ngsim_args["a_obs"] = torch.tensor(a_follow).to("cpu")
    ngsim_args["x_0"] = x_obs[0]
    ngsim_args["v_0"] = v_obs[0]
    ngsim_args["dt"] = filter_dt
    ngsim_args["preceeding_length"] = [preceeding_length[0]]

    if last_ngsim_args is not None:
        ngsim_args["x_lead"] = torch.cat(
            (last_ngsim_args["x_lead"], ngsim_args["x_lead"]), dim=0)
        ngsim_args["v_lead"] = torch.cat(
            (last_ngsim_args["v_lead"], ngsim_args["v_lead"]), dim=0)
        ngsim_args["x_obs"] = torch.cat(
            (last_ngsim_args["x_obs"], ngsim_args["x_obs"]), dim=0)
        ngsim_args["v_obs"] = torch.cat(
            (last_ngsim_args["v_obs"], ngsim_args["v_obs"]), dim=0)
        ngsim_args["a_obs"] = torch.cat((last_ngsim_args["a_obs"], ngsim_args["a_obs"]), dim=0)
        ngsim_args["x_0"] = ngsim_args["x_obs"][0]
        ngsim_args["v_0"] = ngsim_args["v_obs"][0]
        last_ngsim_args["stop_positions"].append(len(ngsim_args["x_obs"]) - 1)
        ngsim_args["stop_positions"] = last_ngsim_args["stop_positions"]
        last_ngsim_args["preceeding_length"].append(preceeding_length[0])
        ngsim_args["preceeding_length"] = last_ngsim_args["preceeding_length"]
        X_array = torch.cat((last_X_array, X_array), dim=0)
        y_array = torch.cat((last_y_array, y_array), dim=0)
    return X_array, y_array, ngsim_args, data_source


def extract_numbers_from_string(s):
    # Finding all numbers in a string using regular expressions
    numbers = re.findall(r'\d+', s)
    # Convert found numbers to a list of integers
    return [num for num in numbers]


def ngsim_single_trajectory(filter_dt=0.5, file_folder=None, filename=None, data_process_algorithm=data_process_IDM):
    if filename is None:
        preceeding_ID = '40'
        following_ID = '45'
    else:
        extracted_numbers = extract_numbers_from_string(filename)
        following_ID = str(extracted_numbers[0])
        preceeding_ID = str(extracted_numbers[1])
    X_array, y_array, ngsim_args, data_source = data_process_algorithm(filter_dt=filter_dt,
                                                                       file_folder="../../data/NGSIM_dataNpy/trajectories-0820am-0835am" if file_folder is None else file_folder,
                                                                       filename="ego45_preceeding40.npy" if filename is None else filename,
                                                                       preceeding_ID=preceeding_ID,
                                                                       following_ID=following_ID)  # 67.5s
    ngsim_args["n_data_sources"] = len(ngsim_args["stop_positions"])
    return X_array, y_array, ngsim_args, data_source, following_ID, preceeding_ID


def ngsim_filefolder_trajectory(filter_dt=0.5,
                                filefolder_list=["../../data/NGSIM_dataNpy/trajectories-0820am-0835am"],
                                need_first_K=-1,
                                data_process_algorithm=data_process_IDM):
    X_array, y_array, ngsim_args, data_source = torch.tensor([]), torch.tensor([]), {}, ""

    def get_all_trajectory_in_file(file_folder="NGSIM_multiVehicle", filename="MultiVehicle1.npy"):
        nonlocal X_array, y_array, ngsim_args, data_source
        now_data_source = file_folder + "/" + filename
        data = np.load(now_data_source, allow_pickle=True).item()
        data_keys = list(data.keys())
        for i in range(len(data_keys) - 1):
            if ngsim_args == {}:
                X_array, y_array, ngsim_args, data_source = data_process_algorithm(filter_dt=filter_dt,
                                                                                   file_folder=file_folder,
                                                                                   filename=filename,
                                                                                   preceeding_ID=data_keys[i],
                                                                                   following_ID=data_keys[i + 1])
            else:
                X_array, y_array, ngsim_args, data_source = data_process_algorithm(filter_dt=filter_dt,
                                                                                   file_folder=file_folder,
                                                                                   filename=filename,
                                                                                   preceeding_ID=data_keys[i],
                                                                                   following_ID=data_keys[i + 1],
                                                                                   last_ngsim_args=ngsim_args,
                                                                                   last_X_array=X_array,
                                                                                   last_y_array=y_array)

    def get_all_trajectory_in_filefolder(filefolder="NGSIM_multiVehicle", need_first_K=-1):
        all_files = os.listdir(filefolder)
        need_files = all_files[:need_first_K] if need_first_K != -1 else all_files
        for file in need_files:
            if file.endswith(".npy"):
                print("file_name", file)
                get_all_trajectory_in_file(file_folder=filefolder, filename=file)

    for filefolder in filefolder_list:
        get_all_trajectory_in_filefolder(filefolder=filefolder, need_first_K=need_first_K)
    ngsim_args["n_data_sources"] = len(ngsim_args["stop_positions"])
    return X_array, y_array, ngsim_args, "AllMultiVehicle"


def calculate_minutes_between_times(start_hour, start_minute, start_second, end_hour, end_minute, end_second):
    start_time = datetime.strptime(f"{start_hour}:{start_minute}:{start_second}", "%H:%M:%S")
    end_time = datetime.strptime(f"{end_hour}:{end_minute}:{end_second}", "%H:%M:%S")
    time_difference = end_time - start_time
    return time_difference.total_seconds() / 60


def is_single_or_double_layer_list(prog):
    if not isinstance(prog, list):
        return "Not a list"

    if all(isinstance(item, list) for item in prog):
        return "Double-layer nested list"
    elif all(not isinstance(item, list) for item in prog):
        return "Single-layer list"
    else:
        return "Mixed or other structure"
