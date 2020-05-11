import os
import csv
import flow_shop_SL as fs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import simpy
from tqdm import tqdm

input_dir = os.getcwd() + '/input_data/'
out_dir   = os.getcwd() + '/assets/'
train_dir = out_dir + 'training_data/'
test_dir  = out_dir + 'testing_data/'
dir_list  = [input_dir, train_dir, test_dir]

for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)
#=============================================================
class Factory(fs.Factory):
    def __init__(self, env, job_info, AV, DD, PT_table,
                 ST_table, ME_table, DP_rule, log=False):
        super().__init__(env, job_info, AV, DD, PT_table,
                         ST_table, ME_table, DP_rule, log)
#=============================================================
# standard data information
job_info = pd.read_excel(input_dir + "job_information.xlsx")
PT_table = pd.read_excel(input_dir + "PT_table.xlsx")
ST_table = pd.read_excel(input_dir + "ST_table.xlsx")
ME_table = pd.read_excel(input_dir + "ME_table.xlsx")
# parameters set
simulationTime = 300
DP_rule_lst = ['MST', 'EDD', 'FIFO']
AV_interval = 5
DD_factor   = 10
# labels
train_labels = []
test_labels  = []

for repeat in range(1):
    # parameters set
    simulationTime = 100
    DP_rule = np.random.choice(DP_rule_lst);print(DP_rule)
    AV_interval = np.random.randint(0, high=50);print(AV_interval)
    DD_factor   = np.random.randint(100, high=200);print(DD_factor)
    # simulation environment
    env = simpy.Environment()
    fac = Factory(env, job_info, AV_interval, DD_factor, PT_table,
                  ST_table, ME_table, DP_rule, log=True)
    fac.initialize()
    # run simulation
    env.run(until = simulationTime)
    # results
    throughput = fac.throughput
    makespan = fac.env.now
    avgFT = np.mean(fac.sink.job_statistic['flow_time'])
    avgWIP = fac.WIP_area / fac.env.now
    # print("====================================================================")
    # print("makespan: {}".format(makespan))
    # print("Average folw time: {}".format(avgFT))
    # print("Aveage WIP: {}".format(avgWIP))
    # print("throughput: {}".format(throughput))
    # print("====================================================================")
    # print()
    # lable set
    lbl = {'AVInterval': AV_interval, 'DD_factor': DD_factor,
           'DP_rule': DP_rule, 'throughput': throughput,
           'avgFT': avgFT, 'avgWIP': avgWIP}

    if repeat >= 70:
        train_labels.append(lbl)
    else:
        test_labels.append(lbl)

train_labels = pd.DataFrame(train_labels)
test_labels  = pd.DataFrame(test_labels)
train_labels.to_csv(train_dir + 'train_labels', index=0)
test_labels.to_csv(test_dir + 'test_labels',  index=0)
