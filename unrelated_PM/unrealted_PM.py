# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:00:31 2019

@author: lonsichang
"""


import os
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#entity
class Job:
    def __init__(self, ID, jtype, AT):
        self.ID = ID
        self.jtype = jtype
        self.AT = AT    #AT: arrival time
        self.CT = None    #CT: complete time

#resource in factory
class Source:
    def __init__(self, fac, job_info):
        self.fac = fac
        self.job_info = job_info    #job_info: job information

    def initialize(self, OP):
        #reference
        self.env = self.fac.env
        self.OP = OP    #OP: output port, connect to a queue
        #attribute
        #statistics
        self.output = 0
        #initial process
        self.process = self.env.process(self.arrival())

    def arrival(self):
        job_num = self.job_info.shape[0]
        for i in range(job_num):
            #wait until job arrive
            arrival_interval = self.job_info.loc[self.output, "arrival_interval"]
            #arrival_interval = np.random.exponential(arrival_interval)
            yield self.env.timeout(arrival_interval)
            #generate job
            ID      = self.job_info.loc[self.output, "ID"]
            jtype = self.job_info.loc[self.output, "type"]
            AT      = self.env.now
            job   = Job(ID, jtype, AT)

            if self.fac.log:
                print("{} job {} release.".format(self.env.now, job.ID))

            self.output += 1
            #send job to next station
            self.OP.job_arrive(job)

            self.fac.update_WIP(1)
            '''
            event_trace
            '''

class Queue:
    def __init__(self, fac, stage, DP_rule):
        self.fac = fac
        self.stage = stage
        self.DP_rule = DP_rule    #DP_rule: dispatching rule

    def initialize(self, OP):
        #reference
        self.env = self.fac.env
        self.OP = OP
        #attribute
        self.space = []
        #statistics
        #initial process

    def MST(self, MC):
        if MC.mode == None:
            job = np.random.choice(self.space)
            ME = [int(i) for i in self.fac.ME_table.loc[(self.fac.ME_table["jtype"] == job.jtype)&\
                                       (self.fac.ME_table["stage"] == MC.stage)]["mtype"].values[0].split(',')]
            while MC.mtype not in ME:
                job = np.random.choice(self.space)

        else:
            ST_list = []
            for j in self.space:
                ###########
                ME = [int(i) for i in self.fac.ME_table.loc[(self.fac.ME_table["jtype"] == j.jtype)&\
                                       (self.fac.ME_table["stage"] == MC.stage)]["mtype"].values[0].split(',')]
                if MC.mtype not in ME:
                    ST_list.append(1000)
                ###########
                _from = MC.mode.jtype
                to = j.jtype
                if _from == to:
                    ST_list.append(0)
                else:
                    ST = self.fac.ST_table.loc[(self.fac.ST_table["stage"] == self.stage)&\
                                               (self.fac.ST_table["from"]  == _from)&\
                                               (self.fac.ST_table["to"]    == to)]["setup_time"].values[0]
                    ST_list.append(ST)
            job_index = np.argmin(ST_list)
            job = self.space[job_index]
            # ###########
            # ME = [int(i) for i in self.fac.ME_table.loc[(self.fac.ME_table["jtype"] == job.jtype)&\
            #                            (self.fac.ME_table["stage"] == MC.stage)]["mtype"].values[0].split(',')]
            # while MC.mtype not in ME:
            #     ST_list[job_index] += 1000000
            #     print(ST_list)
            #     job_index = np.argmin(ST_list)
            #     job = self.space[job_index]
            # ########
        return job

    def dispatch(self, MC = None):
        if MC == None:
            for MC in self.OP:
                if len(self.space) <= 0:
                    break
                if MC.state == "idle":
                    job = self.MST(MC)
                    if job != None:
                        MC.job_accept(job)
                        self.space.remove(job)
        else:
            job = self.MST(MC)
            if job != None:
                MC.job_accept(job)
                self.space.remove(job)


    def job_arrive(self, job):
        self.space.append(job)
        ME = set()
        for MC in self.OP:
            for i in self.fac.ME_table.loc[(self.fac.ME_table["jtype"] == job.jtype)&\
                    (self.fac.ME_table["stage"] == MC.stage)]["mtype"].values[0].split(','):
                ME.add(int(i))

        idle_machine = sum([1 if MC.state == "idle" and MC.mtype in ME else 0 for MC in self.OP])
        if idle_machine > 0:
            self.dispatch()

    def get_job(self, MC):
        if len(self.space) > 0:
            ME = set()
            for job in self.space:
                for i in self.fac.ME_table.loc[(self.fac.ME_table["jtype"] == job.jtype)&\
                        (self.fac.ME_table["stage"] == MC.stage)]["mtype"].values[0].split(','):
                    ME.add(int(i))
            if MC.mtype in ME:
                self.dispatch(MC)

class Machine:
    def __init__(self, fac, stage, mtype, ID):
        self.fac = fac
        self.stage = stage
        self.mtype = mtype
        self.ID = ID

    def initialize(self, IP, OP):
        #reference
        self.env = self.fac.env
        self.IP = IP    #IP: input port, connect to a queue
        self.OP = OP    #OP: output port, connect to a queue or sink
        #attribute
        self.space = []     #the job on machine put in space
        self.state = "idle"    # state is one of "idle", "setup", "processing"
        self.mode = None    #mode record the last processing job for checking the setup time
        #statistics
        self.SST = 0    #SST: start setup time
        self.SPT = 0    #SPT: start process time
        #initial process

    def job_accept(self, job):    #all job sould pass this operation not going stright to setup or process state
        self.space.append(job)
        #check setup requirement
        c1 = self.mode == None
        if not c1:
            c2 = self.mode.jtype == job.jtype
        else:
            c2 = False
        if c1 or c2:    #don't need setup
                self.state = "processing"
                self.mode = self.space[0]
                self.SPT = self.env.now
                if self.fac.log:    #log message
                    print("{} : machine {}-{} start processing job {}".format(self.env.now, self.stage, self.ID, self.space[0].ID))
                #set next event
                self.process = self.env.process(self.process_job())
        else:    #need setup
            self.state = "setup"
            self.SST = self.env.now
            if self.fac.log:    #log message
                print("{} : machine {}-{} start setup {} -> {}".format(self.env.now, self.stage, self.ID, self.mode.jtype, self.space[0].jtype))
            #set next evnet
            self.process = self.env.process(self.setup())

    def setup(self):
        #setnext event
        stage = self.stage
        _from = self.mode.jtype
        to = self.space[0].jtype
        setup_time = self.fac.ST_table[(self.fac.ST_table["stage"] == stage)&\
                                       (self.fac.ST_table["from"]  == _from)&\
                                       (self.fac.ST_table["to"]    == to)]["setup_time"].values[0]
        yield self.env.timeout(setup_time)
        #complete setup
        self.mode = self.space[0]
        '''
        setup stastic
        '''
        #save process information for gantt
        ST = self.SST
        FT = self.env.now
        MC = self
        job = "setup {} -> {}".format(_from, to)
        self.fac.update_gantt(MC, ST, FT, job)

        #process job
        self.state = "processing"
        self.SPT = self.env.now
        if self.fac.log:    #log message
            print("{} : machine {}-{} start processing job {}".format(self.env.now, self.stage, self.ID, self.space[0].ID))
        #set next event
        self.process = self.env.process(self.process_job())
        '''
        event trace
        '''

    def process_job(self):
        #processing order for PT mins
        PT = self.fac.PT_table.loc[(self.fac.PT_table["jtype"] == self.space[0].jtype)&\
                                   (self.fac.PT_table["mtype"] == self.mtype)]["process_time"].values[0]
        yield self.env.timeout(PT)
        #complete process
        if self.fac.log:    #log message
            print("{} : machine {}-{} finish processing job {}".format(self.env.now, self.stage, self.ID, self.space[0].ID))
        '''
        setup stastic
        '''
        #save process information for gantt
        ST = self.SPT
        FT = self.env.now
        MC = self
        job = "j{} - {}".format(self.space[0].ID, self.space[0].jtype)
        self.fac.update_gantt(MC, ST, FT, job)

        #send job to next station
        self.OP.job_arrive(self.space[0])
        self.space = []
        #change state
        self.state = "idle"
        #get next job in queue
        self.IP.get_job(self)
        '''
        event trace
        '''

class Sink:
    def __init__(self, fac):
        self.fac = fac

    def initialize(self):
        #reference
        self.env = self.fac.env
        #attribute
        #statistics
        self.job_statistic = pd.DataFrame(columns = ["ID", "arrival_time", "complete_time", "flow_time"])
        #initial process

    def job_arrive(self, job):
        #update factory statistic
        self.fac.throughput += 1
        self.fac.update_WIP(-1)
        #update job statistic
        job.CT = self.env.now
        self.update_job_statistic(job)
        #ternimal condition
        job_num = self.fac.job_info.shape[0]
        if self.fac.throughput >= job_num:
            self.fac.terminal.succeed()

    def update_job_statistic(self, job):
        ID = job.ID
        AT = job.AT
        CT = job.CT
        flow_time = CT - AT
        self.job_statistic.loc[ID] = [ID, AT, CT, flow_time]

#factory
class Factory:
    def __init__(self, env, job_info, PT_table, ST_talbe, ME_table, DP_rule, log = True):
        self.env = env
        self.log = log
        self.job_info = job_info    #job_info: job information
        self.PT_table = PT_table    #PT: process time
        self.ST_table = ST_table    #ST: setup time
        self.ME_table = ME_table    #ME: machine eligibility
        self.DP_rule = DP_rule    #DP_rule: dispatching rule


    def initialize(self):
        #build
        self.source = Source(self, self.job_info)
        self.queues = {"Q1" : Queue(self, 1, self.DP_rule),}
                       # "Q2" : Queue(self, 2, self.DP_rule),
                       # "Q3" : Queue(self, 3, self.DP_rule)}
        self.machines = {"M1-1" : Machine(self, 1, 1, 1),
                         "M1-2" : Machine(self, 1, 2, 2),
                         "M1-3" : Machine(self, 1, 3, 3),}
                         # "M2-1" : Machine(self, 2, 1, 1),
                         # "M2-2" : Machine(self, 2, 2, 2),
                         # "M2-3" : Machine(self, 2, 2, 3),
                         # "M3-1" : Machine(self, 3, 1, 1),
                         # "M3-2" : Machine(self, 3, 1, 2),
                         # "M3-3" : Machine(self, 3, 2, 3),}
        self.sink = Sink(self)
        #initialize
        self.source.initialize(self.queues["Q1"])

        self.queues["Q1"].initialize([self.machines["M1-1"], self.machines["M1-2"], self.machines["M1-3"]])
        # self.queues["Q2"].initialize([self.machines["M2-1"], self.machines["M2-2"], self.machines["M2-3"]])
        # self.queues["Q3"].initialize([self.machines["M3-1"], self.machines["M3-2"], self.machines["M3-3"]])

        self.machines["M1-1"].initialize(self.queues["Q1"], self.sink)#queues["Q2"])
        self.machines["M1-2"].initialize(self.queues["Q1"], self.sink)#queues["Q2"])
        self.machines["M1-3"].initialize(self.queues["Q1"], self.sink)#queues["Q2"])
        # self.machines["M2-1"].initialize(self.queues["Q2"], self.queues["Q3"])
        # self.machines["M2-2"].initialize(self.queues["Q2"], self.queues["Q3"])
        # self.machines["M2-3"].initialize(self.queues["Q2"], self.queues["Q3"])
        # self.machines["M3-1"].initialize(self.queues["Q3"], self.sink)
        # self.machines["M3-2"].initialize(self.queues["Q3"], self.sink)
        # self.machines["M3-3"].initialize(self.queues["Q3"], self.sink)

        self.sink.initialize()
        #attribute
        #statistics
        self.throughput = 0

        self.WIP = 0
        self.WIP_area = 0
        self.WIP_change_time = 0
        '''
        self.event_ID = 0
        self.event_record = pd.DataFrame(columns = ["type",
                                                    "time",
                                                    "queue A", "MC A",
                                                    "queue B", "MC B",
                                                    "queue C", "MC C",
                                                    "WIP"])
        '''
        self.gantt_data = {"MC_name"         : [],
                           "start_process" : [],
                           "process_time"  : [],
                           "job"        : []}
        #terminal event
        self.terminal = self.env.event()
        '''
        self.print_state("Initializtion")
        '''
    def update_WIP(self, change):
        self.WIP_area += self.WIP*(self.env.now - self.WIP_change_time)
        self.WIP_change_time = self.env.now
        self.WIP += change
    '''
    def print_state(self, Etype):
        Q_A = "{}({})".format(len(self.queues["A"].space), [O.ID for O in self.queues["A"].space])
        Q_B = "{}({})".format(len(self.queues["B"].space), [O.ID for O in self.queues["B"].space])
        Q_C = "{}({})".format(len(self.queues["C"].space), [O.ID for O in self.queues["C"].space])

        MC_A = self.machines["A"].state
        if MC_A != "idle":
            MC_A = "{}".format(self.machines["A"].state.ID)
        MC_B = self.machines["B"].state
        if MC_B != "idle":
            MC_B = "{}".format(self.machines["B"].state.ID)
        MC_C = self.machines["C"].state
        if MC_C != "idle":
            MC_C = "{}".format(self.machines["C"].state.ID)

        self.event_record.loc[self.event_ID] = [Etype,
                                                self.env.now,
                                                Q_A, MC_A,
                                                Q_B, MC_B,
                                                Q_C, MC_C,
                                                self.WIP]
        self.event_ID += 1
    '''
    def update_gantt(self, MC, ST, FT, job):
        self.gantt_data['MC_name'].append("M-{}".format(MC.ID))
        self.gantt_data['start_process'].append(ST)
        self.gantt_data['process_time'].append(FT-ST)
        self.gantt_data['job'].append(job)

    def draw_gantt(self, save = None):
        #set color list
        #colors = list(mcolors.CSS4_COLORS.keys())
        colors = list(mcolors.TABLEAU_COLORS.keys())
        #np.random.shuffle(colors)
        #draw gantt bar
        y = self.gantt_data['MC_name']
        width = self.gantt_data['process_time']
        left = self.gantt_data['start_process']
        color = [colors[0] if len(j) == 12 else colors[int(j[1])] for j in self.gantt_data['job']]
        plt.barh(y = y, width = width, height = 0.5, left = left, color = color, align = 'center', alpha = 0.6)
        #add text
        for i in range(len(self.gantt_data['MC_name'])):
            text_x = self.gantt_data['start_process'][i] + self.gantt_data['process_time'][i]/2
            text_y = self.gantt_data['MC_name'][i]
            text = self.gantt_data['job'][i]
            if len(text) >=10:
                text = ""
            plt.text(text_x, text_y, text, verticalalignment='center', horizontalalignment='center')
        #figure setting
        plt.xlabel("time (mins)")
        plt.xticks(np.arange(0, self.env.now+1, 1))
        plt.ylabel("Machine")
        plt.title("hybrid flow shop - gantt")
        plt.grid(True)

        if save == None:
            plt.show()
        else:
            plt.savefig(save)
    '''
    def print_schduling_data(self):
        df = pd.DataFrame(columns = ["job ID", "step", "MC ID", "start time", "end time"])
        for i in range(len(self.gantt_data["MC_id"])):
            order_id = self.gantt_data["order_id"][i]
            step = self.gantt_data["order_progress"][i]
            MC_id = self.gantt_data["MC_id"][i]
            ST = self.gantt_data["start_process"][i]
            ET = self.gantt_data["start_process"][i] + self.gantt_data["process_time"][i]
            df.loc[i] = [order_id, step, MC_id, ST, ET]
        with pd.option_context('display.max_rows', None, 'display.max_columns', df.shape[1]):
            print(df)
    '''

if __name__ == '__main__':
    input_dir = os.getcwd() + '/input_data/'
    output_dir = os.getcwd() + '/results/'
    #environment
    env = simpy.Environment()
    #parameter
    job_info = pd.read_excel(input_dir + "job_information.xlsx")
    PT_table = pd.read_excel(input_dir + "PT_table.xlsx")
    ST_table = pd.read_excel(input_dir + "ST_table.xlsx")
    ME_table = pd.read_excel(input_dir + "ME_table.xlsx")
    DP_rule = "MST"
    #build job shop factory
    fac = Factory(env, job_info, PT_table, ST_table, ME_table, DP_rule, log = True)
    fac.initialize()
    #run simulation
    env.run(until = fac.terminal)

    #print result
    '''
    print("====================================================================")
    print("dispatching rule: {}".format(DP_rule))
    print(fac.event_record)
    print("====================================================================")
    with pd.option_context('display.max_rows', None, 'display.max_columns', fac.sink.job_statistic.shape[1]):
        print(fac.sink.job_statistic.sort_values(by = "ID"))
    '''
    print("====================================================================")
    print("makespan: {}".format(fac.env.now))
    print("Average folw time: {}".format(np.mean(fac.sink.job_statistic['flow_time'])))
    print("Aveage WIP: {}".format(fac.WIP_area/fac.env.now))
    print("====================================================================")
    '''
    print("schduling data")
    fac.print_schduling_data()
    print("====================================================================")
    '''
    fac.draw_gantt("gantt.png")
