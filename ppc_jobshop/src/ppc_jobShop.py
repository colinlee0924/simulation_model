#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
##-------- [PPC]  Jobshop Scheduling ---------
# * Author: Colin, Lee
# * Date: Apr 30th, 2020
# * Description:
#       Using the event-driven scheuling method
#       to solve the JSS prob. Here is a sample
#       code with the style of OOP. Feel free to
#       modify it as you like.
##--------------------------------------------
#

import os
import operator
import numpy as np
import pandas as pd
from gantt_plot import Gantt

#entity
class Order:
    def __init__(self, ID, AT, DD, routing, PT):
        self.ID   = ID
        self.AT    = AT         #AT: arrival time
        self.DD    = DD         #DD: due date

        self.PT       = PT      #PT: processing time
        self.routing  = routing
        self.progress = 0

#resource in factory
class Source:
    def __init__(self, order_info):
        self.order_info = order_info
        self.output = 0

    def arrival_event(self, fac):
        order_num = self.order_info.shape[0] #num of total orders

        #generate and release the order
        ID      = self.order_info.loc[self.output, "ID"]
        routing = self.order_info.loc[self.output, "routing"].split(',')
        PT      = [int(i) for i in self.order_info.loc[self.output, "process_time"].split(',')]
        DD      = self.order_info.loc[self.output, "due_date"]
        AT      = T_NOW
        order   = Order(ID, AT, DD, routing, PT)
        if LOG == True:
            print("{} : order {} release.".format(T_NOW, order.ID))

        self.output += 1

        #update the future event list - next order arrival event
        if self.output < order_num:
            fac.event_lst.loc["Arrival"]["time"] = self.order_info.loc[self.output, "arrival_time"]
        else:
            fac.event_lst.loc['Arrival']['time'] = M

        #send order to correlated station
        target = order.routing[order.progress]
        machine = fac.machines[target]
        machine.buffer.append(order)

        #update the future event list - dispatch machines to process the jobs
        if machine.state == 'idle':
            fac.event_lst.loc["dispatching"]['time'] = T_NOW

class Machine:
    def __init__(self, ID, DP_rule):
        self.ID     = ID
        self.state  = 'idle'
        self.buffer = []
        self.wspace = [] #wspace: working space
        self.DP_rule = DP_rule
        self.first_time = True

    def start_processing(self, fac):
        #check state
        if self.state == 'idle':
            #check the status of buffer
            if len(self.buffer) > 0:
                #get a new order from buffer by DP_rule
                if self.first_time == False:
                    self.buffer = sorted(self.buffer, key=operator.attrgetter('DD'))
                    if self.DP_rule == "FIFO":
                        order = self.buffer[0]
                    elif self.DP_rule == "EDD":
                        idx = np.argmin([j.DD for j in self.buffer])
                        order = self.buffer[idx]
                    elif self.DP_rule == "SPT":
                        idx = np.argmin([j.PT[j.progress] for j in self.buffer])
                        order = self.buffer[idx]

                #Guys, here me need to implement release rule
                else:
                    idx = np.argmin([j.DD for j in self.buffer])
                    order = self.buffer[idx]
                    self.first_time = False

                #remove order from buffer
                self.buffer.remove(order)

                #start processing the order
                self.wspace.append(order)
                self.state = 'busy'
                processing_time = order.PT[order.progress]

                #[Gantt plot preparing] udate the start/finish processing time of machine
                fac.gantt_plot.update_gantt(self.ID, T_NOW, processing_time, order.ID)
                if LOG == True:
                    print("{} : machine {} start processing order {} - {} progress".format(T_NOW, self.ID, order.ID, order.progress))

                #update the future event list - job complete event
                fac.event_lst.loc["{}_complete".format(self.ID)]['time'] = T_NOW + processing_time
                order.progress += 1

    def end_process_event(self, fac):
        order = self.wspace[0]
        if LOG == True:
            print("{} : machine {} complete order {} - {} progress".format(T_NOW, self.ID, order.ID, order.progress))
        self.wspace.remove(order)
        self.state = 'idle'

        #send the processed order to next place
        if order.progress >= len(order.routing):
            #update factory statistic
            fac.throughput += 1
            #update order statistic
            fac.update_order_statistic(order)
        else:
            #send the order to next station
            target = order.routing[order.progress]
            next_machine = fac.machines[target]
            next_machine.buffer.append(order)

        #update the future event list - wait for the dispatching to get a new job
        fac.event_lst.loc["dispatching"]['time'] = T_NOW
        fac.event_lst.loc["{}_complete".format(self.ID)]["time"] = M

class Factory:
    def __init__(self, order_info):#, DP_rule):
        self.order_info = order_info
        # self.DP_rule    = DP_rule

        #[Plug in] tool of gantt plotting
        self.gantt_plot = Gantt()

        #statistics
        self.throughput = 0
        self.order_statistic = pd.DataFrame(columns = ["ID", "release_time", "complete_time", "due_date", "flow_time", "tardiness", "lateness"])

    def build(self):
        self.source   = Source(self.order_info)
        self.machines = {'A': Machine('A', 'EDD'),
                         'B': Machine('B', 'SPT'),
                         'C': Machine('C', 'SPT'),
                         'D': Machine('D', 'EDD')}

    def initialize(self, order_info):
        self.event_lst = pd.DataFrame(columns=["event_type", "time"])
        self.event_lst.loc[0] = ["Arrival", order_info.loc[0, "arrival_time"]]
        self.event_lst.loc[1] = ["A_complete", M]
        self.event_lst.loc[2] = ["B_complete", M]
        self.event_lst.loc[3] = ["C_complete", M]
        self.event_lst.loc[4] = ["D_complete", M]
        self.event_lst.loc[5] = ["dispatching", M]
        self.event_lst = self.event_lst.set_index('event_type')

    def next_event(self, stop_time):
        global T_NOW, T_LAST
        T_NOW, T_LAST = M, M
        self.initialize(self.order_info)
        T_NOW      = self.event_lst.min()["time"]
        event_type = self.event_lst['time'].astype(float).idxmin()

        while T_NOW < stop_time:
            # print()
            # print('T-NOW: ', T_NOW)
            # print(self.event_lst)
            # print()
            self.event(event_type)
            T_LAST     = T_NOW
            T_NOW      = self.event_lst.min()["time"]
            event_type = self.event_lst['time'].astype(float).idxmin()
        self.t_now = T_LAST
        T_NOW = stop_time

    def event(self, event_type):
        #Arrival event
        if event_type == 'Arrival':
            self.source.arrival_event(self)

        #Complete event
        elif event_type == 'A_complete':
            self.machines['A'].end_process_event(self)
        elif event_type == 'B_complete':
            self.machines['B'].end_process_event(self)
        elif event_type == 'C_complete':
            self.machines['C'].end_process_event(self)
        elif event_type == 'D_complete':
            self.machines['D'].end_process_event(self)

        #Dispatch event
        else:
            for mc in self.machines.values():
                mc.start_processing(self)
            self.event_lst.loc["dispatching"]['time'] = M

    def update_order_statistic(self, order):
        ID = order.ID
        AT = order.AT
        DD = order.DD
        complete_time = T_NOW
        flow_time = complete_time - AT
        lateness = complete_time - DD
        tardiness = max(0, lateness)
        self.order_statistic.loc[ID] = [ID, AT, complete_time, DD, flow_time, tardiness, lateness]

M = 100000000000
LOG = True
stop_time = 500

if __name__ == '__main__':
    #read the input data sheet
    data_dir = os.getcwd() + "/data/"
    order_info = pd.read_excel(data_dir + "order_information.xlsx")

    #data preprocessing
    order_info = order_info.sort_values(['arrival_time']).reset_index(drop=True)

    # DP_rule = 'SPT' #'EDD'

    #build the factory
    fac = Factory(order_info)#, DP_rule)
    fac.build()

    #start the simulation
    fac.next_event(stop_time)

    #output result
    print(fac.order_statistic)
    fac.gantt_plot.draw_gantt(fac.t_now)
