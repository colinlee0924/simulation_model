#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
##------- [Tool] Gantt chart plotting tool --------
# * Author: CIMLab
# * Date: Apr 30th, 2020
# * Description:
#       This program is a tool to help you guys
#       to draw the gantt chart of your scheduling
#       result. Feel free to modify it as u like.
##-------------------------------------------------
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Gantt:
    def __init__(self):
        self.gantt_data = {"MC": [],
                           "Order" : [],
                           "Start time" : [],
                           "Process time" : []}

    def update_gantt(self, MC, ST, PT, job):
        self.gantt_data['MC'].append("M{}".format(MC))
        self.gantt_data['Order'].append(job)
        self.gantt_data['Start time'].append(ST)
        self.gantt_data['Process time'].append(PT)

    def draw_gantt(self, T_NOW, save = None):
        #set color list
        plt.figure() # plt.figure(figsize=(12, 6))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        #draw gantt bar
        y = self.gantt_data['MC']
        width = self.gantt_data['Process time']
        left = self.gantt_data['Start time']
        color = []
        for j in self.gantt_data['Order']:
            color.append(colors[int(j)-1])

        plt.barh(y = y, width = width, height = 0.5, color=color,left = left, align = 'center',alpha = 0.6)
        #add text
        for i in range(len(self.gantt_data['MC'])):
            text_x = self.gantt_data['Start time'][i] + self.gantt_data['Process time'][i]/2
            text_y = self.gantt_data['MC'][i]
            text = self.gantt_data['Order'][i]
            plt.text(text_x, text_y, 'J'+str(text), verticalalignment='center', horizontalalignment='center', fontsize=8)
        #figure setting
        plt.xlabel("time (hrs)")
        plt.ylabel("Machine")
        plt.title("Gantt Chart")
        plt.xticks(np.arange(0, T_NOW+1, 1))
        # plt.grid(True)

        if save == None:
            plt.show()
        else:
            plt.savefig(save)
