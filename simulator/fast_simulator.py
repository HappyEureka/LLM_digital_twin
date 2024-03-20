# retrive simulated data
from .simulator import Simulator
import os
import pickle
import json
from .utils.utils import *
from .sim_time import SimTime

class FastSimulator():
    def __init__(
            self,
            town_population=6000,
            setup_fname="simulator/setup.json",
            interval=60, # minutes
            open_hour=10,
            open_minute=0,
            close_hour=20,
            close_minute=0,
            suffix = '',
        ):
        self.interval = interval
        self.sim_time = SimTime(self.interval,
                                  open_hour, 
                                  open_minute,
                                  close_hour,
                                  close_minute)
        self.sim = Simulator(town_population, setup_fname, interval, open_hour, open_minute, close_hour, close_minute)
        
        fname = f'simulator/saved_simulations/{town_population}_{interval}_{open_hour}_{open_minute}_{close_hour}_{close_minute}'
        if suffix != '':
            fname += f'_{suffix}.pkl'
        else:
            fname += '.pkl'

        if os.path.isfile(fname):
            self.rsts = pickle.load(open(fname, 'rb'))
        else:
            self.rsts = []
            operation_time = (close_hour - open_hour) * 60 + (close_minute - open_minute) # close minute must be larger or equal to open minute
            for _ in range(operation_time // self.sim.interval):
                self.rsts.append(self.sim.step())
                print(self.sim.sim_time.get_time_string())
            pickle.dump(self.rsts, open(fname, 'wb'))
        # track which rsts to return
        self.ind = 0
        # load setup
        self.setup_fname = setup_fname
        self.load_setup()
        # init temperatures for each store
        self.curr_temps = {k: 26 for k in self.stores.keys()}
        

    def load_setup(self):
        # load setup file
        with open(self.setup_fname) as f:
            setup = json.load(f)
        self.mall = setup['Happy Mall']
        self.groups = setup['Groups']
        # init stores
        self.stores = {}
        for key, item in self.mall.items():
            self.stores = {**self.stores, **item}

    def step(self):
        # extract results
        self.sim_time.step()
        rst_df = self.rsts[self.ind]
        rst_df['store_temp'] = rst_df['Store'].apply(lambda x: self.curr_temps[x])
        rst_df['vote'] = rst_df.apply(lambda x: get_vote(x, self.groups), axis=1)
        self.ind += 1
        return rst_df