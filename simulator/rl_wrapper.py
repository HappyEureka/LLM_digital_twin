from simulator.constants import *
from datetime import timedelta
import numpy as np
mall_layout = {
    "Happy’s Department Store": (100, 50, 200, 200),
    "Weekend Outdoors": (900, 50, 200, 200),
    "Elegant Styles": (100, 550, 150, 150),
    "Urban Gear": (260, 550, 150, 150),  # Updated position
    "Tech Haven": (400, 100, 100, 100),
    "Gamer’s Paradise": (700, 100, 100, 100),
    "Savor’s Grille": (400, 250, 150, 150),
    "Café Fresco": (600, 250, 150, 150),
    "Quick Bites Food Court": (800, 350, 200, 100),
    "Serene Spa and Salon": (250, 400, 150, 150),
    "Vital Pharmacy": (600, 450, 150, 150),
    "Book Nook": (100, 700, 100, 100),  # Updated position
    "Happy Florist": (500, 600, 100, 100),
    "Gifts & More": (500, 600, 100, 100),
    "Cineplex": (650, 600, 200, 150),  # Updated size
    "Child’s Play": (900, 650, 150, 150),  # Updated position and size
    "Powerhouse Gym": (50, 250, 150, 150),
    "Information Desk": (1000, 350, 100, 100),
    "Restrooms and Baby Care Rooms": (1000, 500, 100, 100),
    "Weekend Live"  : (900, 50, 200, 100),
    "Holiday Celebrations": (900, 50, 200, 100),
}

VOTE_WEIGHTS = {'down': -1, 'up': -1, 'no_change': 2}
REWARD_OFFSET = 820

store_areas = {}
for key, val in mall_layout.items():
    store_areas[key] = val[2] * val[3]

class AggregatedSim():
    def __init__(self, sim, alpha=ALPHA_, beta=BETA_):
        self.sim = sim
        self.stores = self.sim.stores
        self.alpha = alpha
        self.beta = beta
        self.name = "AggregatedSim"
    
    def __repr__(self) -> str:
        model_info = f"{self.name}(alpha={self.alpha}, beta={self.beta}, power_offset={REWARD_OFFSET})"
        print(model_info)
        return model_info
    
    def step(self):
        self.rst_df = self.sim.step()

    def get_score(self):
        df = self.rst_df.copy()
        votes = df['vote'].values
        votes = np.concatenate(votes).reshape(-1, 3)
        sums = votes.sum(axis=1)
        with np.errstate(invalid='ignore'):
            normalized_votes = votes / sums.reshape(-1, 1)
            normalized_votes[sums == 0] = 0
        df['store_area'] = df['Store'].map(store_areas)
        m = df['store_area'].values * ROOM_HEIGHT * DENSITY
        q = -m * c_p * abs(df['store_temp'].values - AMBIENT_TEMP)
        power = q / (self.sim.interval / 60) / 1000
        weights = np.array([VOTE_WEIGHTS['down'], VOTE_WEIGHTS['up'], VOTE_WEIGHTS['no_change']])
        
        comfort_score = ALPHA_ * np.dot(normalized_votes, weights)
        energy_score = BETA_ * (power + REWARD_OFFSET)
        score = comfort_score + energy_score
        return (sum(score), sum(comfort_score), sum(energy_score))
        

# assume there is one centralized aircon
class CentralizedAggregatedSim(AggregatedSim):
    def __init__(self, sim, alpha=ALPHA_, beta=BETA_):
        super().__init__(sim, alpha, beta)
        self.name = "CentralizedAggregatedSim"

    def get_states(self):
        # people in each store + mall temperature + time
        return self.rst_df.iloc[:, 1: -2].sum(axis=0).tolist() + \
                [self.rst_df.store_temp[0]] + \
                [self.sim.sim_time.get_time_float()] + \
                self.rst_df.vote.sum().tolist()
    
    def get_reward(self):
        votes = self.rst_df.vote.sum()
        normalized_votes = votes / sum(votes)
        down, up, no_change = normalized_votes
        
        VOLUME = ROOM_AREA * ROOM_HEIGHT
        m = VOLUME * DENSITY
        q = - m * c_p * abs(self.rst_df.store_temp.mean() - AMBIENT_TEMP)
        power = q / (self.sim.interval / 60) / 1000 # W
        #if np.random.rand() < 0.01: # print 1% of the time
        #    print(down, up, no_change)
        #    print("####", self.alpha * (VOTE_WEIGHTS['down'] * down +
        #                        VOTE_WEIGHTS['up'] * up +
        #                        VOTE_WEIGHTS['no_change'] * no_change), self.beta * power)
        return self.alpha * (VOTE_WEIGHTS['down'] * down +
                        VOTE_WEIGHTS['up'] * up +
                        VOTE_WEIGHTS['no_change'] * no_change) + (self.beta * (power + REWARD_OFFSET))

    def apply_action(self, action):
        # for all stores, same action (temperature)
        self.sim.curr_temps = {k: action for k in self.sim.curr_temps}

class DecentralizedAggregatedSim(AggregatedSim):
    def __init__(self, sim, alpha=ALPHA_, beta=BETA_):
        super().__init__(sim, alpha, beta)
        self.name = "DistributedAggregatedSim"

    def get_states(self):
        # people in each store + mall temperature + time
        df = self.rst_df.iloc[:, 1: -1]
        df['curr_time'] = self.sim.sim_time.get_time_float()
        votes = np.concatenate(self.rst_df.vote.values).reshape(-1, 3)
        state_matrix = np.concatenate([df.values, votes], axis=1) 
        return (self.rst_df.Store.values, state_matrix)

    def get_rewards(self):
        df = self.rst_df.copy()
        votes = df['vote'].values
        votes = np.concatenate(votes).reshape(-1, 3)
        sums = votes.sum(axis=1)
        with np.errstate(invalid='ignore'):
            normalized_votes = votes / sums.reshape(-1, 1)
            normalized_votes[sums == 0] = 0
        df['store_area'] = df['Store'].map(store_areas)
        m = df['store_area'].values * ROOM_HEIGHT * DENSITY
        q = -m * c_p * abs(df['store_temp'].values - AMBIENT_TEMP)
        power = q / (self.sim.interval / 60) / 1000
        weights  = np.array([VOTE_WEIGHTS['down'], VOTE_WEIGHTS['up'], VOTE_WEIGHTS['no_change']])
        return self.alpha * np.dot(normalized_votes, weights) + (self.beta * (power + REWARD_OFFSET))
    
    def apply_action(self, actions):
        # different actions for different stores
        self.sim.curr_temps = {k: action for k, action in zip(self.sim.curr_temps, actions)}