import pandas as pd
import json
from .llm import LLM
from .sim_time import SimTime
from .utils.utils import *

class Simulator():
    def __init__(
            self,
            town_population=6000,
            setup_fname="simulator/setup.json",
            interval=60, # minutes
            open_hour=10,
            open_minute=0,
            close_hour=20,
            close_minute=0,
        ):
        self.town_population = town_population
        self.setup_fname = setup_fname

        self.load_setup()
        # init num_stores and num_groups 
        self.num_stores = len(self.stores)
        self.num_groups = len(self.groups)
        # init LLM
        self.llm = LLM(self.num_groups, self.num_stores)
        # init time tracker
        self.interval = interval
        self.sim_time = SimTime(self.interval,
                                  open_hour, 
                                  open_minute,
                                  close_hour,
                                  close_minute)
        # init descriptions
        self.init_descriptions()
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

    def init_descriptions(self):
        # generate store descriptions
        self.stores_descriptions = ""
        for key, value in self.stores.items():
            self.stores_descriptions += f"{key}: {value[:-1]}; "
        # group descriptions
        self.groups_descriptions = ""
        for key, value in self.groups.items():
            self.groups_descriptions += f"{key}: {value['Description'][:-1]}; "

    def generate_prompts(self):
        population_prompt = f"Located in a town center with a population of {self.town_population}. The Happy Mall opens at {self.sim_time.get_open_time()} and closes at {self.sim_time.get_close_time()}. Currently, it's {self.sim_time.get_time_string()}. Descriptions of different groups in the mall are as follows. {self.groups_descriptions}. Given the time of day, list the number of people in each group using the following format. [group name]: [number]; [reason]."

        distribution_prompts = []
        for group in self.groups:
            distribution_prompt = f"Currently, it's {self.sim_time.get_time_string()}. The descriptions of the stores in the Happy Mall are as follows. {self.stores_descriptions}. You belong to {group}, described as {self.groups[group]['Description']}, with a thermal preference of {self.groups[group]['Thermal Preference']}. What is your group's distribution in the mall? List the percentile for each store using the following format. [store]: [percentile]; [reason]."
            distribution_prompts.append(distribution_prompt)
        #temperature_prompt = f"It is currently {self.sim_time.get_time_string()}. Would you like #to raise the temperature or lower the temperature?"
        temperature_prompt = "" # not used
        return population_prompt, distribution_prompts, temperature_prompt
    
    def extract_pop_response(self, pop_response):
        pop_rsts = {}
        for text in pop_response.split('\n'):
            if text != '\n' and text != '':
                most_similar, n, similarity_score = get_most_similar_name_and_corresponding_result(text, self.groups.keys())
                if similarity_score > 0:
                    pop_rsts[most_similar] = extract_from_string(n)
        return pop_rsts
    
    def extract_dist_responses(self, dist_responses):
        dist_rsts = {}
        for group, dist_res in zip(self.groups, dist_responses):
            rsts = {k:0 for k in self.stores.keys()}
            for text in dist_res.split('\n'):
                if text != '\n' and text != '':
                    most_similar, n, similarity_score = get_most_similar_name_and_corresponding_result(text, self.stores.keys())
                    if similarity_score > 0:
                        n = extract_from_string(n)
                        rsts[most_similar] = n
            dist_rsts[group] = rsts
        return dist_rsts
    
    def extract_results(self, pop_response, dist_responses):
        # extract results
        pop_rsts = self.extract_pop_response(pop_response)
        dist_rsts = self.extract_dist_responses(dist_responses)
        # normalize percentage to sum up to 1, and get the number for each group at each store
        for group in self.groups:
            total = sum(dist_rsts[group].values())
            for store in self.stores:
                normalized_value = dist_rsts[group][store] / total
                dist_rsts[group][store] = round(normalized_value * pop_rsts[group])

        df = pd.DataFrame.from_dict(dist_rsts)
        df = df.reset_index().rename(columns={'index':'Store'})
        df['store_temp'] = df['Store'].apply(lambda x: self.curr_temps[x])
        df['vote'] = df.apply(lambda x: get_vote(x, self.groups), axis=1)
        return df
        
    def step(self):
        self.sim_time.step()
        # generate prompts
        population_prompt, distribution_prompts, _ = self.generate_prompts()
        # generate responses
        responses = self.llm.generate_responses_parallel([population_prompt] + distribution_prompts)
        pop_response = responses[0]
        dist_responses = responses[1:]
        # extract results
        rst_df = self.extract_results(pop_response, dist_responses)
        return rst_df