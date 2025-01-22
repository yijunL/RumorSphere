import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
from agentverse.abm_model import abm_registry
from agentverse.logging import get_logger
import json
from tqdm import tqdm

logger = get_logger()


class DeffuantAgent(mesa.Agent):
    """
    Deffuant's model is a BC model where N_J=1
    It assumes that if the message 𝑚𝑗,𝑡 is close enough to the agent 𝑖’s attitude 𝑎𝑖,𝑡 , 
    the message has an assimilation force on the agent’s attitude. 
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, bc_bound = 0.2, following_info=[]):
        """
        bc_bound: the confidence bound
        """
        
        super().__init__(unique_id, model)
        
        self.name = name
        # initial attitude
        self.att =  [init_att]
        # strength of the social influence
        self.alpha = alpha
        self.bc_bound = bc_bound
        self.following_info = following_info

        
    def step(self):
        """
        Selection Function: one random agent j in the system within the confidence bound
        Message Function: m_jt = a_jt
        Assimilation Force: asm(a_it, m_jt) = (m_jt-a_it)
        Similarity Bias: sim(a_it, m_jt) = 1 if diff< bc_bound, else 0
        """
        # attitude update
        att = self.att[-1]
        att_update = 0
        candidate_agents = []        

        weighted_update = []
        for agent in self.model.neighbor_agents[self.name]:
            # exclude the agent itself
            # if agent == self or agent.name not in self.following_info:continue
            if abs(att-agent.att[-1])<self.bc_bound:
                candidate_agents.append(agent)

            if(agent.unique_id in self.model.core_agent_id):
                weighted_update.append(agent.att[-1]-att)
                
        # Aggregate the opinions of all neighbors
        if len(candidate_agents):
            att_update=0
            for target_agent in candidate_agents:
                sim = 1
                att_update += target_agent.att[-1]-att
            att_update /= len(candidate_agents)
        else:
            sim = 0 
            att_update = 0

        key_weight = 0.5
        if(len(weighted_update)):
            att_update = (1-key_weight)*att_update+key_weight*(sum(weighted_update)/len(weighted_update))

        random_update = 0
        if(len(self.model.core_agent_id)==0):
            random_update = self.model.schedule.agents[random.randint(0, self.model.num_agents-1)].att[-1]-att 
        att = att + self.alpha * att_update + self.alpha * random_update
        self.att.append(att)
#         print(self.name, att)



@abm_registry.register("deffuant")
class Deffuant(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order = 'concurrent', alpha=0.1, bc_bound=0.1,llm_agents_atts=[],following_info=None):
        super().__init__()
        self.num_agents = len(agent_config_lst)
        self.llm_agents_atts = llm_agents_atts
        self.name2idx = {}
        self.following_info = {}
        self.neighbor_atts = []
        self.alpha = alpha
        self.bc_bound = bc_bound
        self.neighbor_agents = {}
        self.core_agent_id = []
        self.leader_id = []
        # Create scheduler and assign it to the model
        
        if order =='concurrent':
            self.schedule = mesa.time.BaseScheduler(self)
        elif order =='simultaneous':
            self.schedule = mesa.time.SimultaneousActivation(self)
        elif order =='random':
            self.schedule = mesa.time.RandomActivation(self)
        else:
            raise NotImplementedError
        self.order =order
        
        # initialize following info of abm agent
        if following_info is None:
            for i in range(self.num_agents):
                self.following_info[agent_config_lst[i]['name']]=[]
        else:
            if isinstance(following_info, str):
                groups = json.load(open(following_info, 'r'))
            else:
                groups = following_info
            
            for i in range(self.num_agents):
                agent_name = agent_config_lst[i]['name']
                if agent_name in groups:
                    self.following_info[agent_name]=groups[agent_name]
                else:
                    # following no one
                    self.following_info[agent_name]=[]
        
        # Create agents
        self.name2idx = {}
        for i in range(self.num_agents):
            a = DeffuantAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'], alpha=alpha, bc_bound=bc_bound, following_info=self.following_info[agent_config_lst[i]['name']])
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
            
        for agent in tqdm(self.schedule.agents,desc="initialize neighbor agent..."):
            self.neighbor_agents[agent.name] = []
            for agent2 in self.schedule.agents:
                if agent == agent2 or agent2.name not in agent.following_info:
                    continue
                else:
                    self.neighbor_agents[agent.name].append(agent2)

        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)
        self.leader_id = self.get_leader_id()

    def step(self, core_agent_id):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.core_agent_id = core_agent_id
        self.schedule.step()
        for agent in self.llm_agents_atts:
            self.update_mirror(agent, self.llm_agents_atts[agent][self._steps-1])
        
    def get_attitudes(self):
        atts = [a.att[-1] for a in self.agents]
        return atts

    def get_neighbor_attitudes(self):
        neighbor_atts = []
        last_attitudes = [a.att[-1] for a in self.agents]
        neighbor_idxs = [[self.name2idx[nei] for nei in a.following_info] for a in self.agents]

        for following_info in neighbor_idxs:
            temp_atts = []
            for idx in following_info:
                temp_atts.append(last_attitudes[idx])
            neighbor_atts.append(temp_atts)
        return neighbor_atts

    def get_leader_id(self, num=5):
        neighbor_atts = self.get_neighbor_attitudes()
        lengths_with_indices = [(i, len(sublist)) for i, sublist in enumerate(neighbor_atts)]
        
        # 根据长度降序排序并取前10个
        top_c_indices = [index for index, _ in sorted(lengths_with_indices, key=lambda x: x[1], reverse=True)[:num]]
        return top_c_indices
        
    def get_measures(self, target_attitudes,ne_att=0):
        """
        target_attitudes: empirical data
        output measures: bias, diversity
        - bias: the deviation of the mean attitude from the neutral attitude
        - diversity: the standard deviation of attitudes
        """
        simu_atts = self.get_attitudes()
        
        # empirical
        bias = np.mean(target_attitudes)-ne_att
        diversity = np.var(target_attitudes)
        
        # simu
        simu_bias = np.mean(simu_atts)-ne_att
        simu_diversity = np.var(simu_atts)
        
        delta_bias = abs(simu_bias-bias)
        delta_diversity = abs(simu_diversity-diversity)

        return {'bias':bias,
               'diversity':diversity,
               'simu_bias':simu_bias,
               'simu_diversity':simu_diversity,
               'delta_bias':delta_bias,
               'delta_diversity':delta_diversity}

    def update_mirror(self, name, att):
        idx = self.name2idx[name]
        self.agents[idx].att[-1] = att