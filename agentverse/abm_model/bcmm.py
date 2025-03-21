import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
from agentverse.abm_model import abm_registry
from agentverse.logging import get_logger

logger = get_logger()


class BCMultiAgent(mesa.Agent):
    """
    Deffuant's model is a BC model where N_J=1
    It assumes that if the message 𝑚𝑗,𝑡 is close enough to the agent 𝑖’s attitude 𝑎𝑖,𝑡 , 
    the message has an assimilation force on the agent’s attitude. 
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, bc_bound = 0.2):
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
        

    def step(self):
        """
        Attitude Update Function: delta = N_eps_i/(N_eps_i+1)*(1/N_eps_i)*sum_\j in J sim(a_it, a_jt)*(a_it-a_jt)
            N_eps_i: the number of agents within the bc bound
        Assimilation Force: asm(a_it, m_jt)=m_jt-a_it
        Similarity Bias: sim(a_it,m_jt)=1 if abs(a_jt-a_it)<eps_i, eps_i is the bc bound of agent i
        Other Assumptions:
        Selection Function: all the agents within the bc bound of agent i
        Message Function: m_jt = a_jt
        """
        # attitude update
        att = self.att[-1]
        att_update = 0
        candidate_agents = []
        for agent in self.model.schedule.agents:
            # exclude the agent itself
            if agent == self:continue
            if abs(att-agent.att[-1])<self.bc_bound:
                candidate_agents.append(agent)
        N = len(candidate_agents)
        if N:
            for agent in candidate_agents:
                a_jt = agent.att[-1]
                sim = 1 if abs(a_jt-att)<self.bc_bound else 0
                att_update+=sim*(a_jt-att)
            att_update = N/(N+1)*1/N*att_update
            
        random_update = 0
        if(len(self.model.core_agent_id)==0):
            random_update = self.model.schedule.agents[random.randint(0, self.model.num_agents-1)].att[-1]-att
            
        att = att + att_update + self.alpha * random_update
        self.att.append(att)
#         print(self.name, att)


@abm_registry.register("bcmm")
class BCMultiModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order = 'concurrent', alpha=0.1, bc_bound=0.1, llm_agents_atts=[], following_info=None):
        super().__init__()
        self.num_agents = len(agent_config_lst)
        self.llm_agents_atts = llm_agents_atts
        self.name2idx = {}
        self.alpha = alpha
        self.bc_bound = bc_bound
        self.core_agent_id = []
        # Create scheduler and assign it to the model
        if order =='concurrent':
            self.schedule = mesa.time.BaseScheduler(self)
        elif order =='simultaneous':
            self.schedule = mesa.time.SimultaneousActivation(self)
        elif order =='random':
            self.schedule = mesa.time.RandomActivation(self)
        else:
            raise NotImplementedError

        # Create agents
        for i in range(self.num_agents):
            a = BCMultiAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'],
                         alpha=alpha, bc_bound = bc_bound)
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)

    def step(self, core_agent_id=[]):
        """Advance the model by one step."""
        self.core_agent_id = core_agent_id
        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
#         print(self._steps)
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