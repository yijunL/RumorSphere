from __future__ import annotations

from typing import TYPE_CHECKING, List
import random
import math
import numpy as np
from . import order_registry as OrderRegistry
from .base import BaseOrder
from tqdm import tqdm

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@OrderRegistry.register("twitter")
class TwitterOrder(BaseOrder):
    # """
    # The agents speak concurrently in a random order
    # """

    # def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
    #     res = list(range(len(environment.agents)))
    #     random.shuffle(res)
    #     return res

    
    """
    In each round, we determine whether agents are in an information bubble based on 
    information similarity and diversity to classify them as core or regular agents.
    """
    def get_next_agent_idx(self, environment: BaseEnvironment, core_num=10, k=1, min_fan=50) -> List[int]:
        order_type = 0 # 0 RumorAgent, 1 Hisim, 2 Random
        
        if(core_num==0):
            print("core_num = 0, no core agent.")
            return []
        
        atts = environment.abm_model.get_attitudes()
        neighbor_atts = environment.abm_model.get_neighbor_attitudes()       


        if(len(atts)==100):
            min_fan = 15
        elif(len(atts)==100):
            min_fan = 100
            

        
        # core ratio
        if(core_num<1):
            core_num = int(core_num*len(atts))

        if(order_type == 1):
            return environment.abm_model.get_leader_id(core_num)

        if(order_type == 2):
            return random.sample(range(len(atts)), core_num)
        
        difs = []
        divers = []
        fusion_factors = []
        for i in tqdm(range(len(atts)),desc="get fusion factors..."):
            if(len(neighbor_atts[i])<min_fan):
                fusion_factors.append(0)
                continue
            
            att = atts[i]
            neighbor_att = np.array(neighbor_atts[i])
            att = np.array([att] * len(neighbor_att))
            dif = (abs(att - neighbor_att))/2
            dif = 1-np.mean(dif)
            difs.append(dif)

            diver = np.std(neighbor_att)
            divers.append(diver)

            fusion_factor = self.calculate_confusion(dif, diver, k=k)
            fusion_factors.append(fusion_factor) 

        # print("get top_k_indices...")
        core_agent_id = self.top_k_indices(fusion_factors, core_num)
        # print("get top_k_indices completed...")

        # print("\nfusion_factors:",fusion_factors)
        # print("Atts:",atts)
        # print("core_agent_id:",core_agent_id)
        
        return core_agent_id

    def calculate_confusion(self, a, b, k=1):
        """
        Calculate the confusion score based on average divergence a and variance b,
        using an exponential function and normalizing the result to the range [0, 1].
    
        Parameters:
        a (float): Average divergence, range [0, 1]
        b (float): Variance, range [0, 1]
    
        Returns:
        float: Confusion score, range [0, 1]
        """
        # Numerator: e^kb - e^ka 
        numerator = (math.exp(b))-(math.exp(k*a))  - 2
        
        # Denominator: e^2k-2 to normalize the result to [0, 1]
        denominator = 2*math.exp(k) - 2
        
        # Return the normalized confusion score
        return numerator / denominator

    def top_k_indices(self, lst, core_num):
        sorted_with_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        return [index for index, value in sorted_with_indices[:core_num]]