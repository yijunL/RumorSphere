from typing import NamedTuple, Union
from enum import Enum

import abc
import re
import openai
from tenacity import retry, stop_after_attempt, wait_exponential


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class AgentCriticism(NamedTuple):
    """Agent's criticism."""

    is_agree: bool
    criticism: str
    sender_agent: object = None


class AGENT_TYPES(Enum):
    ROLE_ASSIGNMENT = 0
    SOLVER = 1
    CRITIC = 2
    EXECUTION = 3
    EVALUATION = 4
    MANAGER = 5

    @staticmethod
    def from_string(agent_type: str):
        str_to_enum_dict = {
            "role_assigner": AGENT_TYPES.ROLE_ASSIGNMENT,
            "solver": AGENT_TYPES.SOLVER,
            "critic": AGENT_TYPES.CRITIC,
            "executor": AGENT_TYPES.EXECUTION,
            "evaluator": AGENT_TYPES.EVALUATION,
            "manager": AGENT_TYPES.MANAGER,
        }
        assert (
            agent_type in str_to_enum_dict
        ), f"Unknown agent type: {agent_type}. Check your config file."
        return str_to_enum_dict.get(agent_type.lower())


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def judge_response(response):
    try:
        response = float(response)
    except:
        response = 0
    return response

def label_stance(text, target):
    # pro_patterns={
    #     'Metoo Movement':['#Withyou',],
    #     'Black Lives Matter Movement':['#BlackLivesMatter', '#GeorgeFloyd', '#PoliceBrutality', '#BLM',],
    #     'the protection of Abortion Rights':['#roevwadeprotest', 'roe v wade protest', 'pro choice', 'pro-choice', 
    #              '#prochoice', '#forcedbirth', 'forced birth', '#AbortionRightsAreHumanRights', 
    #              'abortion rights Are Human Rights', '#MyBodyMyChoice', 'My Body My Choice', 
    #              '#AbortionisHealthcare', 'abortion is healthcare', 'AbortionIsAHumanRight', 
    #              'abortion is a human right', 'ReproductiveHealth', 'Reproductive Health', 
    #              'AbortionRights', 'abortion rights' ]}
    # for w in pro_patterns[target]:
    #     if w.lower() in text.lower():
    #         return 'Support'
    # con_patterns = {
    #     'Metoo Movement':[],
    #     'Black Lives Matter Movement':[],        
    #     'the protection of Abortion Rights': ['#prolife', '#EndAbortion',
    #                 '#AbortionIsMurder', '#LifeIsAHumanRight', '#ChooseLife',
    #                 '#SaveTheBabyHumans', '#ValueLife', '#RescueThePreborn', '#EndRoeVWade',
    #                 '#MakeAbortionUnthinkable','#LiveActionAmbassador','#AbortionIsNotARight', '#AbortionIsRacist']}
    # for w in con_patterns[target]:
    #     if w.lower() in text.lower():
    #         return 'Oppose'
    # prompt = f"Given the comment, please answer whether the person who posted this comment believes in the occurrence of event {target}. Only output choice from Acceptance (believe this event), Doubt (doubt this event), and Rejection (do not believe this event).\n"
    # prompt = "What's the author's opinion on event {}? Please choose from Acceptance (believe this event), Doubt (doubt this event), and Rejection (do not believe this event). Only output your choice (Acceptance/Doubt/Rejection).\n\n".format(target)
    prompt = f"Based on the comment, output the confidence level of the person who made the comment in believing {target}. -1 means disbelief (they don't believe it), and 1 means belief (They believe it). only output a score (float number) in the range of [-1, 1]."
    text_sample = "Comment: "+text+'\n'+'Belief score: '
    prompt = prompt+text_sample
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "user", "content": prompt}
        ],
        temperature = 0,
        max_tokens = 16,
        )
    response = completion.choices[0].message.content
    ans = judge_response(response)
    # ans = judge_response(response, ['Acceptance', 'Doubt','Rejection'])
    return ans   



