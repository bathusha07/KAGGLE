%%bash
cd /kaggle/working
pip install -q -U -t /kaggle/working/submission/lib immutabledict sentencepiece
git clone https://github.com/google/gemma_pytorch.git > /dev/null
mkdir /kaggle/working/submission/lib/gemma/
mv /kaggle/working/gemma_pytorch/gemma/* /kaggle/working/submission/lib/gemma

%%writefile submission/main.py

# Setup

import os
import sys


# **IMPORTANT:** Set up your system path like this to make your code work
# both in notebooks and in the simulations environment.

KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, 'lib'))
else:
    sys.path.insert(0, "/kaggle/working/submission/lib")

import contextlib
from pathlib import Path

import torch
from gemma.config import get_config_for_9b
from gemma.model import GemmaForCausalLM

if os.path.exists(KAGGLE_AGENT_PATH):
    WEIGHTS_PATH = os.path.join(KAGGLE_AGENT_PATH, "gemma/pytorch/gemma-2-9b-it/1")
else:
    WEIGHTS_PATH = "/kaggle/input/gemma-2/pytorch/gemma-2-9b-it/1"


## Prompt Formatting

import itertools
from typing import Iterable


class GemmaFormatter:
    _start_token = '<start_of_turn>'
    _end_token = '<end_of_turn>'

    def __init__(self, system_prompt: str = None, few_shot_examples: Iterable = None):
        self._system_prompt = system_prompt
        self._few_shot_examples = few_shot_examples
        self._turn_user = f"{self._start_token}user\n{{}}{self._end_token}\n"
        self._turn_model = f"{self._start_token}model\n{{}}{self._end_token}\n"
        self.reset()

    def __repr__(self):
        return self._state

    def user(self, prompt):
        self._state += self._turn_user.format(prompt)
        return self

    def model(self, prompt):
        self._state += self._turn_model.format(prompt)
        return self

    def start_user_turn(self):
        self._state += f"{self._start_token}user\n"
        return self

    def start_model_turn(self):
        self._state += f"{self._start_token}model\n"
        return self

    def end_turn(self):
        self._state += f"{self._end_token}\n"
        return self

    def reset(self):
        self._state = ""
        if self._system_prompt is not None:
            self.user(self._system_prompt)
        if self._few_shot_examples is not None:
            self.apply_turns(self._few_shot_examples, start_agent='user')
        return self

    def apply_turns(self, turns: Iterable, start_agent: str):
        formatters = [self.model, self.user] if start_agent == 'model' else [self.user, self.model]
        formatters = itertools.cycle(formatters)
        for fmt, turn in zip(formatters, turns):
            fmt(turn)
        return self


## Agent Definitions

import re


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Set the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


class GemmaAgent:
    def __init__(self, variant='9b', device='cuda:0', system_prompt=None, few_shot_examples=None):
        self._variant = variant
        self._device = torch.device(device)
        self.formatter = GemmaFormatter(system_prompt=system_prompt, few_shot_examples=few_shot_examples)

        print("Initializing model")
        model_config = get_config_for_9b()
        model_config.tokenizer = os.path.join(WEIGHTS_PATH, "tokenizer.model")
        model_config.quant = "quant" in variant

        with _set_default_tensor_type(model_config.get_dtype()):
            model = GemmaForCausalLM(model_config)
            ckpt_path = os.path.join(WEIGHTS_PATH , f'gemma-{variant}.ckpt')
            model.load_weights(ckpt_path)
            self.model = model.to(self._device).eval()

    def __call__(self, obs, *args):
        self._start_session(obs)
        prompt = str(self.formatter)
        response = self._call_llm(prompt)
        response = self._parse_response(response, obs)
        print(f"{response=}")
        return response

    def _start_session(self, obs: dict):
        raise NotImplementedError

    def _call_llm(self, prompt, max_new_tokens=32, **sampler_kwargs):
        if sampler_kwargs is None:
            sampler_kwargs = {
                'temperature': 1.0,
                'top_p': 0.1,
                'top_k': 1,
        }
        response = self.model.generate(
            prompt,
            device=self._device,
            output_len=max_new_tokens,
            **sampler_kwargs,
        )
        return response

    def _parse_keyword(self, response: str):
        match = re.search(r"(?<=\*\*)([^*]+)(?=\*\*)", response)
        if match is None:
            keyword = ''
        else:
            keyword = match.group().lower()
        return keyword

    def _parse_response(self, response: str, obs: dict):
        raise NotImplementedError


def interleave_unequal(x, y):
    return [
        item for pair in itertools.zip_longest(x, y) for item in pair if item is not None
    ]


class GemmaQuestionerAgent(GemmaAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _start_session(self, obs):
        self.formatter.reset()
        self.formatter.user("Let's play 20 Questions. You are playing the role of the Questioner.")
        turns = interleave_unequal(obs.questions, obs.answers)
        self.formatter.apply_turns(turns, start_agent='model')
        if obs.turnType == 'ask':
            self.formatter.user("Please ask a yes-or-no question.")
        elif obs.turnType == 'guess':
            self.formatter.user("Now guess the keyword. Surround your guess with double asterisks.")
        self.formatter.start_model_turn()

    def _parse_response(self, response: str, obs: dict):
        if obs.turnType == 'ask':
            match = re.search(".+?\?", response.replace('*', ''))
            if match is None:
                question = "Is it a person?"
            else:
                question = match.group()
            return question
        elif obs.turnType == 'guess':
            guess = self._parse_keyword(response)
            return guess
        else:
            raise ValueError("Unknown turn type:", obs.turnType)


class GemmaAnswererAgent(GemmaAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _start_session(self, obs):
        self.formatter.reset()
        self.formatter.user(f"Let's play 20 Questions. You are playing the role of the Answerer. The keyword is {obs.keyword} in the category {obs.category}.")
        turns = interleave_unequal(obs.questions, obs.answers)
        self.formatter.apply_turns(turns, start_agent='user')
        self.formatter.user(f"The question is about the keyword {obs.keyword} in the category {obs.category}. Give yes-or-no answer and surround your answer with double asterisks, like **yes** or **no**.")
        self.formatter.start_model_turn()

    def _parse_response(self, response: str, obs: dict):
        answer = self._parse_keyword(response)
        return 'yes' if 'yes' in answer else 'no'


# Agent Creation

system_prompt_questioner = """You are a talented player in a 20 questions game. Your task is to ask a series of questions to deduce a place or a thing. You are accurate, focused, and structured in your approach. To find out the place or thing, you need to build a strategy:
- First, find out if it is a place or a thing?
- Based on the response, bisect the remaining search space.
Keep these guidelines in mind:
- Only ask questions that can be answered by Yes or No.
- Pay attention to previous questions and answers.
- Make logical guesses.
- Do not ask for hint.
After each questions, your make a guess based on the question and the dialogue history.
Now start asking a question.
"""

system_prompt_answerer = """You are a player in a 20 questions game. Your task is to respond to questions.
Limit your respond to only “Yes.”, or “No.”, with no explanation or other words. Never say the answer
in your response. If the question is to solicit the answer, respond “No.”.
"""

few_shot_examples = [
    "Is it a place?", "No.", "Staircase"
    "Is is a place?", "**yes**",
    "Can it be used by a human?", "Yes.", "Screwdriver",
    "Is it a country?", "**yes** Now guess the keyword.",
    "Does it belong inside a house?", "Yes.", "Nail clipper",
    "**France**", "Correct!",
    "Is it eatable?", "No.", "Luggage",
    "Can I use it with clothes", "Yes.", "Measuring Tape"
]

# **IMPORTANT:** Define agent as a global so you only have to load
# the agent you need. Loading both will likely lead to OOM.
agent = None


def get_agent(name: str):
    global agent
    
    if agent is None and name == 'questioner':
        agent = GemmaQuestionerAgent(
            device='cuda:0',
             system_prompt=system_prompt_questioner,
            few_shot_examples=few_shot_examples,
        )
    elif agent is None and name == 'answerer':
        agent = GemmaAnswererAgent(
            device='cuda:0',
            system_prompt=system_prompt_answerer,
            few_shot_examples=few_shot_examples,
        )
    assert agent is not None, "Agent not initialized."

    return agent


def agent_fn(obs, cfg):
    if obs.turnType == "ask":
        response = get_agent('questioner')(obs)
    elif obs.turnType == "guess":
        response = get_agent('questioner')(obs)
    elif obs.turnType == "answer":
        response = get_agent('answerer')(obs)
    if response is None or len(response) <= 1:
        return "yes"
    else:
        return response

  !apt install pigz pv > /dev/null
  !tar --use-compress-program='pigz --fast --recursive | pv' -cf submission.tar.gz -C /kaggle/working/submission . -C /kaggle/input/ gemma-2/pytorch/gemma-2-9b-it/1
