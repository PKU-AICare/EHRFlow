from abc import abstractmethod
from typing import Any, List, Optional
import re

from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PrintUtils import *
from Tools.CSVTool import get_first_n_rows
from langchain.callbacks.manager import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import ChatPromptTemplate

from langchain_experimental.plan_and_execute.schema import Plan, PlanOutputParser, Step
from langchain_experimental.pydantic_v1 import BaseModel


def _format_short_term_memory(memory):
    messages = memory.chat_memory.messages
    string_messages = [messages[i].content for i in range(1, len(messages))]
    return "\n".join(string_messages)


class PlanningOutputParser(PlanOutputParser):
    """Planning output parser."""

    def parse(self, text: str) -> Plan:
        text = text[text.find("&计划&"):]
        step_texts = re.findall(r"\d+\.(.*?)(?=\d+\.|$)", text, flags=re.DOTALL)
        steps = [Step(value=step.strip()) for step in step_texts]
        return Plan(steps=steps)


class GracePlanner:
    """LLM planner."""

    def __init__(self, llm: BaseLanguageModel, tools, csv_files,  output_parser: PlanOutputParser = PlanningOutputParser(),
                 stop: Optional[List] = None, verbose=False):
        self.llm = llm
        self.prompt_plan_template = PromptTemplateBuilder(prompt_path="./prompts/planner",
                                                          prompt_file="planner.json").build(
        ).partial(tool=str([(tool.name, tool.description) for tool in tools]), csv_files=str([get_first_n_rows(csv_file) for csv_file in csv_files]))
        self.prompt_refine_template = PromptTemplateBuilder(prompt_path="./prompts/planner",
                                                            prompt_file="planner_refine.json").build(
        ).partial(tool=str([(tool.name, tool.description) for tool in tools]))
        self.output_parser = output_parser
        self.stop = stop
        self.verbose = verbose

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""
        prompt_plan = self.prompt_plan_template.partial(
            previous_steps=inputs['previous_steps'] if 'previous_steps' in inputs else "",)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt_plan)
        llm_response = llm_chain.invoke(**inputs, stop=self.stop, callbacks=callbacks)
        color_print(str(llm_response['text']), CODE_COLOR, end="")
        plan = self.output_parser.parse(llm_response['text'])
        return plan

    def refine_plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        # finished_plan = [item[0].value for item in inputs['previous_steps']] if 'previous_steps' in inputs else ""
        prompt_refine = self.prompt_refine_template.partial(
            previous_steps=inputs['previous_steps'] if 'previous_steps' in inputs else "",
            plan=inputs['plan'] if 'plan' in inputs else "")
        llm_chain = LLMChain(llm=self.llm, prompt=prompt_refine)
        llm_response = llm_chain.invoke(**inputs, stop=self.stop, callbacks=callbacks)
        plan = self.output_parser.parse(llm_response['text'])
        return plan
