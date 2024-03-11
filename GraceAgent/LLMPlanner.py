from abc import abstractmethod
from typing import Any, List, Optional
import re

from Utils.PromptTemplateBuilder import PromptTemplateBuilder
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

    def __init__(self, llm: BaseLanguageModel,tools, output_parser: PlanOutputParser = PlanningOutputParser(),
                 stop: Optional[List] = None):
        self.prompt = None
        self.llm = llm
        self.prompt_template = PromptTemplateBuilder(prompt_path="./prompts/planner",
                                                     prompt_file="planner.json").build(
            tools=tools
        ).partial(
        )
        self.output_parser = output_parser
        self.stop = stop

    def plan(self, inputs: dict, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""
        self.prompt = self.prompt_template.partial(previous_steps=inputs['previous_steps'] if 'previous_steps' in inputs else "", plan=inputs['plan'] if 'plan' in inputs else "")
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        llm_response = llm_chain.invoke(**inputs, stop=self.stop)
        plan = self.output_parser.parse(llm_response['text'])
        return plan


