from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from pydantic import ValidationError

from GraceAgent.Action import Action
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PrintUtils import *


def _format_short_term_memory(memory):
    messages = memory.chat_memory.messages
    string_messages = [messages[i].content for i in range(1, len(messages))]
    return "\n".join(string_messages)


def _format_long_term_memory(task_description, memory):
    return memory.load_memory_variables(
        {"prompt": task_description},
    )["history"]


class GraceExecutor:
    """executor:GraceGPT 内部循环中负责具体任务执行的部分"""

    def __init__(
            self,
            llm: BaseChatModel,
            prompts_path: str,
            tools,
            work_dir: str = "./datasets",
            main_prompt_file: str = "executor.json",
            final_prompt_file: str = "final_step.json",
            max_thought_steps: Optional[int] = 10,

    ):
        self.llm = llm
        self.prompt_path = prompts_path
        self.tools = tools
        self.work_dir = work_dir
        self.max_thought_steps = max_thought_steps

        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)

        self.main_prompt_file = main_prompt_file
        self.final_prompt_file = final_prompt_file

    def run(self, inputs, verbose=True) -> str:
        thought_step_count = 0

        prompt_template = PromptTemplateBuilder(
            self.prompt_path,
            self.main_prompt_file
        ).build(
            tools=self.tools, output_parser=self.output_parser
        ).partial(
            # input=inputs["input"],
            history_info=inputs["previous_steps"],
            current_step=inputs["current_step"],
            work_dir=self.work_dir,
        )
        short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_tokens=4000
        )
        short_term_memory.save_context(
            {"input": "\初始化"},
            {"output": "\初始化"}
        )
        chain = (prompt_template | self.llm | StrOutputParser())

        reply = ""

        while thought_step_count < self.max_thought_steps:
            if verbose:
                color_print(f">>>>Round :{thought_step_count}<<<<", ROUND_COLOR)
            action, response = self._step(
                chain,
                short_term_memory=short_term_memory,
                verbose=verbose
            )
            print(action.name)
            if action.name == "FINISH":
                if verbose:
                    color_print(f"\n----\nFINISH", OBSERVATION_COLOR)
                reply = self._final_step(short_term_memory, inputs["current_step"], inputs["previous_steps"])
                break

            observation = self._exec_action(action)

            if verbose:
                color_print(f"\n----\n结果{observation}", OBSERVATION_COLOR)

            # 保存短时记忆
            short_term_memory.save_context(
                {"input": response + action.name},
                {"output": "返回结果：\n" + observation}
            )
            thought_step_count += 1
        if not reply:
            reply = "抱歉，我没能完成您的任务。"

        return reply

    def _step(self, reason_chain, short_term_memory, verbose=False):
        """执行一步步的思考"""
        response = ""
        for s in reason_chain.stream({
            "short_term_memory": _format_short_term_memory(short_term_memory),
        }):
            if verbose:
                color_print(s, THOUGHT_COLOR, end="")
            response += s

        action = self.robust_parser.parse(response)
        return action, response

    def _final_step(self, short_term_memory, task_description, previous_steps):
        final_prompt = PromptTemplateBuilder(
            self.prompt_path,
            self.final_prompt_file
        ).build().partial(
            task_description=task_description,
            history_info=previous_steps,
            short_term_memory=_format_short_term_memory(short_term_memory),
        )
        chain = (final_prompt | self.llm | StrOutputParser())
        response = chain.invoke({})
        # chain.invoke({})
        return response

    def _exec_action(self, action):
        # 查找工具
        tool = self._find_tool(action.name)
        if tool is None:
            observation = (
                f"Error:找不到工具或指令 '{action.name}'."
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                observation = (
                    f"Validation Error in args :{str(e)}, args:{action.args}")
            except Exception as e:
                observation = (
                    f"Error:{str(e)} ,{type(e).__name__},args:{action.args}"
                )
        return observation

    def _find_tool(self, tool_name):
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
