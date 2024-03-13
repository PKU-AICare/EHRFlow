import re
from langchain.tools import StructuredTool
from langchain_core.output_parsers import BaseOutputParser

from Utils.PrintUtils import color_print, CODE_COLOR
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain.output_parsers import PydanticOutputParser

import os
from langchain_core.pydantic_v1 import Field, BaseModel
from typing import Optional, List, Dict, Any


class PassAction(BaseModel):
    type: bool = Field(description="代码是否可以通过审核")
    content: str = Field(description="审核通过与否的原因")


class PythonCodeParser(BaseOutputParser):
    """从 Openai 返回的文本中提取 Python 代码"""

    def _remove_marked_lines(self, input_str: str) -> str:
        lines = input_str.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            del lines[0]
        if lines and lines[-1].strip().startswith('```'):
            del lines[-1]

        ans = '\n'.join(lines)
        return ans

    def parse(self, text: str) -> str:
        # 使用正则表达式找到所有的Python代码块
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        # 从re返回结果提取出Python代码文本
        python_code = None
        if len(python_code_blocks) > 0:
            python_code = python_code_blocks[0]
            python_code = self._remove_marked_lines(python_code)
        return python_code


class xiaoyaAnalyser:
    def __init__(self, prompt_path, info_path, verbose=True):
        self.prompt_code = PromptTemplateBuilder(prompt_path, "xiaoyaAgent.json").build()
        self.prompt_review = PromptTemplateBuilder(prompt_path, "code_reviewer.json")
        self.info = {}
        file_names = os.listdir(info_path)
        for file_name in file_names:
            if file_name.endswith(".txt"):
                key = file_name.rstrip(".txt")
                with open(os.path.join(info_path, file_name), 'r') as file:
                    content = file.read()
                    self.info[key] = content
        self.info['其他'] = ""
        self.verbose = verbose

    def analyse(self, inputs):
        """分析一个结构化文件（比如CSV文件）的内容"""
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            model_kwargs={"seed": 42}
        )
        code_type = inputs['type']

        passParser = PydanticOutputParser(pydantic_object=PassAction)

        code = ""
        res = {"type": False, "content": ""}

        while not res['type']:
            prompt_code_new = self.prompt_code.partial(code_description=self.info[code_type], code=code,
                                                       review_opinions=res['content'])
            chain_code = prompt_code_new | llm | PythonCodeParser()

            if self.verbose:
                color_print("\n#!/usr/bin/env python", CODE_COLOR, end="\n")

            for c in chain_code.stream({
                "query": inputs['content'],
                "filename": inputs['files'],
            }):
                if self.verbose:
                    color_print(c, CODE_COLOR, end="")
                code += c
            prompt_review_new = self.prompt_review.build(output_parser=passParser)
            chain_review = prompt_review_new | llm | passParser
            res = chain_review.invoke({"query": inputs['content'], "code": code})

        if code:
            ans = PythonREPLTool().run(code)
            return ans
        else:
            return "没有找到可执行的 Python 代码"

    def as_tool(self):
        return StructuredTool.from_function(
            func=self.analyse,
            name="AnalyseCSV",
            description="""通过程序脚本分析一个结构化文件（例如CSV文件）的内容。输入必须是以字典的形式进行指明,其中包含三个键： type、files和content。
            type:只能是(数据预处理、数据分析、数据可视化、其他)其中一项
            files:文件的完整路径,可以是多个文件
            content:尽可能完整的阐述当前分析阶段、具体分析方式和分析依据，阈值常量等。
            如果输入信息不完整，你可以拒绝回答。"""

        )
