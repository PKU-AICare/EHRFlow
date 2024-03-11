from langchain.tools import BaseTool, Tool
from langchain.prompts import load_prompt, BasePromptTemplate
from langchain.schema.output_parser import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain_core.prompts import PipelinePromptTemplate, BasePromptTemplate

import sys

sys.path.append('..')
from typing import Optional, List, Dict, Any, Sequence
import os
import json
import tempfile


def chinese_friendly(string) -> str:
    lines = string.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return "\n".join(lines)


def load_file(file_name: str) -> str:
    """Loads a file into a string."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found.")
    f = open(file_name, 'r', encoding='utf-8')
    s = f.read()
    f.close()
    return s


class PromptTemplateBuilder:
    def __init__(self, prompt_path: str, prompt_file: str) -> None:
        self.prompt_path = prompt_path
        self.prompt_file = prompt_file

    def _check_or_redirect(self, prompt_file: str) -> str:
        with open(prompt_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "template_path" in config:
            if not os.path.isabs(config["template_path"]):
                config["template_path"] = os.path.join(
                    self.prompt_path, config["template_path"]
                )
                tmp_file = tempfile.NamedTemporaryFile(
                    suffix=".json",
                    mode="w",
                    encoding="utf-8",
                    delete=False)
                tmp_file.write(json.dumps(config, ensure_ascii=False))
                tmp_file.close()
                return tmp_file.name
        return prompt_file

    def _get_tools_prompt(self, tools: Sequence[BaseTool]) -> str:
        tools_prompt = ""
        for i, tool in enumerate(tools):
            prompt = f"{i + 1}. {tool.name}:{tool.description},\
                    args json schema:{json.dumps(tool.args, ensure_ascii=False)}\n"
            tools_prompt += prompt
        return tools_prompt

    def build(
            self,
            output_parser: Optional[BaseOutputParser] = None,
            tools: Optional[Sequence[BaseTool]] = None
    ) -> BasePromptTemplate:
        main_file = os.path.join(self.prompt_path, self.prompt_file)
        # 主要是找到 executor.json 文件，但是langchain 中 path必须是绝对路径，因此需要转换
        main_prompt_template = load_prompt(self._check_or_redirect(main_file))
        variables = main_prompt_template.input_variables
        partial_variables = {}
        recursive_templates = []

        # 遍历所有变量，检查是否存在对应的模版文件
        for var in variables:
            if os.path.exists(os.path.join(self.prompt_path, f"{var}.json")):
                sub_template = PromptTemplateBuilder(
                    self.prompt_path, f"{var}.json"
                ).build(tools=tools, output_parser=output_parser)
                recursive_templates.append(sub_template)
            elif os.path.exists(os.path.join(self.prompt_path, f"{var}.txt")):
                var_str = load_file(os.path.join(self.prompt_path, f"{var}.txt"))
                partial_variables[var] = var_str

        if tools is not None and "tools" in variables:
            tools_prompt = self._get_tools_prompt(tools)
            partial_variables["tools"] = tools_prompt

        if output_parser is not None and "format_instructions" in variables:
            partial_variables["format_instructions"] = chinese_friendly(
                output_parser.get_format_instructions())

        if recursive_templates:
            # 将有值嵌套的模版嵌套到主模版中
            main_prompt_template = PipelinePromptTemplate(
                final_prompt=main_prompt_template,
                pipeline_prompts=recursive_templates
            )
        main_prompt_template = main_prompt_template.partial(**partial_variables)
        return main_prompt_template


if __name__ == "__main__":
    prompt = PromptTemplateBuilder(prompt_path="../prompts/planner", prompt_file="planner.json").build().partial(
            work_dir='./datasets'
        )
    print(prompt)
