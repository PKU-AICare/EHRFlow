from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, List, Dict, Any


class Action(BaseModel):
    name: str = Field(description="工作或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工作或指令参数,由参数名称和参数值组成")

if __name__ == "__main__":
    from langchain.output_parsers import PydanticOutputParser
    outputparser = PydanticOutputParser(pydantic_object=Action)