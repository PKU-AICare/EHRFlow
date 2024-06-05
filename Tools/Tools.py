import warnings

from pydantic import Field

warnings.filterwarnings("ignore")
from langchain.tools import StructuredTool
from .FileTools import list_files_in_directory
from .WriterTool import write
from .CSVTool import get_first_n_rows, get_column_names

#
# document_generation_tool = StructuredTool.from_function(
#     func=write,
#     name="GenerateDocument",
#     description="根据需求描述生成一篇正式文档"
# )


CSV_inspection_tool = StructuredTool.from_function(
    func=get_first_n_rows,
    name="InspectCSV",
    description="探查表格文件的内容和结构，展示它的列名和前n行，n默认为3"
)

directory_inspection_tool = StructuredTool.from_function(
    func=list_files_in_directory,
    name="ListDirectory",
    description="探查文件夹的内容和结构，展示它的文件名和文件夹名"
)

finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符：FINISH"
)
