import pandas as pd
import os

def get_column_names(
        filename: str
) -> str:
    """获取 CSV 文件的列名"""
    df = pd.read_csv(filename)
    # sheet_name=0 表示第一个工作表

    column_names = '\n'.join(df.columns.to_list())
    result = f"这是'{filename}'文件的列名:\n\n {column_names}"
    return result


def is_csv(filename):
    """
    判断文件类型是否为 CSV
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() == '.csv'


def get_first_n_rows(filename: str, n: int = 3) -> str:
    """
    获取 CSV 文件的前 n 行，如果不是 CSV 文件则返回文件名
    """
    if not is_csv(filename):
        return filename
    """获取 CSV 文件的前 n 行"""
    result = f"这是'{filename}'文件所有列名：{get_column_names(filename)}"+ "\n\n"

    df = pd.read_csv(filename)

    n_lines = '\n'.join(df.head(n).to_string(index=False, header=True).split('\n'))

    result += f"这是'{filename}'文件的前{n}行样例:\n\n{n_lines}"
    return result
