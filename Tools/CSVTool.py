import pandas as pd


def get_column_names(
        filename: str
) -> str:
    """获取 CSV 文件的列名"""
    df = pd.read_csv(filename)
    # sheet_name=0 表示第一个工作表

    column_names = '\n'.join(df.columns.to_list())
    result = f"这是'{filename}'文件的列名:\n\n {column_names}"
    return result


def get_first_n_rows(
        filename: str,
        n: int = 3
) -> str:
    """获取 CSV 文件的前 n 行"""
    result = get_column_names(filename) + "\n\n"

    df = pd.read_csv(filename)

    n_lines = '\n'.join(df.head(n).to_string(index=False, header=True).split('\n'))

    result += f"这是'{filename}'文件的前{n}行样例:\n\n{n_lines}"
    return result
