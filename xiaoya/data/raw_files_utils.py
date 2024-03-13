import os
from typing import List

import pandas as pd


def get_features(df, table_type: str) -> List:
    """
    Get list of features from DataFrame.

    Args:
        df: DataFrame.
        table_type: str.
            'labtest' or 'events' or 'target'.

    Returns:
        feats: List.
            特征列表
    """

    if table_type in ['labtest', 'events']:
        feats = df['Name'].dropna().unique().tolist()
    else:
        feats = df.columns.tolist()
        feats.remove('PatientID')
        feats.remove('RecordTime')
    return feats

def to_dataframe(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """
    Change the format of DataFrame.

    Args:
        df: DataFrame.
        table_type: str.
            'labtest' or 'events' or 'target'.

    Returns:
        df: DataFrame.
            DataFrame in standard format.
    """
    if table_type == 'target':
        df = df.drop_duplicates(subset=['PatientID', 'RecordTime'], keep='last')
    else:
        df = df.drop_duplicates(subset=['PatientID', 'RecordTime', 'Name'], keep='last')
        columns = ['PatientID', 'RecordTime'] + list(df['Name'].dropna().unique())
        df_new = pd.DataFrame(data=None, columns=columns)
        grouped = df.groupby(['PatientID', 'RecordTime'])
        for i, group in enumerate(grouped):
            patient_id, record_time = group[0]
            df_group = group[1]
            df_new.loc[i, 'PatientID'] = patient_id
            df_new.loc[i, 'RecordTime'] = record_time
            for _, row in df_group.iterrows():
                df_new.loc[i, row['Name']] = row['Value']
        df = df_new

    df['RecordTime'] = pd.to_datetime(df['RecordTime'])
    df.sort_values(by=['PatientID', 'RecordTime'], inplace=True)
    return df

def merge_dfs(df_labtest, df_events, df_target) -> pd.DataFrame:
    """
    Merge DataFrames.

    """
    df = df_labtest 
    df = pd.merge(df, df_events, left_on=['PatientID', 'RecordTime'], right_on=['PatientID', 'RecordTime'], how='outer')
    df = pd.merge(df, df_target, left_on=['PatientID', 'RecordTime'], right_on=['PatientID', 'RecordTime'], how='outer')
    
    # Forward fill events.
    for col in df_events.columns.tolist():
        df[col] = df[col].fillna(method='ffill')

    # Change the order of columns.
    cols = ['PatientID', 'RecordTime', 'Outcome', 'LOS', 'Sex', 'Age']
    all_cols = df.columns.tolist()
    for col in cols:
        all_cols.remove(col) if col in all_cols else None
    all_cols = cols + all_cols
    df = df[all_cols]
    return df
