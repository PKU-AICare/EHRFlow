import pandas as pd
import numpy as np


def df_column_switch(df: pd.DataFrame, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df


def calculate_data_existing_length(data):
    res = 0
    for i in data:
        if not pd.isna(i):
            res += 1
    return res


# elements in data are sorted in time ascending order
def fill_missing_value(data, to_fill_value=0):
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)
    if data_len == data_exist_len:
        return data
    elif data_exist_len == 0:
        # data = [to_fill_value for _ in range(data_len)]
        for i in range(data_len):
            data[i] = to_fill_value
        return data
    if pd.isna(data[0]):
        # find the first non-nan value's position
        not_na_pos = 0
        for i in range(data_len):
            if not pd.isna(data[i]):
                not_na_pos = i
                break
        # fill element before the first non-nan value with median
        for i in range(not_na_pos):
            data[i] = to_fill_value
    # fill element after the first non-nan value
    for i in range(1, data_len):
        if pd.isna(data[i]):
            data[i] = data[i - 1]
    return data


def forward_fill_pipeline(
    df: pd.DataFrame,
    default_fill: pd.DataFrame,
    demographic_features: list[str],
    labtest_features: list[str],
):
    grouped = df.groupby('PatientID')

    all_x = []
    all_y = []
    all_pid = []
    all_missing_mask = []

    for name, group in grouped:
        sorted_group = group.sort_values(by=['RecordTime'], ascending=True)
        patient_x = []
        patient_y = []
        patient_missing_mask = pd.isna(sorted_group[labtest_features].values).tolist()

        for f in ['Age'] + labtest_features:
            to_fill_value = default_fill[f]
            # take median patient as the default to-fill missing value
            fill_missing_value(sorted_group[f].values, to_fill_value)

        for _, v in sorted_group.iterrows():
            patient_y.append([v['Outcome'], v['LOS']])
            x = []
            for f in demographic_features + labtest_features:
                x.append(v[f])
            patient_x.append(x)
        all_x.append(patient_x)
        all_y.append(patient_y)
        all_pid.append(name)
        all_missing_mask.append(patient_missing_mask)
    return all_x, all_y, all_pid, all_missing_mask

# outlier processing
def filter_outlier(element):
    if np.abs(float(element)) > 1e4:
        return 0
    else:
        return element


def normalize_dataframe(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    normalize_features: list[str],
):
    # Calculate the quantiles
    q_low = train_df[normalize_features].quantile(0.05)
    q_high = train_df[normalize_features].quantile(0.95)

    # Filter the DataFrame based on the quantiles
    filtered_df = train_df[(train_df[normalize_features] > q_low) & (
        train_df[normalize_features] < q_high)]

    # Calculate the mean and standard deviation and median of the filtered data, also the default fill value
    train_mean = filtered_df[normalize_features].mean()
    train_std = filtered_df[normalize_features].std()
    train_median = filtered_df[normalize_features].median()
    default_fill: pd.DataFrame = (train_median - train_mean) / (train_std + 1e-12)

    # LOS info
    los_info = {'los_mean': train_mean['LOS'], 'los_std': train_std['LOS'], 'los_median': train_median['LOS']}

    # Calculate large los and threshold (optional, designed for covid-19 benchmark)
    los_array = train_df.groupby('PatientID')['LOS'].max().values
    los_p95 = np.percentile(los_array, 95)
    los_p5 = np.percentile(los_array, 5)
    filtered_los = los_array[(los_array >= los_p5) & (los_array <= los_p95)]
    los_info.update({'large_los': los_p95.item(), 'threshold': filtered_los.mean().item()*0.5})

    # Z-score normalize the train, val, and test.py sets with train_mean and train_std
    train_df[normalize_features] = (train_df[normalize_features] - train_mean) / (train_std + 1e-12)
    val_df[normalize_features] = (val_df[normalize_features] - train_mean) / (train_std + 1e-12)
    test_df[normalize_features] = (test_df[normalize_features] - train_mean) / (train_std + 1e-12)

    train_df.loc[:, normalize_features] = train_df.loc[:, normalize_features].applymap(filter_outlier)
    val_df.loc[:, normalize_features] = val_df.loc[:, normalize_features].applymap(filter_outlier)
    test_df.loc[:, normalize_features] = test_df.loc[:, normalize_features].applymap(filter_outlier)

    return train_df, val_df, test_df, default_fill, los_info, train_mean, train_std


def normalize_df_with_statatistics(
    df: pd.DataFrame, 
    normalize_features: list[str], 
    train_mean, 
    train_std
):
    df[normalize_features] = (df[normalize_features] - train_mean) / (train_std + 1e-12)
    df.loc[:, normalize_features] = df.loc[:, normalize_features].applymap(filter_outlier)
    return df


def save_record_time(
    df: pd.DataFrame,
):
    grouped = df.groupby('PatientID')
    all_record_time = []
    for _, group in grouped:
        sorted_group = group.sort_values(by=['RecordTime'], ascending=True)
        record_time = sorted_group['RecordTime'].astype(int).tolist()
        all_record_time.append(record_time)
    return all_record_time


def one_hot_encoder(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    n_values = np.max(arr) + 1

    return np.eye(n_values)[arr]
