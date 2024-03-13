import os
from typing import List, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .processed_datasets_utils import (
    normalize_dataframe,
    forward_fill_pipeline,
    save_record_time,
)


class DataHandler:
    """
    Import user uploaded data, merge data tables, stats...

    Args:
        labtest_data: DataFrame.
        events_data: DataFrame.
        target_data: DataFrame.
        data_path: Path.
            path to save processed data, default is Path('./datasets').
    """

    def __init__(
            self, 
            labtest_data: pd.DataFrame,
            events_data: pd.DataFrame,
            target_data: pd.DataFrame,
            data_path: Path = Path('./datasets'),
        ) -> None:

        self.raw_df = {
            'labtest': pd.DataFrame(labtest_data),
            'events': pd.DataFrame(events_data),
            'target': pd.DataFrame(target_data),
        }
        self.raw_features = {}
        self.standard_df = {}
        self.merged_df = None
        self.data_path = data_path

    def format_dataframe(
            self, 
            format: str,
        ) -> pd.DataFrame:
        """
        Formats the data in the DataFrame according to the specified format.
        
        Args:
            format: str. 
                The format to use for formatting the data, must be one of ['labtest', 'events', 'target'].
        
        Returns:
            pd.DataFrame: The formatted DataFrame.
        """
        assert format in ['labtest', 'events', 'target'], "format must be one of ['labtest', 'events', 'target']"

        df = self.raw_df[format]
        if format == 'target':
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
        self.standard_df[format] = df
        return df

    def merge_dataframes(self) -> pd.DataFrame:
        """
        Merge the dataframes.

        Returns:
            pd.DataFrame: The merged Dataframe.
        """
        labtest_standard_df: pd.DataFrame = self.standard_df.get('labtest', None)
        events_standard_df: pd.DataFrame = self.standard_df.get('events', None)
        target_standard_df: pd.DataFrame = self.standard_df.get('target', None)

        assert labtest_standard_df is not None and events_standard_df is not None and target_standard_df is not None, \
        "Please format all dataframes first."

        df = labtest_standard_df 
        df = pd.merge(df, events_standard_df, left_on=['PatientID', 'RecordTime'], right_on=['PatientID', 'RecordTime'], how='outer')
        df = pd.merge(df, target_standard_df, left_on=['PatientID', 'RecordTime'], right_on=['PatientID', 'RecordTime'], how='outer')
        
        # Forward fill events.
        for col in events_standard_df.columns.tolist():
            df[col] = df[col].fillna(method='ffill')
        
        # Change the order of columns.
        cols = ['PatientID', 'RecordTime', 'Outcome', 'LOS', 'Sex', 'Age']
        all_cols = df.columns.tolist()
        for col in cols:
            all_cols.remove(col) if col in all_cols else None
        all_cols = cols + all_cols
        merged_df = df[all_cols]
        self.merged_df = merged_df
        return merged_df
    
    def format_and_merge_dataframes(self) -> pd.DataFrame:
        """
        Format and merge the dataframes.

        Returns:
            pd.DataFrame: The merged Dataframe.
        """
        self.format_dataframe('labtest')
        self.format_dataframe('events')
        self.format_dataframe('target')
        merged_df = self.merge_dataframes()
        return merged_df

    def save_processed_data(self) -> None:
        """
        Save processed data to specified directory.
        """
        
        self.standard_df['labtest'].to_csv(os.path.join(self.data_path, 'standard_labtest_data.csv'), index=False)
        self.standard_df['events'].to_csv(os.path.join(self.data_path, 'standard_events_data.csv'), index=False)
        self.standard_df['target'].to_csv(os.path.join(self.data_path, 'standard_target_data.csv'), index=False)
        self.merged_df.to_csv(os.path.join(self.data_path, 'standard_merged_data.csv'), index=False)

    def extract_features(
            self, 
            format: str,
        ) -> Dict:
        """
        Extract features from the merged dataframe.

        Args:
            format: str.
                'labtest' or 'events' or 'target'.
        
        Returns:
            Dict: Extracted features from raw dataframe.
        """
        assert format in ['labtest', 'events', 'target'], "format must be one of ['labtest', 'events', 'target']"

        df: pd.DataFrame = self.raw_df[format]
        if format in ['labtest', 'events']:
            feats = df['Name'].dropna().unique().tolist()
        else:
            feats = df.columns.tolist()
            feats.remove('PatientID')
            feats.remove('RecordTime')
        self.raw_features[format] = feats
        return feats

    def list_all_features(self) -> Dict:
        """
        List all features.

        Returns:
            Dict: All extracted features from raw dataframes.
        """
        features = {}
        for format in ['labtest', 'events', 'target']:
            features[format] = self.extract_features(format)
        return features

    def analyze_dataset(self) -> Dict:
        """
        Analyze the dataset.

        Returns:
            Dict: The main data in 'detail' is a List of imformation of all features.
        """
        
        detail = []
        features = list(self.merged_df.columns)
        features.remove('RecordTime') if 'RecordTime' in features else None
        for idx, feature in enumerate(features):
            info = {}
            info["name"] = feature
            info["value"] = list(self.merged_df[feature])
            info["stats"] = []
            info["stats"].append({"name": "id", "value": idx})
            info["stats"].append({"name": "count", "value": int(self.merged_df[feature].count())})
            info["stats"].append({"name": "missing", "value": str(round(float((100 - self.merged_df[feature].count() * 100 / len(self.merged_df.index))), 2)) + "%"})
            info["stats"].append({"name": "mean", "value": round(float(self.merged_df[feature].mean()), 2)})
            info["stats"].append({"name": "max", "value": round(float(self.merged_df[feature].max()), 2)})
            info["stats"].append({"name": "min", "value": round(float(self.merged_df[feature].min()), 2)})
            info["stats"].append({"name": "median", "value": round(float(self.merged_df[feature].median()), 2)})
            info["stats"].append({"name": "std", "value": round(float(self.merged_df[feature].std()), 2)})
            detail.append(info)
        return {'detail': detail}
    
    def split_dataset(self, 
            train_size: int = 70, 
            val_size: int = 10, 
            test_size: int = 20, 
            seed: int = 42
        ) -> None:
        """
        Split the dataset into train/val/test sets.

        Args:
            train_size: int.
                train set percentage.
            val_size: int.
                val set percentage.
            test_size: int.
                test set percentage.
            seed: int.
                random seed.
        """
        assert train_size + val_size + test_size == 100, "train_size + val_size + test_size must equal to 100"

        # Group the dataframe by patient ID
        grouped = self.merged_df.groupby('PatientID')
        patients = np.array(list(grouped.groups.keys()))
        
        # Get the train_val/test patient IDs
        patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])
        train_val_patients, test_patients = train_test_split(patients, test_size=test_size/(train_size+val_size+test_size), random_state=seed, stratify=patients_outcome)

        # Get the train/val patient IDs
        train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
        train_patients, val_patients = train_test_split(train_val_patients, test_size=val_size/(train_size+val_size), random_state=seed, stratify=train_val_patients_outcome)

        #  Create train, val, test, [traincal, calib] dataframes for the current fold
        self.train_raw_df = self.merged_df[self.merged_df['PatientID'].isin(train_patients)]
        self.val_raw_df = self.merged_df[self.merged_df['PatientID'].isin(val_patients)]
        self.test_raw_df = self.merged_df[self.merged_df['PatientID'].isin(test_patients)]
    
    def save_record_time(self) -> None:
        """
        Save the record time of each patient.
        """

        self.train_record_time = save_record_time(self.train_raw_df)
        self.val_record_time = save_record_time(self.val_raw_df)
        self.test_record_time = save_record_time(self.test_raw_df)

    def normalize_dataset(self,
            normalize_features: List[str]
        ) -> None:
        """
        Normalize the dataset.

        Args:
            normalize_features: List[str].
                features to be normalized.
        """

        # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
        train_after_zscore, val_after_zscore, test_after_zscore, self.default_fill, self.los_info, self.train_mean, self.train_std = \
            normalize_dataframe(self.train_raw_df, self.val_raw_df, self.test_raw_df, normalize_features)
        
        # Drop rows if all features are recorded NaN
        self.train_after_zscore = train_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        self.val_after_zscore = val_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        self.test_after_zscore = test_after_zscore.dropna(axis=0, how='all', subset=normalize_features)

    def forward_fill_dataset(self,
            demographic_features: List[str],
            labtest_features: List[str]
        ) -> None:
        """
        Forward fill the dataset.

        Args:
            demographic_features: List[str].
                demographic features.
            labtest_features: List[str].
                lab test features.
        """
        
        # Forward Imputation after grouped by PatientID
        self.train_x, self.train_y, self.train_pid, self.train_missing_mask = forward_fill_pipeline(self.train_after_zscore, self.default_fill, demographic_features, labtest_features)
        self.val_x, self.val_y, self.val_pid, self.val_missing_mask = forward_fill_pipeline(self.val_after_zscore, self.default_fill, demographic_features, labtest_features)
        self.test_x, self.test_y, self.test_pid, self.test_missing_mask = forward_fill_pipeline(self.test_after_zscore, self.default_fill, demographic_features, labtest_features)

    def execute(self,
            train_size: int = 70,
            val_size: int = 10,
            test_size: int = 20,
            seed: int = 42,
        ) -> None:
        """
        Execute the preprocessing pipeline, including format and merge dataframes, split the dataset, normalize the dataset, and forward fill the dataset.

        Args:
            train_size: int.
                train set percentage.
            val_size: int.
                val set percentage.
            test_size: int.
                test set percentage.
            seed: int.
                random seed.
        """
        
        data_path = self.data_path
        
        # Extract features
        self.list_all_features()

        # Format and merge the dataframes
        self.format_and_merge_dataframes()
        
        # Save processed data
        self.save_processed_data()
        
        demographic_features: List = self.raw_features['events']
        labtest_features: List = self.raw_features['labtest']
        if 'Age' in labtest_features:
            demographic_features.append('Age')
            labtest_features.remove('Age') 

        # Split the dataset
        self.split_dataset(train_size, val_size, test_size, seed)

        # Save record time
        self.save_record_time()

        # Normalize the dataset
        self.normalize_dataset(['Age'] + labtest_features + ['LOS'])

        # Forward fill the dataset
        self.forward_fill_dataset(demographic_features, labtest_features)

        # Save the dataframes
        data_path.mkdir(parents=True, exist_ok=True)
        self.train_raw_df.to_csv(os.path.join(data_path, 'train_raw.csv'), index=False)
        self.val_raw_df.to_csv(os.path.join(data_path, 'val_raw.csv'), index=False)
        self.test_raw_df.to_csv(os.path.join(data_path, 'test_raw.csv'), index=False)

        self.train_after_zscore.to_csv(os.path.join(data_path, 'train_after_zscore.csv'), index=False)
        self.val_after_zscore.to_csv(os.path.join(data_path, 'val_after_zscore.csv'), index=False)
        self.test_after_zscore.to_csv(os.path.join(data_path, 'test_after_zscore.csv'), index=False)

        pd.to_pickle(self.train_x, os.path.join(data_path, 'train_x.pkl'))
        pd.to_pickle(self.train_y, os.path.join(data_path, 'train_y.pkl'))
        pd.to_pickle(self.train_record_time, os.path.join(data_path, 'train_record_time.pkl'))
        pd.to_pickle(self.train_pid, os.path.join(data_path, 'train_pid.pkl'))
        pd.to_pickle(self.train_missing_mask, os.path.join(data_path, 'train_missing_mask.pkl'))
        pd.to_pickle(self.val_x, os.path.join(data_path, 'val_x.pkl'))
        pd.to_pickle(self.val_y, os.path.join(data_path, 'val_y.pkl'))
        pd.to_pickle(self.val_record_time, os.path.join(data_path, 'val_record_time.pkl'))
        pd.to_pickle(self.val_pid, os.path.join(data_path, 'val_pid.pkl'))
        pd.to_pickle(self.val_missing_mask, os.path.join(data_path, 'val_missing_mask.pkl'))
        pd.to_pickle(self.test_x, os.path.join(data_path, 'test_x.pkl'))
        pd.to_pickle(self.test_y, os.path.join(data_path, 'test_y.pkl'))
        pd.to_pickle(self.test_record_time, os.path.join(data_path, 'test_record_time.pkl'))
        pd.to_pickle(self.test_pid, os.path.join(data_path, 'test_pid.pkl'))
        pd.to_pickle(self.test_missing_mask, os.path.join(data_path, 'test_missing_mask.pkl'))
        pd.to_pickle(self.los_info, os.path.join(data_path, 'los_info.pkl'))
        pd.to_pickle(dict(self.train_mean), os.path.join(data_path, 'train_mean.pkl'))
        pd.to_pickle(dict(self.train_std), os.path.join(data_path, 'train_std.pkl'))
