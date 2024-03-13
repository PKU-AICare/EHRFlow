from typing import List, Dict, Optional

import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import optimize
from sklearn.cluster import KMeans

from xiaoya.pyehr.pipelines import DlPipeline
from xiaoya.pyehr.dataloaders.utils import unpad_batch
from xiaoya.pipeline import Pipeline


class DataAnalyzer:
    """
    DataAnalyzer.

    Parameters:
        config: Dict.
            the config of the pipeline.
        model_path: str.
            the saved path of the model.
    """

    def __init__(self, 
        config: Dict,
        model_path: str,
    ) -> None:
        self.config = config
        self.model_path = model_path

    def adaptive_feature_importance(
            self, 
            df: pd.DataFrame,
            x: List,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> Dict:
        """
        Return the adaptive feature importance of a patient.

        Parameters:
            df: A dataframe representing the patients' raw data.
            x: A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            patient_index: 
                The index of the patient in dataframe.
            patient_id: 
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns: 
            detail: List.
                a List of shape [time_step, feature_dim], representing the adaptive feature importance of the patient.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        _, _, feat_attn = pipeline.predict_step(x)
        feat_attn = feat_attn[0]
            
        return feat_attn.detach().cpu().numpy().tolist() # [ts, f]
    
    def feature_importance(
            self,
            df: pd.DataFrame,
            x: List,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> List:
        """
        Return feature importance of a patient.

        Parameters:
            df: pd.DataFrame.
                A dataframe representing the patients' raw data.
            x: List.
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            patient_index: Optional[int].
                The index of the patient in dataframe.
            patient_id: Optional[int].
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns:
            List: a List of dicts with shape [lab_dim],
                name: the name of the feature.
                value: the feature importance value.
                adaptive: the adaptive feature importance value.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        _, _, feat_attn = pipeline.predict_step(x)
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, f] feature importance value
        column_names = list(df.columns[6:])
        
        detail = sorted([{
            'name': column_names[i],
            'value': feat_attn[-1, i].item(),
            'adaptive': feat_attn[:, i].tolist(),
        } for i in range(len(column_names))], key=lambda x: x['value'], reverse=True)
        return detail # {'detail': detail}

    def risk_curve(
            self, 
            df: pd.DataFrame,
            x: List,
            mean: Dict,
            std: Dict,
            mask: Optional[List] = None,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> (List, List, List):
        """
        Return risk curve of a patient.
        
        We use EMR of patients as the input of the model `Concare`,
        and calculate the risk of the patient at each visit,
        also with feature importance of the patient at each visit.

        Parameters:
            df:
                A dataframe representing the patients' raw data.
            x:
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            mask:
                A List of shape [batch_size, time_step, feature_dim],
                representing the missing status of the patients's raw data.
            patient_index:
                The index of the patient in dataframe.
            patient_id:
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns: 
            List: A List of dicts with shape [lab_dim],
                name: the name of the feature.
                value: the value of the feature in all visits.
                importance: the feature importance value.
                adaptive: the adaptive feature importance value.
                missing: the missing status of the feature in all visits.
                unit: the unit of the feature.
            List: A List of shape [time_step],
                representing the date of the patient's visits.
            List: A List of shape [time_step],
                representing the death risk of the patient at each visit.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        y_hat, _, feat_attn = pipeline.predict_step(x)
        x = x[0, :, 2:].detach().cpu().numpy()  # [ts, lab]
        y_hat = y_hat[0].detach().cpu().numpy()  # [ts, 2]
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, lab]
        mask = np.array(mask[xid]) if mask is not None else np.zeros_like(feat_attn)  # [ts, lab]
        column_names = list(df.columns[6:])
        record_times = list(df[df['PatientID'] == patient_id]['RecordTime'].values) # [ts]
        
        detail = sorted([{
            'name': column_names[i],
            'value': (x[:, i] * std[column_names[i]] + mean[column_names[i]]).tolist(),
            'importance': feat_attn[-1, i].tolist(),
            'adaptive': feat_attn[:, i].tolist(),
            'missing': mask[:, i].tolist(),
            'unit': ''
        } for i in range(len(column_names))], key=lambda x: x['importance'], reverse=True)
        return detail, record_times, y_hat[:, 0]
        # {
        #     'detail': detail,
        #     'time': record_times,   # ts
        #     'time_risk': y_hat[:, 0],  # ts
        # }
    
    def ai_advice(
            self,
            df: pd.DataFrame,
            x: List,
            mean: Dict,
            std: Dict,
            time_index: int = -1,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> List:
        """
        Return the advice of the AI system.

        Parameters:
            df:
                A dataframe representing the patients' raw data.
            x: 
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            mean:
                A Dict with keys of feature names and mean values of all features.
            std:
                A Dict with keys of feature names and std values of all features.
            patient_index:
                The index of the patient in dataframe.
            patient_id:
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.
            time_index:
                The index of the visit of the patient, default is -1.

        Returns:
            List: A List of dicts with shape [num_advice], default `num_advice` is 3,
                representing the advice of the AI system.
                name: the name of the feature.
                old_value: the old value of the feature.
                new_value: the new value of the feature.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        device = torch.device('cuda:0' if pipeline.on_gpu else 'cpu')
        _, _, feat_attn = pipeline.predict_step(x.to(device))
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, f]

        demo_dim = 2
        column_names = list(df.columns[4 + demo_dim:])
        feature_last_step: List = feat_attn[time_index].tolist()
        index_dict = {index: value for index, value in enumerate(feature_last_step)}
        max_indices = sorted(index_dict, key=index_dict.get, reverse=True)
        if len(max_indices) > 3:
            max_indices = max_indices[:3]

        def f(x, args):
            input, i = args
            input[-1][-1][i] = torch.from_numpy(x).float()
            input = torch.cat((input, input), dim=0)
            y_hat, _, _ = pipeline.predict_step(input.to(device))      # y_hat: [bs, seq_len, 2]
            return y_hat[0][time_index][0].cpu().detach().numpy().item()

        result = []
        for i in max_indices:
            x0 = float(x[-1][-1][i])
            bounds = (max(-3, x0 - 1), min(3, x0 + 1))
            args = (x, i)
            res = optimize.minimize(f, x0=x0, bounds=(bounds,), args=(args,), method='nelder-mead', options={'disp': True})
            result.append({
                'name': column_names[i],
                'old_value': x0  * std[column_names[i]] + mean[column_names[i]],
                'new_value': res.x[0] * std[column_names[i]] + mean[column_names[i]]
            })
        return result # {'detail': result}

    def data_dimension_reduction(
            self,
            df: pd.DataFrame,
            x: List,
            mean_age: Optional[float],
            std_age: Optional[float],
            method: str = "PCA",
            dimension: int = 2,
            target: str = "outcome",
        )-> List:
        """
        Return dimension reduced data of the patients.

        Parameters:
            df:
                A dataframe representing the patients' raw data.
            x:
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            mean_age:
                The mean age of the patients.
            std_age:
                The std age of the patients.
            method:
                The method of dimension reduction, one of "PCA" and "TSNE", default is "PCA".
            dimension:
                The dimension of dimension reduction, one of 2 and 3, default is 2.
            target:
                The target of the model, one of "outcome", "los" and "multitask", default is "outcome".

        Returns:
            List: A List of dicts with shape [lab_dim],
                data: the dimension reduced data of the patient.
                patient_id: the patient ID of the patient.
                record_time: the visit datetime of the patient.
                age: the age of the patient.      
        """
        num = len(x)
        patients = []
        pid = df['PatientID'].drop_duplicates().tolist()  # [b]
        record_time = df.groupby('PatientID')['RecordTime'].apply(list).tolist()  # [b, ts]
        for i in range(num):
            xi = torch.tensor(x[i]).unsqueeze(0)
            pidi = torch.tensor(pid[i]).unsqueeze(0)
            timei = record_time[i]
            pipeline = DlPipeline(self.config)
            pipeline = pipeline.load_from_checkpoint(self.model_path)
            xi = torch.cat((xi, xi), dim=0)
            if pipeline.on_gpu:
                xi = xi.to('cuda:0')   # cuda
            y_hat, embedding, _ = pipeline.predict_step(xi)
            embedding = embedding[0].cpu().detach().numpy().squeeze()  # cpu
            y_hat = y_hat[0].cpu().detach().numpy().squeeze()      # cpu
            
            df = pd.DataFrame(embedding)
            if method == "PCA":  # 判断降维类别
                reduction_model = PCA().fit_transform(df)
            elif method == "TSNE":
                reduction_model = TSNE(n_components=dimension, learning_rate='auto', init='random').fit_transform(df)
            if(reduction_model.shape[0] != reduction_model.shape[1]):
                continue
            if target == "outcome":
                y_hat = y_hat[:, 0].flatten().tolist()
            else:
                y_hat = y_hat[:, 1].flatten().tolist()   
            
            patient = {}
            if dimension == 2:  # 判断降维维度
                patient['data'] = [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], y_hat)]
            elif dimension == 3:
                patient['data'] = [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], reduction_model[:, 2], y_hat)]
            patient['patient_id'] = pidi.item()
            patient['record_time'] = [str(x) for x in timei]
            if std_age is not None and mean_age is not None:
                patient['age'] = int(xi[0][0][1].item() * std_age + mean_age)
            patients.append(patient)
        return patients # {'detail': patients}
    
    def similar_patients(
            self,
            df: pd.DataFrame,
            x: List,
            mean: Dict,
            std: Dict,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
            n_clu: int = 10,
            topk: int = 6,
        ) -> List:
        """
        Return similar patients information.
        
        Parameters:
            df:
                A dataframe representing the patients' raw data.
            x:
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            mean:
                A Dict with keys of feature names and mean values of all features.
            std:
                A Dict with keys of feature names and std values of all features.
            patient_index:
                The index of the patient in dataframe.
            patient_id:
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.
            n_clu:
                The number of clusters, default is 10.
            topk:
                The number of similar patients, default is 6.
                
        Returns:
            List: A List of dicts with shape [topk],
                pid: the patient ID of the similar patient.
                context: the context of the similar patient.
                distance: the distance between the patient and the similar patient.
                similarity: the similarity between the patient and the similar patient.
        """
        
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat([x, x], dim=0)
        patients = [torch.Tensor(patient) for patient in x]       # [b, ts, f]
        patients = torch.nn.utils.rnn.pad_sequence(patients, batch_first=True, padding_value=0)
        if pipeline.on_gpu:
            x = x.to('cuda:0')
            patients = patients.to('cuda:0')
        _, x_context, _ = pipeline.predict_step(x)
        _, patients_context, _ = pipeline.predict_step(patients)
        patients, x_context, patients_context = patients.detach().cpu().numpy(), \
            x_context[0, -1, :].detach().cpu().numpy(), patients_context[:, -1, :].detach().cpu().numpy()
        # Cluster by Kmeans
        cluster = KMeans(n_clusters=n_clu).fit(patients_context)
        center_id = cluster.predict(x_context.reshape(1, -1))
        similar_patients_index = (cluster.labels_ == center_id)
        similar_patients_id = df['PatientID'].drop_duplicates()[similar_patients_index].tolist()
        similar_patients_x = patients[similar_patients_index]
        similar_patients_context = patients_context[similar_patients_index]
        
        detail = [{
            'pid': similar_patients_id[i],
            'context': (similar_patients_x[i] * np.array(list(std.values())) + np.array(list(mean.values()))).tolist(),
            'distance': np.sqrt(np.sum(np.square(x_context - similar_patients_context[i]))).item()
        } for i in range(similar_patients_context.shape[0])]
        
        dist = [x['distance'] for x in detail]
        maxDist, minDist = np.max(dist), np.min(dist)
        for x in detail:
            x['similarity'] = (x['distance'] - minDist) / (maxDist - minDist)
        
        return detail[:topk] 
        # {
        #     'detail': detail[:topk]
        # }
        
    def analyze_dataset(
            self,
            df: pd.DataFrame,
            x: List,
            feature: str,
            mean: Dict,
            std: Dict,
        ):
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        labtest_feature_index = df.columns[6:].tolist().index(feature)
        lens = [len(item) for item in x]
        
        x = [torch.Tensor(item) for item in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0) 
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        y_hat, _, feat_attn = pipeline.predict_step(x)
        x = x[:, :, 2 + labtest_feature_index].unsqueeze(-1)
        y_hat = y_hat[:, :, 0].unsqueeze(-1)
        feat_attn = feat_attn[:, :, labtest_feature_index].unsqueeze(-1)
        
        _, y = unpad_batch(x, y_hat, torch.Tensor(lens))
        x, feat_attn = unpad_batch(x, feat_attn, torch.Tensor(lens))
        data = pd.DataFrame(data={
            'Value': x * std[feature] + mean[feature],
            'Attention': feat_attn * 100.0,
            'Outcome': y,
        })
        # 2D data
        
        # 3D data
        outcome_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        attn_bins = list(np.arange(0.0, 101.0, 1.0))
        max_value = data['Value'].max()
        min_value = data['Value'].min()
        value_bins = list(map(lambda x: round(x, 1), np.arange(min_value, max_value, (max_value - min_value) / 51)))
        
        data_bar_2D = []
        data_line_2D = []
        data_3D = []
        for _, by_outcome in data.groupby(pd.cut(data['Outcome'], bins=outcome_bins), observed=False):
            data_2D_outcome = []
            data_3D_outcome = []
            for i, by_value in enumerate(by_outcome.groupby(pd.cut(by_outcome['Value'], bins=value_bins), observed=False)):
                data_2D_outcome.append([value_bins[i + 1], len(by_value[1])])
                for j, by_attn in enumerate(by_value[1].groupby(pd.cut(by_value[1]['Attention'], bins=attn_bins), observed=False)):
                    data_3D_outcome.append([value_bins[i + 1], attn_bins[j + 1], len(by_attn[1])])
            data_bar_2D.append(data_2D_outcome)
            data_3D.append(data_3D_outcome)
        for _, by_value in data.groupby(pd.cut(data['Value'], bins=value_bins), observed=False):
            data_line_2D.append([value_bins[i + 1], by_value[1]['Attention'].mean()])
        return data_bar_2D, data_line_2D, data_3D