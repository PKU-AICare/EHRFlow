from typing import List, Dict
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_vis_dataset(
        data: List,
        save_path: str,
    ) -> None:
    """
    Plot the distribution of the dataset.

    Args:
        data: List.
            List of patients' data.
        save_path: str.
            Path to save the plot.
    """

    Path(save_path).mkdir(parents=True, exist_ok=True)
    for feature in data:
        plt.cla()
        plt.hist(feature['value'], bins=20, edgecolor='black')
        plt.title(f'{feature["name"]}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_path, f'{feature["name"]}_hist.png'))


def plot_feature_importance(
        data: List,
        save_path: str,
        feature_num: int=10,
        file_name: str='feature_importance',
    ) -> None:
    """
    Plot the feature importance as a bar chart.

    Args:
        data: List.
            List of patients' data.
        save_path: str.
            Path to save the plot.
        feature_num: int.
            Number of features to plot, default 10.
        file_name: str.
            File name of the plot.
    """

    importance = {data[i]['name']: data[i]['value'] for i in range(len(data))}
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in importance[:feature_num]]
    values = [item[1] for item in importance[:feature_num]]

    plt.figure(figsize=(12, 6))
    plt.barh(names, values, color='blue', alpha=0.75)
    plt.xlabel('Importance Index')
    plt.title('Feature Importance')
    plt.xlim(min(values) / 2, max(values))  # Adjust the x-axis range for better visualization
    plt.gca().invert_yaxis()  # Invert y-axis for top-down display
    plt.tight_layout()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'))


def plot_risk_curve(
        data: List,
        time: List,
        time_risk: List,
        save_path: str,
        feature_num: int=3,
        file_name: str='risk_curve',
    ) -> None:
    """
    Plot the risk curve of a patient.

    Args:
        data: Dict.
            Data to plot.
        save_path: str.
            Path to save the plot.
        feature_num: int.
            Number of features to plot, default 3.
        file_name: str.
            File name of the plot.
    """

    x = list(range(len(time)))
    x_label = time
    ys = [time_risk]
    y_labels = ['Risk Index']
    for i in range(feature_num):
        ys.append(data[i]['value'])
        y_labels.append(data[i]['name'])
    colors = sns.color_palette("hls", len(ys))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Date', fontsize=14)
    ax.set_xticks(ticks=x, labels=x_label, rotation=45, fontsize=14)

    twins = [ax]
    for i in range(len(ys) - 1):
        twins.append(twins[-1].twinx())
    pad = 0
    for i, twin in enumerate(twins):
        twin.plot(x, ys[i], 'o-', color=colors[i], label=y_labels[i])
        twin.set_ylabel(y_labels[i], fontsize=14)
        pos = 'right' if i > 0 else 'left'
        twin.yaxis.set_ticks_position(pos)
        twin.yaxis.set_label_position(pos)
        twin.yaxis.label.set_color(colors[i])
        if pos == 'right':
            twin.spines[pos].set_position(('outward', pad))
            pad += 60
        else:
            twin.set_ylim(0.2, 1)
            twin.fill_between(x, ys[i], color=colors[i], alpha=0.1)
    plt.title('Health Metrics Over Time')
    plt.tight_layout()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'))


def plot_patient_embedding(
        data: List,
        save_path: str,
        dimension: int = 2,
        file_name: str = 'patient_embedding_reduction',
    ) -> None:
    """
    Plot patients' embeddings in 2D or 3D space.

    Args:
        data: List.
            List of patients' embeddings.
        save_path: str.
            Path to save the plot.
        dimension: int.
            Dimension of the plot. Must be 2 or 3.
    """

    assert dimension in [2, 3], "dimension must be 2 or 3"

    plt.figure(figsize=(6, 6))
    for patient in data:   
        if dimension == 2: 
            df_subset = pd.DataFrame(data=patient['data'], columns=['2d-one', '2d-two', 'target'])
            sns.scatterplot(
                x='2d-one',
                y='2d-two',
                hue='target',
                palette=sns.color_palette('coolwarm', as_cmap=True),
                data=df_subset,
                legend=False,
                alpha=0.3,
            )
        elif dimension == 3:
            df_subset = pd.DataFrame(data=patient['data'], columns=['3d-one', '3d-two', '3d-three', 'target'])
            sns.scatterplot(
                x='3d-one',
                y='3d-two',
                hue='target',
                palette=sns.color_palette('coolwarm', as_cmap=True),
                data=df_subset,
                legend=False,
                alpha=0.3,
            )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{file_name}.png'))
