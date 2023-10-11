#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""Scripts for visualizing the results of the grey box model"""

from __future__ import annotations

__author__ = "Carlos Alejandro Perez Garcia"
__copyright__ = "Copyright 2023"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Carlos Alejandro Perez Garcia"
__email__ = "cpgarcia518@gmail.com"

# Standard libraries
# ==============================================================================
import pandas as pd

# Visualization
# ==============================================================================
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Own libraries
# ==============================================================================
from greyboxmodel.base_model import GreyModelResult

# Functions
def plot_inut_data(data :pd.DataFrame, title: str = "Input data") -> go.Figure:
    """
    Plots the input data in a figure

    Parameters
    ----------
    data: `pandas.DataFrame`
        A pandas DataFrame of the input data
    title: str
        The title of the figure

    Returns
    -------
    fig: `plotly.graph_objs.Figure`
        A plotly figure
    """
    # Grouping input data
    temperature_columns = [col for col in data.columns if col.startswith('T')]
    power_columns = [col for col in data.columns if col.startswith('P')]

    # Variables
    subplots_titles = []
    cols = 1
    if len(power_columns) != 0:
        rows = 2
        subplots_titles = ["Temperature", "Power"]
    else:
        rows = 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        subplot_titles=subplots_titles
    )

    # Plotting temperature
    for col in temperature_columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col].values,
                mode='lines',
                name=col,
                showlegend=True,
                # line=dict(color='royalblue', width=2)
            ),
            row=1,
            col=1
        )

    # Plotting power
    for col in power_columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col].values,
                mode='lines',
                name=col,
                showlegend=True,
                # line=dict(color='firebrick', width=2)
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        title=title,
        # xaxis_title="datetime",
        yaxis_title="Temperature [°C]",
        legend_title="Legend",
        font=dict(
            family="Times New Roman",
            size=16,
        ),
        width=800,
        height=600,
    )

    if len(power_columns) != 0:
        fig.update_yaxes(title_text="Power [W]", row=2, col=1)

    return fig

def plot_results(model_result :GreyModelResult, raw_data :pd.Series=None, title: str = "Model Output") -> go.Figure:
    """
    Plots the input data in a figure

    Parameters
    ----------
    model_result: `GreyModelResult`
        A GreyModelResult
    raw_data: `pandas.Series`
        The raw data

    Returns
    -------
    fig: `plotly.graph_objs.Figure`
        A plotly figure
    """

    fig = make_subplots(rows=2, cols=2)

    # Plotting Results
    for key, value in model_result.var.items():
        fig.add_trace(
            go.Scatter(x=model_result.X.index, y=value, name=f"{key} Modelled", mode='lines'),
            row=1, col=1
        )

    # Plotting Raw Data
    if raw_data is not None:
        fig.add_trace(
            go.Scatter(x=raw_data.index, y=raw_data, name=f"Ti Measured", mode='lines'),
            row=1, col=1
        )

    # Plotting Residuals
    fig.add_trace(
        go.Scatter(
            x=model_result.X.index,
            y=raw_data - model_result.Z,
            name=f"Residuals",
            mode='markers',
            marker=dict(color='black', opacity=0.5)
        ),
        row=1, col=2
    )


    # # Plotting Residuals
    # for i, (key, value) in enumerate(model_result.var.items()):
    #     fig.add_trace(go.Scatter(x=model_result.index, y=model_result.residuals[key], name=f"{key} Residuals", mode='lines'), row=2, col=1)

    return fig


if __name__ == "__main__":
    pass