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
import numpy as np
from statsmodels.tsa.stattools import pacf, acf
from sklearn.metrics import mean_squared_error, r2_score

# Visualization
# ==============================================================================
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Own libraries
# ==============================================================================
from greyboxmodel.base_model import GreyModelResult

# Functions
def plot_inut_data(data :pd.DataFrame, graph_conf :dict=None, title: str = "Input data") -> go.Figure:
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
    )

    if graph_conf is not None:
        fig.update_layout(
            font=dict(
                family=graph_conf['font_family'],
                size=graph_conf['font_size'],
            ),
            width=graph_conf['width'],
            height=graph_conf['height'],
            template=graph_conf['theme'],
        )

    if len(power_columns) != 0:
        fig.update_yaxes(title_text="Power [W]", row=2, col=1)

    return fig

def plot_results(model_result :GreyModelResult, raw_data :pd.Series=None, title: str = "Model Output", graph_conf :dict=None) -> go.Figure:
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

    # Metrics
    rmse = mean_squared_error(y_true=raw_data, y_pred=model_result.Z, squared=False)
    r2 = r2_score(y_true=raw_data, y_pred=model_result.Z)

    if abs(rmse) < 100:
        rmse = f"{rmse:.4f}"
    else:
        rmse = f"{rmse:.4e}"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"RMS Error: {rmse}",
            "Residuals",
            f"R<sup>2</sup>: {r2:.4f}",
            "Partial Autocorrelation"
        )
        # specs=[[{"type": "scatter"}, {"type": "scatter"}],
        #        [{"type": "scatter"}, {"type": "[scatter, bar]]"}]]
    )

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

    # Plotting Autocorrelation
    fig.add_trace(
        go.Scatter(
            x=model_result.Z,
            y=raw_data,
            name=f"Autocorrelation",
            mode='markers',
            marker=dict(color='royalblue', opacity=0.5)
        ),
        row=2, col=1
    )

    # Plotting Autocorrelation Function
    for trace in corr_plotly(raw_data - model_result.Z, lags=50, fnc_to_plot=['PACF']).data:
        fig.add_trace(trace, row=2, col=2)


    fig.update_layout(
        title=title,
        showlegend=False,
        width=1200,
        height=800,
        xaxis_title="Time",
        yaxis_title="Temperature [°C]",
        yaxis2_title="Temperature [°C]",
        xaxis3_title="Ti Modelled [°C]",
        yaxis3_title="Ti Measured [°C]",
    )

    if graph_conf is not None:
        fig.update_layout(
            font=dict(
                family=graph_conf['font_family'],
                size=graph_conf['font_size'],
            ),
            template=graph_conf['theme'],
            # width=graph_conf['width'],
            # height=graph_conf['height'],
        )

    return fig

def corr_plotly(data :pd.Series, lags :int=50, fnc_to_plot :list=['PACF', 'ACF']) -> go.Figure:
    """Creates a plotly figure with the autocorrelation and partial autocorrelation of a time series.

    Parameters
    ----------
    data : `pd.Series`
        Time series to be analyzed.
    lags : `int`, optional
        Number of lags to be plotted. The default is 50.

    Returns
    -------
    fig : `go.Figure`
        Plotly figure with the autocorrelation and partial autocorrelation of the time series.
    """

    # Internal Variables
    cols = 1
    rows = len(fnc_to_plot)

    def _get_fig(data :pd.Series, name :str) -> go.Figure:
        """Creates a plotly figure with the time series.

        Parameters
        ----------
        data : `pd.Series`
            Time series to be analyzed.

        Returns
        -------
        fig : `go.Figure`
            Plotly figure with the time series.
        """
        point = go.Scatter(x=np.arange(len(data)), y=data, mode='markers', marker_color='#1f77b4', marker_size=8, name=name)       # Points
        lines = [go.Scatter(x=(x,x), y=(0,data[x]), mode='lines',line_color='#1f77b4') for x in range(len(data))]  # Lines
        return [point, *lines]

    fig = make_subplots(rows=rows, cols=cols)

    for fnc in fnc_to_plot:
        if fnc == 'PACF':
            pacf_values = pacf(data, nlags=lags)
            fig_partial = _get_fig(pacf_values, name='PACF')
            for trace in fig_partial:
                fig.add_trace(trace, row=1, col=1)
        elif fnc == 'ACF':
            acf_values = acf(data, nlags=lags)
            fig_autocorr = _get_fig(acf_values, name='ACF')
            for trace in fig_autocorr:
                fig.add_trace(trace, row=2, col=1)
        else:
            raise ValueError(f'fnc_to_plot must be a list containing only "PACF" or "ACF"')


    fig.update_layout(showlegend=False)

    return fig

if __name__ == "__main__":
    pass