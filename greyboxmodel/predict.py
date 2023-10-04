#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
""" Train the grey box model """

from __future__ import annotations

__author__ = "Carlos Alejandro Perez Garcia"
__copyright__ = "Copyright 2023"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Carlos Alejandro Perez Garcia"
__email__ = "cpgarcia518@gmail.com"

# Standard libraries
# ==============================================================================
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Union, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Own libraries
# ==============================================================================
# from darkgreybox import logger
from greyboxmodel.base_model import GreyModel, GreyModelResult

def predict_models(
    # models: List[GreyModel],
    models: List[Union[GreyModel, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ic_params_map: Dict,
    error_metric: Callable,
    train_results: List[GreyModelResult],
    n_jobs: int = -1,
    verbose: int = 10
) -> pd.DataFrame:
    """
    Predicts the `models` for the given `X_test` and `y_test` testing data.

    Parameters
    ----------
    models: List[GreyModel]
        The list of models to predict.
    X_test: pd.DataFrame
        The testing data.
    y_test: pd.Series
        The testing target.
    ic_params_map: Dict
        A dictionary of mapping functions that return the initial condition parameters.
    error_metric: Callable
        An error metric function that confirms to the `sklearn.metrics` interface
        and returns a single value.
    train_results: List[GreyModelResult]
        The model results of the previously trained models.
    n_jobs: int
        The number of jobs to run in parallel. -1 means using all processors.
    verbose: int
        The degree of verbosity as described by `joblib.Parallel`.

    Returns
    -------
    pd.DataFrame
        A dataframe with the predictions and errors for each model.

    Examples
    --------
    ```python
    from sklearn.metrics import mean_squared_error

    from darkgreybox.fit import test_models


    prefit_df = train_models(
        models=[trained_model_1, trained_model_2],
        X_test=X_test,
        y_test=y_test,
        ic_params_map={}
        error_metric=mean_squared_error,
        train_results=[trained_model_result_1, trained_model_result_2],
        n_jobs=-1,
        verbose=10
    )
    ```
    """

    num_models = len(models)
    # logger.info(f'Generating predictions for {num_models} models...')

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as p:
            df = cast(pd. DataFrame, pd.concat(
                cast(pd. DataFrame, p(delayed(predict_model)(
                    model, X_test, y_test, ic_params_map, error_metric, train_result)
                    for model, train_result in zip(models, train_results)
                )),
                ignore_index=True
            ))

    else:
        df = pd.concat([predict_model(model, X_test, y_test, ic_params_map, error_metric, train_result)
                        for model, train_result in zip(models, train_results)], ignore_index=True)

    return df

def predict_model(
    model: Union[GreyModel, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ic_params_map: Dict,
    error_metric: Callable,
    train_result: GreyModelResult
) -> pd.DataFrame:
    """
    Predicts the `model` for the given `X_test` and `y_test` testing data.

    Parameters
    ----------
    model: GreyModel
        The model to predict.
    X_test: pd.DataFrame
        The testing data.
    y_test: pd.Series
        The testing target.
    ic_params_map: Dict
        A dictionary of mapping functions that return the initial condition parameters.
    error_metric: Callable
        An error metric function that confirms to the `sklearn.metrics` interface
        and returns a single value.
    train_result: GreyModelResult
        The model result of the previously trained model.

    Returns
    -------
    pd.DataFrame
        A dataframe with the predictions and errors for the model.
    """

    start = timer()
    start_date = X_test.index[0]
    end_date = X_test.index[-1]

    X = X_test.to_dict(orient='list')
    y = y_test.values

    if isinstance(model, GreyModel):
        ic_params = map_ic_params(ic_params_map, model, X_test, y_test, train_result)
        model_result = model.predict(X, ic_params)
        # y_pred = model.predict(X, y, ic_params_map)
        end = timer()

        return pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date],
            # 'model_name': [model.name],
            'model': [model],
            'model_result': [model_result],
            'time': [end - start],
            'error': [error_metric(y, model_result.y_pred)],
        })

    else:
        end = timer()
        return pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date],
            # 'model_name': [model.name],
            'model': [None],
            'model_result': [None],
            'time': [end - start],
            'error': [None],
        })

def map_ic_params(
    ic_params_map: Dict,
    model: GreyModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_result: GreyModelResult
) -> Dict:
    """
    Maps the initial condition parameters for the given `model`.

    Parameters
    ----------
    ic_params_map: Dict
        A dictionary of mapping functions that return the initial condition parameters.
    model: GreyModel
        The model to predict.
    X_test: pd.DataFrame
        The testing data.
    y_test: pd.Series
        The testing target.
    train_result: GreyModelResult
        The model result of the previously trained model.

    Returns
    -------
    Dict
        A dictionary with the initial condition parameters.

    ```python
    # Assuming y_test holds the internal temperatures `Ti`

    ic_params_map = {
        'Ti0': lambda X_test, y_test, train_result: y_test.iloc[0],
        'Th0': lambda X_test, y_test, train_result: y_test.iloc[0],
        'Te0': lambda X_test, y_test, train_result: train_result.Te[-1],
    }

    will map the first internal temperature in the test set to both `Ti0` and `Th0`
    and the last `Te` value from the training results to `Te0`.
    ```
    """

    ic_params = {}
    # TODO: Suggestion by Copilot
    # for param, func in ic_params_map.items():
    #     ic_params[param] = func(X_test, y_test, train_result)

    for key in ic_params_map:
        if key in model.params:
            ic_params[key] = ic_params_map[key](X_test, y_test, train_result)

    return ic_params


if __name__ == "__main__":
    pass