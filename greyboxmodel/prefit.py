#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
""" Add Information about the module here"""

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

from typing import Any, Callable, List, Optional, cast

# Own libraries
# ==============================================================================
from greyboxmodel.train import train_models

def prefit_models(
    # models: Dict[str, GreyModel],
    models: List[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    error_metric: Callable,
    prefit_splits: Optional[List] = None,
    prefit_filter: Optional[Callable] = None,
    method: str = 'nelder',
    obj_func: Optional[Callable] = None,
    n_jobs: int = -1,
    verbose: int = 10
) -> List[Any]:
    """
    Trains the `models` for the given `X_train` and `y_train` training data for `splits` using `method`.
    """
    if prefit_splits is None:
        # prefit_splits = [X_train.index.tolist()]
        return models

    prefit_df = train_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        splits=prefit_splits,
        error_metric=error_metric,
        method=method,
        obj_func=obj_func,
        n_jobs=n_jobs,
        verbose=verbose
    )

    filtered_df = apply_prefit_filter(prefit_df, prefit_filter)

    if 'model' not in filtered_df or len(filtered_df.dropna(subset=['model'])) == 0:
        raise ValueError('No valid models found during prefit')

    return filtered_df['model'].tolist()

def apply_prefit_filter(
    prefit_df: pd.DataFrame,
    prefit_filter: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Applies the prefit filter to the prefit dataframe.
    """
    if prefit_filter is None:
        return prefit_df
    else:
        return cast(pd.DataFrame, prefit_df[prefit_filter(prefit_df['error'])].reset_index(drop=True))

if __name__ == "__main__":
    pass