#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""
Training Prohet Model Model
"""

from __future__ import annotations

__author__ = "Carlos Alejandro Perez Garcia"
__copyright__ = "Copyright 2023"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Carlos Alejandro Perez Garcia"
__email__ = "cpgarcia518@gmail.com"

# Libraries
# ==============================================================================
from copy import deepcopy

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import lmfit
import numpy as np

@dataclass
class GreyModelResult:
    '''
    Dataclass that holds the results of the fitting of the model to a data set.
        yhat: The measured variable's fit / predicted values
        input: The input values used to fit the model
        var: The variables of the model including internal ones if any
        params: The parameters of the model
    '''
    yhat: np.ndarray
    input: Dict
    params: lmfit.Parameters
    var: Dict

class GreyModel:
    """Abstract Base Class for Grey Model"""

    def __init__(self, params: Union[lmfit.Parameters, Dict], rec_duration: float):
        """
        Initialize the model with the model function and the parameters

        Parameters
        ----------
        params : Union[lmfit.Parameters, Dict]
            A dictionary of parameters for the fitting. Key - value pairs should follow the
            `lmfit.Parameters` declaration:
            e.g. {'A' : {'value': 10, 'min': 0, 'max': 30}} - sets the initial value and the bounds
            for parameter `A`
        rec_duration : float
            The duration of the recording in seconds
        """

        self.result = lmfit.minimizer.MinimizerResult()

        # convert the params dict into lmfit parameters
        if isinstance(params, lmfit.Parameters):
            self.params = deepcopy(params)
        else:
            self.params = lmfit.Parameters()
            for k, v in params.items():
                self.params.add(k, **v)

        # set the number of records based on the measured variable's values
        self.rec_duration = rec_duration

    def fit(
        self,
        input: Dict,
        yhat: np.ndarray,
        method: str = "leastsq",
        ic_params: Optional[Dict] = None,
        objective_func: Optional[Callable] = None,
    ):
        """
        Fit the model by minimising the objective function value

        Parameters
        ----------
        input : dict
            A dictionary of input values for the fitting - these values are fixed during the fit.
        yhat : np.ndarray
            The measured variable's values to fit the model to.
        method : str, optional
            The method to use for the minimisation, by default "leastsq"
        ic_params : Optional[Dict], optional
            Initial conditions for the parameters, by default None
        objective_func : Optional[Callable], optional
            The objective function that is passed to `lmfit.minimize`, by default None

        Returns
        -------
        `lmfit.minimizer.MinimizerResult`
            Object containing the optimized parameters and several goodness-of-fit statistics.
        """

        # Overwrite the initial conditions if provided
        if ic_params:
            for k, v in ic_params.items():
                if k in self.params:
                    self.params[k].value = v
                else:
                    raise ValueError(f"Parameter {k} not found in model parameters.")
                    # logger.warning(f'Key `{k}` not found in initial conditions params')

        self.result = lmfit.minimize(
            objective_func or self.objective_func,
            self.params,
            args=(input, yhat),
            method=method,
            kws={'model': self.model, 'input': input, 'yhat': yhat},
        )

        self.params = self.result.params

        return self

    def predict(self, input: Dict, ic_params: Optional[Dict] = None) -> GreyModelResult:
        """
        Generates a prediction based on the result parameters and input.

        Parameters
        ----------
        input : Dict
            A dictionary of input values for the fitting - these values are fixed during the fit.
        ic_params : Optional[Dict], optional
            Initial conditions for the parameters, by default None
        """

        if ic_params:
            for k, v in ic_params.items():
                if k in self.params:
                    self.params[k].value = v
                else:
                    raise ValueError(f"Parameter {k} not found in model parameters.")
                    # logger.warning(f'Key `{k}` not found in initial conditions params')

        return self.model(self.params, input)

    def model(self, params: lmfit.Parameters, input: Dict) -> GreyModelResult:
        """
        The model function that is used to generate a prediction.

        Parameters
        ----------
        params : lmfit.Parameters
            The parameters of the model
        input : Dict
            A dictionary of input values for the fitting - these values are fixed during the fit.
        """

        def lock(self):
            """
            Lock the parameters of the model
            """

            for param in self.params.keys():
                self.params[param].vary = False

            return self

        # raise NotImplementedError

    @staticmethod
    def objective_func(params: lmfit.Parameters, *args, **kwargs):
        """
        The objective function that is used to fit the model to the data.

        Parameters
        ----------
        params : lmfit.Parameters
            The parameters of the model
        *args
            Arguments passed to the objective function
        **kwargs
            Keyword arguments passed to the objective function
        """

        return (kwargs['model'](params=params, input=kwargs['input']).yhat - kwargs['yhat']).ravel()


