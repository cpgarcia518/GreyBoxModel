#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""Grey Box Mdoel Definition"""

from __future__ import annotations
from typing import Dict

__author__ = "Carlos Alejandro Perez Garcia"
__copyright__ = "Copyright 2022"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Carlos Alejandro Perez Garcia"
__email__ = "cpgarcia518@gmail.com"

# Libraries
# ==============================================================================
import numpy as np
import lmfit

from greyboxmodel.base_model import GreyModel, GreyModelResult

class Ti(GreyModel):
    """A DarkGrey Model representing a Ti RC-equivalent circuit

    Notes
    ----------
    Assign internal temperature as the measured variable to be fitted to the model
    y = df['Internal Temperature [˚C]'].values

    Inputs = {
        'Ph': df['Heater Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
    }

    Parameters to be fitted
    'value' - Initial Value
    'min' & 'max' - Bounds
    'vary' - True or False. If false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Ci': {'value': 132},
        'Ria': {'value': 1},
    }

    # Fit using the Nelder-Mead method
    model = Ti(params, rec_duration=1).fit(X, y, method='nelder')
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ic_param_names = ['Ti0']
        self.rc_param_names = ['Ci', 'Ria']
        self.input_param_names = ['Ta', 'Ph']

    def model(self, params: lmfit.Parameters, input: Dict) -> GreyModelResult:
        """Model function

        Parameters
        ----------
        params : lmfit.Parameters
            Parameters to be fitted
        input : Dict
            Inputs to the model

        Returns
        -------
        GreyModelResult
            Model result
        """

        num_rec = len(input['Ta'])
        Ti = np.zeros(num_rec)

        # Alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0'].value
        Ci = params['Ci'].value
        Ria = params['Ria'].value

        Ta = input['Ta']
        Ph = input['Ph']

        # Step through the rest of the time points
        for i in range(1, num_rec):
            # Ti[i] = Ti[i - 1] + (1 / Ci) * (Ph[i - 1] - (Ti[i - 1] - Ta[i - 1]) / Ria)
            dTi = ((Ta[i - 1] - Ti[i - 1]) / (Ria * Ci) + (Ph[i - 1]) / (Ci)) * self.rec_duration
            Ti[i] = Ti[i - 1] + dTi

        return GreyModelResult(Ti, input, params, {'Ti': Ti})

if __name__ == "__main__":
    pass