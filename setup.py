#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="models",
    # packages=find_packages(),
    packages=['greyboxmodel'],
    dependencies=[
        'numpy','pandas','scipy','statsmodels','sklearn'
    ]
    version="1.0.0",
    description="A Python package for Grey Box Modelling",
    author="Carlos Alejandro Perez Garcia",
    author_email='cpgarcia518@gmail.com',
    url='https://github.com/cpgarcia518/GreyBoxModel',
    license="MIT",
)