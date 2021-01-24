AutoTS
=======

`autots`_ is an automated time series forecasting package for Python.

Features:

- Finds optimal time series forecasting model and data transformations by genetic programming optimization
- Handles univariate and multivariate/parallel time series
- Point and probabilistic upper/lower bound forecasts for all models
- Over twenty available model classes, with tens of thousands of possible hyperparameter configurations
	- Includes naive, statistical, machine learning, and deep learning models
	- Multiprocessing for univariate models for scalability on multivariate datasets
	- Ability to add external regressors
- Over thirty time series specific data transformations
	- Ability to handle messy data by learning optimal NaN imputation and outlier removal
- Allows automatic ensembling of best models
	- 'horizontal' ensembling on multivariate series - learning the best model for each series
- Multiple cross validation options
	- 'seasonal' validation allows forecasts to be optimized for the season of your forecast period
- Subsetting and weighting to improve speed and relevance of search on large datasets
	- 'constraint' parameter can be used to assure forecasts don't drift beyond historic boundaries
- Option to use one or a combination of metrics for model selection
- Import and export of model templates for deployment and greater user customization


Installation
------------

.. code:: sh

   pip install autots


Requirements: Python 3.6+, numpy, pandas, statsmodels, and scikit-learn.

Getting Started
===================
.. toctree::
   :maxdepth: 2

   source/intro
   source/tutorial
   
Modules API
===================
.. toctree::
   :maxdepth: 2

   source/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
