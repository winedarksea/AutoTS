# AutoTS
### Project CATS (Catlin Automated Time Series)
Model Selection for Multiple Time Series

Simple package for comparing open-source time series implementations.
For other time series needs, check out the package list here: https://github.com/MaxBenChrist/awesome_time_series_in_python

Requirements:
	Python >= 3.5 (typing)
	pandas
	sklearn >= 0.20.0 (ColumnTransformer)
	statsmodels
	holidays


pip install fredapi # if using samples
conda install -c conda-forge fbprophet
pip install mxnet==1.4.1
    pip install mxnet-cu90mkl==1.4.1 # if you want GPU and have Intel CPU
pip install gluonts==0.4.0
    pip install git+https://github.com/awslabs/gluon-ts.git #if you want dev version
pip install pmdarima==1.4.0 
pip uninstall numpy # might be necessary, even twice, followed by the following
pip install numpy==1.17.4 # gluonts likes to force numpy back to 1.14, but 1.17 seems to be fine with it
pip install sktime==0.3.1


## To-Do
Speed improvements, Parallelization, and Distributed options for general greater speed
Generate list of functional frequences, and improve usability on rarer frequenices
Warning/handling if lots of NaN in most recent (test) part of data
Figures: Add option to output figures of train/test + forecast, other performance figures
Input and Output saved templates as .csv and .json
'Check Package' to check if optional model packages are installed
Pre-clustering on many time series
If all input are Int, convert floats back to int
Trim whitespace on string inputs
Hierachial correction (bottom-up to start with)
Improved verbosity controls and options
Export as simpler code (as TPOT)
AIC metric
Analyze and return inaccuracy patterns
Used saved results to resume a search partway through
Generally improved probabilistic forecasting
Option to drop series which haven't had a value in last N days

#### Ensemble:
	best 3
	forecast distance 20/80
	forecast distance 30/30/30
	best per series ensemble
	best point with best probalistic containment
#### New models:
	Seasonal Naive
	Last Value + Drift Naive
	Simple Decomposition forecasting
	GluonTS Models
	Simulations
	Sklearn + TSFresh
	Sklearn + polynomial features
	Sktime
	Ta-lib
	tslearn
	pydlm
	Isotonic regression
	TPOT if it adds multioutput functionality
	

