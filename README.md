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


preordained_regressor
regressor_type = [None, 'User', 'Holiday']