# AutoTS Agent Overview

## General Code Guidelines

### Small Files & Single Responsibility
- **Aim to keep files < 500 lines.** Large files are difficult to reason about and cause merge conflicts.
- **Single Responsibility per File:** Each module should have one reason to change.

### Hyper-Descriptive Naming
- **Favor Explicit Over Concise:** Use long, descriptive names that explain intent

### High-Signal Comments
- **Explain "Why", Not "What":** Comments should explain the reasoning behind complex algorithms or architectural decisions.
- **Be Token-Efficient in Comments:** Use concise, informative language. Focus on documenting interface contracts and capability tiers.

### Testing & Benchmarking
- **Shared Fixtures:** Extract common test utilities (synthetic signal generators, stub storage) into a shared helper module if reused across 3+ files.

### Dependencies
- **Limited Assumed Dependencies** Only pandas, numpy are assumed always available. scikit-learn, scipy, and statsmodels are usually available but should still be in a guarded import block (with fallbacks in mocks.py). Then other libraries are in a separate guarded import as well. Python 3.9 compatability is to be maintained where possible.

## AutoTS Specific Guidelines

### Data
- **Wide Style Data** All internal objects use `wide` style time series data dataframes directly whenever possible, with custom data classes used sparingly. In wide style data each column is a unique time series and each row has a datetime index.
- **Assumptions on Data** Series will largely be consistent in period, or at least up-sampled to regular intervals. The most recent data will generally be the most important.

### Core objects in AutoTS
- **Transformers** Transformers are preprocessing or postprocessing of wide style data. They are most commonly used as a transform before a forecast on historical data, then inverse_transform after the forecast on the forecast data (which includes new future dates) back to the original space. They sometimes increase or decrease the number of columns or rows of the data (expanding transformers).
- **Models** Models are generally machine learning algorithms that take in historical data (fit) and output future values (predict) as a forecast based upon their algorithm. Both Models and Transformers have `get_new_params` which generate new random parameter values for the genetic optimizer. get_new_params is weighted towards the most consistently accurate and fastest parameter options. Ensembles are a special model type that combine multiple models, and AutoTS has some unique state of the art ensemble methods like mosaic ensembles.
- **Detectors** Detectors such as AnomalyDetector or HolidayDetector identify features of time series data. Most of them are used together in the TimeSeriesFeatureDetector (note code changes should be upstreamed to the more specific detector class where possible). Detectors often can serve as transformers or share library code with a paired transformer.
- **PredictionObject** This object is the class designed for storage of forecasts output from all models. It includes various features to adjust or plot the resulting forecast.
- **AutoTS** the AutoTS class itself is the primary entry point for users to get and iteract with forecasts. While advanced users are expected to utilize the rest of the library code directly, documentation and code style generally focus on the AutoTS class in particular. AutoTS's core function is to run a search across validation holdouts of combinations of transformers and model parameters, usually several combined in an ensemble, returning a prediction object at the end of a tuned forecast.

### Basic Tenants
- **Priorities** Accuracy > Ease of Use > Speed (with speed more important with 'fast' selections). While speed is lowest priority, it is still critical, and all code should be vectorized and use matrix operations, avoiding for loops when possible.
- **Fault tolerance** it is perfectly acceptable for model parameters to fail on some datasets, the higher level API will pass over and use others. However, additional fallbacks should be avoided where possible, and root cause solutions found.
- **Missing data tolerance** large chunks of data can be missing and model will still produce reasonable results (although lower quality than if data is available)
