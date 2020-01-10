import numpy as np
import pandas as pd
import datetime
import copy
import json

class AutoTS(object):
    """"
    Automated time series modeling using genetic algorithms.
    
    Args:
        forecast_length: int = 14
        weighted: bool = False
        frequency: str = 'infer'
        aggfunc: str = 'first'
        prediction_interval: float = 0.9
        no_negatives: bool = False
        random_seed: int = 425
        holiday_country: str = 'US'
        ensemble: bool = True
        subset: int = 200
        na_tolerance: float = 0.95
        metric_weighting: dict = {'smape_weighting' : 10, 'mae_weighting' : 1,
            'rmse_weighting' : 5, 'containment_weighting' : 1, 'runtime_weighting' : 0}
        drop_most_recent: int = 1
        num_validations: int = 2
        models_to_validate: int = 10
        validation_method (str): 'even' or 'backwards' where backwards is better for shorter training sets
        max_generations: int = 1
        verbose: int = 1
    """
    def __init__(self, 
        forecast_length: int = 14,
        weighted: bool = False,
        frequency: str = 'infer',
        aggfunc: str = 'first',
        prediction_interval: float = 0.9,
        no_negatives: bool = False,
        random_seed: int = 425,
        holiday_country: str = 'US',
        ensemble: bool = True,
        subset: int = 200,
        na_tolerance: float = 0.95,
        metric_weighting: dict = {'smape_weighting' : 10, 'mae_weighting' : 1,
            'rmse_weighting' : 5, 'containment_weighting' : 1, 'runtime_weighting' : 0},
        drop_most_recent: int = 0,
        num_validations: int = 3,
        models_to_validate: int = 10,
        validation_method: str = 'even',
        max_generations: int = 5,
        verbose: int = 1 
        ):
        self.weighted = weighted
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.aggfunc = aggfunc
        self.prediction_interval = prediction_interval
        self.no_negatives = no_negatives
        self.random_seed = random_seed
        self.holiday_country = holiday_country
        self.ensemble = ensemble
        self.subset = subset
        self.na_tolerance = na_tolerance
        self.metric_weighting = metric_weighting
        self.drop_most_recent = drop_most_recent
        self.num_validations = num_validations
        self.models_to_validate = models_to_validate
        self.validation_method = validation_method
        self.max_generations = max_generations
        self.verbose = verbose
        
        self.best_model = pd.DataFrame()
        self.regressor_used = False
        self.template_cols = ['Model','ModelParameters','TransformationParameters','Ensemble']
        
    def fit(self, df, preord_regressor = [], weights: dict = {}):
        self.preord_regressor_train = preord_regressor
        self.weights = weights
        
        forecast_length = self.forecast_length
        weighted = self.weighted
        frequency = self.frequency
        aggfunc = self.aggfunc
        prediction_interval = self.prediction_interval
        no_negatives = self.no_negatives
        random_seed = self.random_seed
        holiday_country = self.holiday_country
        ensemble = self.ensemble
        subset = self.subset
        na_tolerance = self.na_tolerance
        metric_weighting = self.metric_weighting
        drop_most_recent = self.drop_most_recent
        num_validations = self.num_validations
        models_to_validate = self.models_to_validate
        validation_method = self.validation_method
        max_generations = self.max_generations
        verbose = self.verbose
        
        random_seed = abs(int(random_seed))
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        template_cols = self.template_cols
        
        from autots.tools.shaping import long_to_wide
        
        df_wide = long_to_wide(df, date_col = 'date', value_col = 'value',
                               id_col = 'series_id', frequency = frequency, na_tolerance = na_tolerance,
                               drop_data_older_than_periods = 1000, aggfunc = aggfunc,
                               drop_most_recent = drop_most_recent)
        
        if weighted == False:
            weights = {x:1 for x in df_wide.columns}
        if weighted == True:
            # handle not all weights being provided
            weights = {col:(weights[col] if col in weights else 1) for col in df_wide.columns}
            # handle non-numeric inputs
            weights = {key:(abs(float(weights[key])) if str(weights[key]).isdigit() else 1) for key in weights}
            
        preord_regressor = pd.Series(np.random.randint(0, 100, size = len(df_wide.index)), index = df_wide.index )
        
        
        
        from autots.tools.shaping import values_to_numeric
        categorical_transformer = values_to_numeric(df_wide)
        self.categorical_transformer = categorical_transformer
        
        df_wide_numeric = categorical_transformer.dataframe
        self.df_wide_numeric = df_wide_numeric
        
        from autots.tools.profile import data_profile
        profile_df = data_profile(df_wide_numeric)
        self.startTimeStamps = profile_df.loc['FirstDate']
        
        from autots.tools.shaping import subset_series
        df_subset = subset_series(df_wide_numeric, list((weights.get(i)) for i in df_wide_numeric.columns), n = subset, na_tolerance = na_tolerance, random_state = random_seed)
        
        if weighted == False:
            current_weights = {x:1 for x in df_subset.columns}
        if weighted == True:
            current_weights = {x: weights[x] for x in df_subset.columns}
            
        from autots.tools.shaping import simple_train_test_split
        df_train, df_test = simple_train_test_split(df_subset, forecast_length = forecast_length)
        preord_regressor_train = preord_regressor[df_train.index]
        preord_regressor_test = preord_regressor[df_test.index]
        
        from autots.evaluator.auto_model import TemplateEvalObject
        main_results = TemplateEvalObject()
        
        from autots.evaluator.auto_model import NewGeneticTemplate
        from autots.evaluator.auto_model import RandomTemplate
        from autots.evaluator.auto_model import TemplateWizard    
        from autots.evaluator.auto_model import unpack_ensemble_models
        from autots.evaluator.auto_model import  generate_score
        model_count = 0
        initial_template = RandomTemplate(40)
        initial_template = unpack_ensemble_models(initial_template, template_cols, keep_ensemble = False)
        submitted_parameters = initial_template.copy()
        template_result = TemplateWizard(initial_template, df_train, df_test, current_weights,
                                         model_count = model_count, ensemble = ensemble, 
                                         forecast_length = forecast_length, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_test, 
                                          holiday_country = holiday_country,
                                          startTimeStamps = self.startTimeStamps,
                                          template_cols = template_cols, random_seed = random_seed, verbose = verbose)
        model_count = template_result.model_count
        main_results.model_results = pd.concat([main_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
        main_results.model_results['Score'] = generate_score(main_results.model_results, metric_weighting = metric_weighting,prediction_interval = prediction_interval)
        main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
        main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
        main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
        main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
        if ensemble:
            main_results.forecasts_list.extend(template_result.forecasts_list)
            main_results.forecasts_runtime.extend(template_result.forecasts_runtime)
            main_results.forecasts.extend(template_result.forecasts)
            main_results.upper_forecasts.extend(template_result.upper_forecasts)
            main_results.lower_forecasts.extend(template_result.lower_forecasts)
        
        
        current_generation = 0
        # eventually, have this break if accuracy improvement plateaus before max_generations
        while current_generation < max_generations:
            current_generation += 1
            if verbose > 0:
                print("New generation: {}".format(current_generation))
            new_template = NewGeneticTemplate(main_results.model_results, submitted_parameters=submitted_parameters, sort_column = "Score", 
                               sort_ascending = True, max_results = 40, top_n = 15, template_cols=template_cols)
            submitted_parameters = pd.concat([submitted_parameters, new_template], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            
            template_result = TemplateWizard(new_template, df_train, df_test, current_weights,
                                         model_count = model_count, ensemble = ensemble, 
                                         forecast_length = forecast_length, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_test, 
                                          holiday_country = holiday_country,
                                          startTimeStamps = profile_df.loc['FirstDate'],
                                          template_cols = template_cols,
                                          random_seed = random_seed, verbose = verbose)
            model_count = template_result.model_count
            main_results.model_results = pd.concat([main_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            main_results.model_results['Score'] = generate_score(main_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
            
            main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
            main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
            main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
            main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
            if ensemble:
                main_results.forecasts_list.extend(template_result.forecasts_list)
                main_results.forecasts_runtime.extend(template_result.forecasts_runtime)
                main_results.forecasts.extend(template_result.forecasts)
                main_results.upper_forecasts.extend(template_result.upper_forecasts)
                main_results.lower_forecasts.extend(template_result.lower_forecasts)
        
        
        
        
        if ensemble:
            ensemble_forecasts_list = []
            
            best3 = main_results.model_results[main_results.model_results['Ensemble'] == 0].nsmallest(3, columns = ['Score'])
            ensemble_models = {}
            for index, row in best3.iterrows():
                temp_dict = {'Model': row['Model'],
                 'ModelParameters': row['ModelParameters'],
                 'TransformationParameters': row['TransformationParameters']
                 }
                ensemble_models[row['ID']] = temp_dict
            best3params = {'models': ensemble_models}    
            
            from autots.models.ensemble import EnsembleForecast
            best3_ens_forecast = EnsembleForecast("Best3Ensemble", best3params, main_results.forecasts_list, main_results.forecasts, main_results.lower_forecasts, main_results.upper_forecasts, main_results.forecasts_runtime, prediction_interval)
            ensemble_forecasts_list.append(best3_ens_forecast)
            
            first_bit = int(np.ceil(forecast_length * 0.2))
            last_bit = int(np.floor(forecast_length * 0.8))
            ens_per_ts = main_results.model_results_per_timestamp_smape[main_results.model_results_per_timestamp_smape.index.isin(main_results.model_results[main_results.model_results['Ensemble'] == 0]['ID'].tolist())]
            first_model = ens_per_ts.iloc[:,0:first_bit].mean(axis = 1).idxmin()
            last_model = ens_per_ts.iloc[:,first_bit:(last_bit + first_bit)].mean(axis = 1).idxmin()
            ensemble_models = {}
            best3 = main_results.model_results[main_results.model_results['ID'].isin([first_model,last_model])].drop_duplicates(subset = ['Model','ModelParameters','TransformationParameters'])
            for index, row in best3.iterrows():
                temp_dict = {'Model': row['Model'],
                 'ModelParameters': row['ModelParameters'],
                 'TransformationParameters': row['TransformationParameters']
                 }
                ensemble_models[row['ID']] = temp_dict
            dist2080params = {'models': ensemble_models,
                              'FirstModel':first_model,
                              'LastModel':last_model} 
            dist2080_ens_forecast = EnsembleForecast("Dist2080Ensemble", dist2080params, main_results.forecasts_list, main_results.forecasts, main_results.lower_forecasts, main_results.upper_forecasts, main_results.forecasts_runtime, prediction_interval)
            ensemble_forecasts_list.append(dist2080_ens_forecast)
        
            from autots.models.ensemble import EnsembleEvaluate
            ens_template_result = EnsembleEvaluate(ensemble_forecasts_list, df_test = df_test, weights = current_weights, model_count = model_count)
            
            model_count = ens_template_result.model_count
            main_results.model_results = pd.concat([main_results.model_results, ens_template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            main_results.model_results['Score'] = generate_score(main_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
            main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(ens_template_result.model_results_per_timestamp_smape)
            main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(ens_template_result.model_results_per_timestamp_mae)
            main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(ens_template_result.model_results_per_series_smape)
            main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(ens_template_result.model_results_per_series_mae)
        
        
        num_validations = abs(int(num_validations))
        max_possible = int(np.floor(len(df_wide_numeric.index)/forecast_length))
        if max_possible < (num_validations + 1):
            num_validations = max_possible - 1
            if num_validations < 0:
                num_validations = 0
            print("Too many training validations for length of data provided, decreasing num_validations to {}".format(num_validations))
        
        validation_template = main_results.model_results.sort_values(by = "Score", ascending = True, na_position = 'last').drop_duplicates(subset = template_cols).head(models_to_validate)[template_cols]
        if not ensemble:
            validation_template[validation_template['Ensemble'] == 0]
            
        validation_results = copy.copy(main_results) 
        
        from autots.evaluator.auto_model import validation_aggregation
        if num_validations > 0:
            if validation_method == 'backwards':
                for y in range(num_validations):
                    # gradually remove the end
                    current_slice = df_wide_numeric.head(len(df_wide_numeric.index) - (y+1) * forecast_length)
                    # subset series (if used) and take a new train/test split
                    df_subset = subset_series(current_slice, list((weights.get(i)) for i in df_wide_numeric.columns), n = subset, na_tolerance = na_tolerance, random_state = random_seed)
                    if weighted == False:
                        current_weights = {x:1 for x in df_subset.columns}
                    if weighted == True:
                        current_weights = {x: weights[x] for x in df_subset.columns}                
                    df_train, df_test = simple_train_test_split(df_subset, forecast_length = forecast_length)
                    preord_regressor_train = preord_regressor[df_train.index]
                    preord_regressor_test = preord_regressor[df_test.index]
        
                    template_result = TemplateWizard(validation_template, df_train, df_test, current_weights,
                                                 model_count = model_count, ensemble = ensemble, 
                                                 forecast_length = forecast_length, frequency=frequency, 
                                                  prediction_interval=prediction_interval, 
                                                  no_negatives=no_negatives,
                                                  preord_regressor_train = preord_regressor_train,
                                                  preord_regressor_forecast = preord_regressor_test, 
                                                  holiday_country = holiday_country,
                                                  startTimeStamps = profile_df.loc['FirstDate'],
                                                  template_cols = template_cols, random_seed = random_seed, verbose = verbose)
                    model_count = template_result.model_count
                    validation_results.model_results = pd.concat([validation_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
                    validation_results.model_results['Score'] = generate_score(validation_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
                    validation_results.model_results_per_timestamp_smape = validation_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
                    validation_results.model_results_per_timestamp_mae = validation_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
                    validation_results.model_results_per_series_smape = validation_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
                    validation_results.model_results_per_series_mae = validation_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
                validation_results = validation_aggregation(validation_results)
        
            if validation_method == 'even':
                for y in range(num_validations):
                    # /num_validations biases it towards the last segment (which I prefer), /(num_validations + 1) would remove that
                    validation_size = int(np.floor((len(df_wide_numeric.index) - forecast_length)/num_validations))
                    current_slice = df_wide_numeric.head(validation_size * (y+1) + forecast_length)
                    # subset series (if used) and take a new train/test split
                    df_subset = subset_series(current_slice, list((weights.get(i)) for i in df_wide_numeric.columns), n = subset, na_tolerance = na_tolerance, random_state = random_seed)
                    if weighted == False:
                        current_weights = {x:1 for x in df_subset.columns}
                    if weighted == True:
                        current_weights = {x: weights[x] for x in df_subset.columns}                
                    df_train, df_test = simple_train_test_split(df_subset, forecast_length = forecast_length)
                    preord_regressor_train = preord_regressor[df_train.index]
                    preord_regressor_test = preord_regressor[df_test.index]
        
                    template_result = TemplateWizard(validation_template, df_train, df_test, current_weights,
                                                 model_count = model_count, ensemble = ensemble, 
                                                 forecast_length = forecast_length, frequency=frequency, 
                                                  prediction_interval=prediction_interval, 
                                                  no_negatives=no_negatives,
                                                  preord_regressor_train = preord_regressor_train,
                                                  preord_regressor_forecast = preord_regressor_test, 
                                                  holiday_country = holiday_country,
                                                  startTimeStamps = profile_df.loc['FirstDate'],
                                                  template_cols = template_cols,
                                                  random_seed = random_seed, verbose = verbose)
                    model_count = template_result.model_count
                    validation_results.model_results = pd.concat([validation_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
                    validation_results.model_results['Score'] = generate_score(validation_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
                    validation_results.model_results_per_timestamp_smape = validation_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
                    validation_results.model_results_per_timestamp_mae = validation_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
                    validation_results.model_results_per_series_smape = validation_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
                    validation_results.model_results_per_series_mae = validation_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
                validation_results = validation_aggregation(validation_results)
        
        if not ensemble:
            validation_template[validation_template['Ensemble'] == 0]
        self.validation_results = validation_results
        self.main_results = main_results
        self.best_model = validation_results.model_results.sort_values(by = "Score", ascending = True, na_position = 'last').drop_duplicates(subset = template_cols).head(1)[template_cols]

        ensemble_check = (self.best_model['Ensemble'].iloc[0])
        param_dict = json.loads(self.best_model['ModelParameters'].iloc[0])
        if ensemble_check == 1:
            self.regression_check = False
            for key in param_dict['models']:
                try:
                    reg_param = json.loads(param_dict['models'][key]['ModelParameters'])['regression_type']
                    if reg_param == 'User':
                        self.regression_check = True
                except:
                    pass
        if ensemble_check == 0:
            self.regression_check = False
            try:
                reg_param =  param_dict['ModelParameters']['regression_type']
                if reg_param == 'User':
                    self.regression_check = True
            except:
                pass
        return self
    def predict(self, forecast_length: int = "self", preord_regressor = []):
        if forecast_length == 'self':
            forecast_length = self.forecast_length
        
        # if the models don't need the regressor, ignore it...
        if self.regression_check == False:
            preord_regressor = []
            self.preord_regressor_train = []
        
        from autots.evaluator.auto_model import PredictWitch
        df_forecast = PredictWitch(self.best_model, df_train = self.df_wide_numeric, 
                                   forecast_length= forecast_length, frequency= self.frequency, 
                                          prediction_interval= self.prediction_interval, 
                                          no_negatives= self.no_negatives,
                                          preord_regressor_train = self.preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor, 
                                          holiday_country = self.holiday_country,
                                          startTimeStamps = self.startTimeStamps,
                                          random_seed = self.random_seed, verbose = self.verbose,
                                       template_cols = self.template_cols)
        
        from autots.tools.shaping import categorical_inverse
        df_forecast.forecast = categorical_inverse(self.categorical_transformer, df_forecast.forecast)
        #df_forecast.lower_forecast = categorical_inverse(self.categorical_transformer, df_forecast.lower_forecast)
        #df_forecast.upper_forecast = categorical_inverse(self.categorical_transformer, df_forecast.upper_forecast)
        
        return df_forecast
        
    def export_template(filename, models: str = 'best'):
        """"
        output_format = 'csv' or 'json' (from filename)
        models = 'best', 'validation', or 'all'
        """
        print("Not yet implemented")
    def import_template(file):
        """"
        
        Args:
            file
        """
        print("Not yet implemented")
    def get_params(self):
        pass
