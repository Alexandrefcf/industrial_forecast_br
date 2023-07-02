import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import re


# analise = TimeSeriesAnalysis(df)
# df_estacionado = analise.estacionar(analise.df_75)


class TimeSeriesAnalysis:
    def __init__(self, df):
        self.df = df
        self.process_dataframe()
        #self.faz_tudo()
        
    def faz_tudo(self):

        #self.df_all_sectors_est = self.estacionar(self.df_all_sectors) # [0] ver se precisa pegar?
        #self.df_macro_sectors_est = self.estacionar(self.df_macro_sectors)
        #self.df_micro_sectors_est = self.estacionar(self.df_micro_sectors)
        
        #self.pred_all_sectors = self.find_best_predictors(self.df_all_sectors,5)
        #self.pred_macro_sectors = self.find_best_predictors(self.df_macro_sectors,5)
        #self.pred_all_sectors_est = self.find_best_predictors(self.df_all_sectors_est[0],5)
        #self.pred_macro_sectors_est = self.find_best_predictors(self.df_macro_sectors_est[0],5)
        
        #self.var_all_sectors, self.metrics_var_all = self.create_var_models(self.df_all_sectors, self.pred_all_sectors, 5, 0.90)
        #self.var_macro_sectors, self.metrics_var_macro = self.create_var_models(self.df_macro_sectors, self.pred_macro_sectors, 5, 0.90)
        #self.var_all_sectors_est, self.metrics_var_all_est = self.create_var_models(self.df_all_sectors_est, self.pred_all_sectors_est, 5, 0.90)
        #self.var_macro_sectors_est, self.metrics_var_macro_est = self.create_var_models(self.df_macro_sectors_est, self.pred_macro_sectors_est, 5, 0.90)
 
        return
    
    def process_dataframe(self):
        df_103 = self.df.iloc[:, :]
        df_103 = self.df.replace('-', 0)

        df_75 = self.df.select_dtypes(include=['float64', 'int64'])

        dicionario = {}

        for coluna in df_75.columns:
            padrao = r'^([\d.]+)\s(.+)$'
            correspondencia = re.match(padrao, coluna)
            numero = correspondencia.group(1)
            texto = correspondencia.group(2)

            dicionario[coluna] = {'numero': numero, 'texto': texto}

        df_75 = df_75.rename(columns=lambda x: dicionario[x]['numero'])

        df_53 = pd.DataFrame()
        df_22 = pd.DataFrame()

        for coluna in df_75.columns:
            numero = coluna.split()[0]
            numero_digitos = len(coluna)

            if numero_digitos == 4:
                df_53[coluna] = df_75[coluna]
            elif numero_digitos == 5:
                df_22[coluna] = df_75[coluna]

        # dict_facilitador = {75:self.df_75, 53:self.df_53, 22:self.df_22}

        self.dataframes = {
                    'df_75': df_75,
                    'df_53': df_53,
                    'df_22': df_22,
                    'dicionario': pd.DataFrame.from_dict(dicionario),
                    }

        # Concatenar os dataframes ao longo do eixo das colunas
    #    self.df_combinado = pd.concat(dataframes.values(), axis=1)
        self.df_all_sectors = df_75
        self.df_macro_sectors = df_53
        self.df_micro_sectors = df_22
    #    self.dicionario = dicionario

        return
    
    def estacionar(self, df_n_est):
        estacionados = []
        diff_counts = []

        for col in df_n_est.columns:
            col_estacionaria = False
            count_diff = 0
            col_data = df_n_est[col].copy()

            while not col_estacionaria:
                if adfuller(col_data)[1] > 0.05:
                    col_data = col_data.diff().dropna()
                    count_diff += 1
                else:
                    col_estacionaria = True

            estacionados.append(col_data)
            diff_counts.append({'coluna': col, 'diff_count': count_diff})

        self.df_estacionado = pd.concat(estacionados, axis=1)
        self.df_estacionado = self.df_estacionado #Presta atenção, para o caso macro sectors, excluiu a primeira linha, pois a NaN e números
        self.diff_count = pd.DataFrame(diff_counts)

        return self.df_estacionado #Ver se precisa colocar eles coom retorno


    def find_best_predictors(self,df_pred, lag):
        predictors = {}

        for target_col in df_pred.columns:
            best_predictors = []
            target_series = df_pred[target_col]
            p_values = []
            #
            for predictor_col in df_pred.columns:
                if predictor_col != target_col:
                    predictor_series = df_pred[predictor_col]
                    results = grangercausalitytests(pd.concat([target_series, predictor_series], axis=1), lag, verbose=False)
                    p_values = [results[1][0]['ssr_ftest'][1] for i in range(lag)]
                    if all(p < 0.05 for p in p_values):
                        best_predictors.append(predictor_col)
            predictors[target_col] = best_predictors

        return predictors

    def create_var_models(self, data_base, predictors, lag, train_test_ratio):
        var_models = {}
        metrics = {}

        for target_col, predictor_cols in predictors.items():
            data = self.data_base[[target_col] + predictor_cols]
            data.index = pd.date_range(start='2002-01-01', periods=len(data), freq='M')

            train_size = int(len(data) * train_test_ratio)
            train_data = data[:train_size]
            test_data = data[train_size:]

            model = VAR(train_data)
            result = model.fit(maxlags=lag)
            var_models[target_col] = result

            pred = result.forecast(test_data.values, len(test_data))

            mae = round(mean_absolute_error(test_data.values, pred), 4)
            mape = round(mean_absolute_percentage_error(test_data.values, pred), 4)
            mse = round(mean_squared_error(test_data.values, pred), 4)

            metrics[target_col] = {'MAE': mae, 'MAPE': mape, 'MSE': mse}

        return var_models, metrics
