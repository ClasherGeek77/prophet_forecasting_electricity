import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("⚡ Electricity Forecasting with Auto-Ensemble")

st.write("Upload an Excel file with columns:")
st.markdown("""
- `Month` (e.g. 2020-01)
- `Electricity Sales to Ultimate Customers`
- `Electricity Net Generation, Electric Power Sector`
- `Electricity Exports`
""")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

if uploaded_file:
    with st.spinner("Running forecasting pipeline..."):
        df = pd.read_excel(uploaded_file)
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        target = 'Electricity Sales to Ultimate Customers'
        regressors = ['Electricity Net Generation, Electric Power Sector', 'Electricity Exports']
        for reg in regressors:
            df[f'{reg}_lag12'] = df[reg].shift(12)
        df.dropna(inplace=True)

        df_eval = df.iloc[-12:]
        actual = df_eval[target].values
        proportions = np.linspace(0.1, 1.0, 10)

        # Prophet
        prophet_results = []
        for p in proportions:
            train_size = int(len(df) * p)
            prophet_train = df.iloc[-(train_size + 12):-12]
            if len(prophet_train) < 24:
                continue
            train_df = prophet_train[[target] + [f'{reg}_lag12' for reg in regressors]].reset_index()
            train_df.rename(columns={'Month': 'ds', target: 'y'}, inplace=True)

            future_df = df_eval[regressors].reset_index()
            future_df['ds'] = df_eval.index
            for reg in regressors:
                future_df[f'{reg}_lag12'] = future_df[reg]
            future_df = future_df[['ds'] + [f'{reg}_lag12' for reg in regressors]]
            future_df['y'] = None

            combined_df = pd.concat([train_df, future_df], ignore_index=True)

            model = Prophet()
            for reg in [f'{reg}_lag12' for reg in regressors]:
                model.add_regressor(reg)
            try:
                model.fit(train_df)
                forecast = model.predict(combined_df)
                pred = forecast[forecast['ds'].isin(df_eval.index)]['yhat'].values
                rmse, mae, mape = evaluate(actual, pred)
                prophet_results.append((p, pred, rmse, mae, mape))
            except:
                continue
        best_prophet = min(prophet_results, key=lambda x: x[4])
        best_prophet_p, prophet_preds, *_ = best_prophet

        # VAR
        var_data = df[[target, 'Electricity Exports']]
        var_results = []
        for p in proportions:
            train_size = int(len(var_data) * p)
            var_train = var_data.iloc[-(train_size + 12):-12]
            if len(var_train) < 24:
                continue
            try:
                model = VAR(var_train)
                selected_lag = model.select_order(12).selected_orders['aic']
                results_model = model.fit(selected_lag)
                forecast_input = var_train.values[-selected_lag:]
                var_forecast = results_model.forecast(y=forecast_input, steps=12)
                pred = pd.DataFrame(var_forecast, index=df_eval.index, columns=var_data.columns)[target].values
                rmse, mae, mape = evaluate(actual, pred)
                var_results.append((p, pred, rmse, mae, mape))
            except:
                continue
        best_var = min(var_results, key=lambda x: x[4])
        best_var_p, var_preds, *_ = best_var

        # BiLSTM
        bilstm_results = []
        scaler = MinMaxScaler()
        lag = 12
        scaled_df = pd.DataFrame(scaler.fit_transform(df[[target] + regressors]),
                                columns=[target] + regressors,
                                index=df.index)
        for p in [0.8]:
            train_size = int(len(df) * p)
            scaled_train = scaled_df.iloc[-(train_size + 12):-12]
            if len(scaled_train) <= lag:
                continue
            X, y = [], []
            for i in range(lag, len(scaled_train)):
                X.append(scaled_train.iloc[i-lag:i].values)
                y.append(scaled_train.iloc[i][0])
            X, y = np.array(X), np.array(y)

            model = Sequential([
                Bidirectional(LSTM(64, activation='relu'), input_shape=(lag, X.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=100, verbose=0)

            inputs = scaled_df.iloc[-lag:].values
            preds = []
            for _ in range(12):
                inp = inputs[-lag:]
                inp = inp.reshape(1, lag, X.shape[2])
                pred_scaled = model.predict(inp, verbose=0)[0][0]
                preds.append(pred_scaled)
                new_row = np.concatenate([[pred_scaled], inputs[-1][1:]])
                inputs = np.vstack([inputs, new_row])

            forecast = scaler.inverse_transform(np.column_stack([preds, np.zeros((12, len(regressors)))]))[:, 0]
            rmse, mae, mape = evaluate(actual, forecast)
            bilstm_results.append((p, forecast, rmse, mae, mape))
        best_bilstm = min(bilstm_results, key=lambda x: x[4])
        best_bilstm_p, bilstm_preds, *_ = best_bilstm

        # Ensemble
        ensemble_results = []
        weights = np.linspace(0, 1, 11)
        for w1 in weights:
            for w2 in weights:
                if w1 + w2 > 1:
                    continue
                w3 = 1 - w1 - w2
                blended = w1 * prophet_preds + w2 * var_preds + w3 * bilstm_preds
                rmse, mae, mape = evaluate(actual, blended)
                ensemble_results.append((w1, w2, w3, rmse, mae, mape))
        best_ens = min(ensemble_results, key=lambda x: x[5])
        best_w1, best_w2, best_w3, best_rmse, best_mae, best_mape = best_ens
        best_blended = best_w1 * prophet_preds + best_w2 * var_preds + best_w3 * bilstm_preds

        # Evaluation Plot
        st.subheader("📊 Actual vs Forecasted (Evaluation Period)")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_eval.index, actual, label='Actual', color='black', linewidth=2)
        ax.plot(df_eval.index, prophet_preds, label='Best Prophet', linestyle='--')
        ax.plot(df_eval.index, var_preds, label='Best VAR', linestyle='--')
        ax.plot(df_eval.index, bilstm_preds, label='Best BiLSTM', linestyle='--')
        ax.plot(df_eval.index, best_blended, label='Best Ensemble', color='red')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Future Forecast
        future_regressors = df[regressors].iloc[-12:].copy()
        future_regressors.index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        for reg in regressors:
            future_regressors[f'{reg}_lag12'] = future_regressors[reg]

        future_df = future_regressors[[f'{reg}_lag12' for reg in regressors]].copy()
        future_df['ds'] = future_regressors.index
        future_df = future_df[['ds'] + [f'{reg}_lag12' for reg in regressors]]

        last_train_df = df.reset_index().rename(columns={'Month': 'ds', target: 'y'})
        for reg in regressors:
            last_train_df[f'{reg}_lag12'] = last_train_df[reg].shift(12)
        last_train_df = last_train_df.dropna()

        final_prophet = Prophet()
        for reg in [f'{reg}_lag12' for reg in regressors]:
            final_prophet.add_regressor(reg)
        final_prophet.fit(last_train_df[['ds', 'y'] + [f'{reg}_lag12' for reg in regressors]])
        prophet_future_pred = final_prophet.predict(future_df)['yhat'].values

        final_var_model = VAR(df[[target, 'Electricity Exports']])
        final_var_fitted = final_var_model.fit(selected_lag)
        var_future = final_var_fitted.forecast(df[[target, 'Electricity Exports']].values[-selected_lag:], steps=12)
        var_future_pred = pd.DataFrame(var_future, columns=[target, 'Electricity Exports']).iloc[:, 0].values

        bilstm_future_input = scaled_df.iloc[-lag:].values.copy()
        bilstm_future_preds = []
        for _ in range(12):
            inp = bilstm_future_input[-lag:]
            inp = inp.reshape(1, lag, inp.shape[1])
            pred_scaled = model.predict(inp, verbose=0)[0][0]
            bilstm_future_preds.append(pred_scaled)
            new_row = np.concatenate([[pred_scaled], bilstm_future_input[-1][1:]])
            bilstm_future_input = np.vstack([bilstm_future_input, new_row])
        bilstm_future_pred = scaler.inverse_transform(np.column_stack([bilstm_future_preds, np.zeros((12, len(regressors)))]))[:, 0]

        future_ensemble = best_w1 * prophet_future_pred + best_w2 * var_future_pred + best_w3 * bilstm_future_pred
        future_index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

        st.subheader("🔮 Future Forecast (Next 12 Months)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(future_index, future_ensemble, label='Forecasted', marker='x', color='green')
        ax2.set_title('Ensemble Forecast: Electricity Sales')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Electricity Sales')
        ax2.grid(True)
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        forecast_df = pd.DataFrame({
            'Month': future_index.strftime('%Y-%m'),
            'Forecast': future_ensemble
        })
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')
        st.subheader("📊 Forecast Table (Next 12 Months) ")
        st.dataframe(forecast_df)


        st.subheader("📋 Forecast Summary")
        st.markdown(f"**Best Training Proportions**  \n- Prophet: `{int(best_prophet_p*100)}%`  \n- VAR: `{int(best_var_p*100)}%`  \n- BiLSTM: `{int(best_bilstm_p*100)}%`")
        st.markdown(f"**Best Ensemble Weights**  \n- Prophet: `{int(best_w1*100)}%`  \n- VAR: `{int(best_w2*100)}%`  \n- BiLSTM: `{int(best_w3*100)}%`")
        st.success(f"RMSE: {best_rmse:.2f} | MAE: {best_mae:.2f} | MAPE: {best_mape:.2f}%")
