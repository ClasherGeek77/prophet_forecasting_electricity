import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

st.title("📊 Electricity Forecasting with Prophet")
st.write("Upload an Excel file with columns:")
st.markdown("""
- `Month` (e.g. 2020-01)
- `Electricity Sales to Ultimate Customers`
- `Electricity Net Generation, Electric Power Sector`
- `Electricity Exports`
""")
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    try:
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        df.fillna(method='ffill', inplace=True)

        regressors = ['Electricity Net Generation, Electric Power Sector', 'Electricity Exports']
        for reg in regressors:
            df[f'{reg}_lag12'] = df[reg].shift(12)

        df.dropna(inplace=True)

        df_train = df.iloc[-(508 + 12):-12]
        df_eval = df.iloc[-12:]

        target = 'Electricity Sales to Ultimate Customers'
        lagged_regs = [f'{reg}_lag12' for reg in regressors]
        train_df = df_train[[target] + lagged_regs].reset_index().rename(columns={'Month': 'ds', target: 'y'})

        future_df = df_eval[regressors].reset_index()
        future_df['ds'] = df_eval.index
        future_df.drop(columns='Month', inplace=True)
        for reg in regressors:
            future_df[f'{reg}_lag12'] = future_df[reg]
        future_df = future_df[['ds'] + [f'{reg}_lag12' for reg in regressors]]
        future_df['y'] = None

        combined_df = pd.concat([train_df, future_df], ignore_index=True)

        model = Prophet()
        for reg in lagged_regs:
            model.add_regressor(reg)
        model.fit(train_df)

        forecast = model.predict(combined_df)

        forecast_eval = forecast[forecast['ds'].isin(df_eval.index)].copy()
        actual = df_eval[target].values
        predicted = forecast_eval['yhat'].values

        st.subheader("📈 Actual vs Forecasted (Evaluation Period)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(forecast_eval['ds'], actual, label='Actual', color='blue', marker='o')
        ax1.plot(forecast_eval['ds'], predicted, label='Forecasted', color='red', marker='x')
        ax1.set_title('Actual vs Forecasted Electricity Sales')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Electricity Sales')
        ax1.grid(True)
        ax1.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        def evaluate_model(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return rmse, mae, mape

        rmse, mae, mape = evaluate_model(actual, predicted)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        last_month = df.index[-1]
        future_months = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=12, freq='MS')
        future_ext = pd.DataFrame({'ds': future_months})
        for reg in regressors:
            future_ext[f'{reg}_lag12'] = df[reg].iloc[-12:].values

        future_forecast = model.predict(future_ext)

        st.subheader("🔮 Future Forecast (Next 12 Months)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecasted', color='green', marker='x')
        ax2.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], color='green', alpha=0.2)
        ax2.set_title('Forecasted Electricity Sales')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Electricity Sales')
        ax2.grid(True)
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.subheader("📋 Forecast Table (Next 12 Months)")
        table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        table.columns = ['Month', 'Forecast', 'Lower Bound', 'Upper Bound']
        table['Month'] = table['Month'].dt.strftime('%Y-%m')

        csv = table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name='future_forecast.csv',
            mime='text/csv',
        )


        st.dataframe(table.style.format({
            'Forecast': '{:,.2f}',
            'Lower Bound': '{:,.2f}',
            'Upper Bound': '{:,.2f}'
        }))

    except Exception as e:
        st.error(f"Error processing file: {e}")
