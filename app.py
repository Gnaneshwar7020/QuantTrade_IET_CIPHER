# Import required libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Define the AnomalyDetector class
class AnomalyDetector:
    def __init__(self, contamination=0.05, max_value=1e6):
        self.contamination = contamination
        self.max_value = max_value
        self.scaler = StandardScaler()
        self.models = {}

    def preprocess_data(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(upper=self.max_value)
        df['Year'] = df['Date'].dt.year
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_MA'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        return df

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return (macd - signal_line).fillna(0)

    def extract_features(self, df):
        features = pd.DataFrame()
        features['returns'] = df['Returns']
        features['volatility'] = df['Volatility']
        features['price_ma_ratio'] = df['Close'] / df['Price_MA']
        features['volume_ma_ratio'] = df['Volume'] / df['Volume_MA']
        features['volume_price_corr'] = df['Volume'].rolling(5).corr(df['Close'])
        features['high_low_ratio'] = df['High'] / df['Low']
        features['close_open_ratio'] = df['Close'] / df['Open']
        features['rsi'] = df['RSI']
        features['macd'] = df['MACD']
        features['price_volatility'] = features['returns'] * features['volatility']
        features['volume_intensity'] = features['volume_ma_ratio'] * features['price_volatility']
        features = features.fillna(0)
        features = features.clip(upper=self.max_value)
        features.replace([np.inf, -np.inf], self.max_value, inplace=True)
        return features

    def detect_anomalies(self, df, train=True):
        df_copy = df.copy()
        features = self.extract_features(df_copy)
        for year in df_copy['Year'].unique():
            year_mask = df_copy['Year'] == year
            year_features = features[year_mask]
            if len(year_features) < 10:
                continue
            X_scaled = self.scaler.fit_transform(year_features)
            if train:
                model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100,
                    max_samples='auto'
                )
                self.models[year] = model
                predictions = model.fit_predict(X_scaled)
            else:
                if year not in self.models:
                    continue
                predictions = self.models[year].predict(X_scaled)
            df_copy.loc[year_mask, 'Anomaly'] = predictions
            if train:
                scores = model.score_samples(X_scaled)
                df_copy.loc[year_mask, 'AnomalyScore'] = scores
        return df_copy

    def visualize_anomalies(self, df, year):
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        year_data = df[df['Year'] == year]
        normal_data = year_data[year_data['Anomaly'] == 1]
        anomaly_data = year_data[year_data['Anomaly'] == -1]
        fig.add_trace(
            go.Scatter(
                x=normal_data['Date'], y=normal_data['Close'], mode='lines',
                name=f'Normal {year}', line=dict(color='blue')
            ), row=1, col=1
        )
        if len(anomaly_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data['Date'], y=anomaly_data['Close'], mode='markers',
                    name=f'Anomalies {year}', marker=dict(color='red', size=8)
                ), row=1, col=1
            )
        fig.add_trace(
            go.Bar(
                x=year_data['Date'], y=year_data['Volume'], name='Volume',
                marker_color='lightgray', opacity=0.5
            ), row=2, col=1, secondary_y=False
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_layout(title_text=f"Anomaly Detection for {year}", showlegend=True)
        return fig

# Streamlit UI
def main():
    st.title("Anomaly Detection in Stock Data")
    st.write("Upload your CSV data to detect anomalies in stock prices and volume.")

    detector = AnomalyDetector(contamination=0.05)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data")
        st.write(df.head())

        st.write("### Processing and Detecting Anomalies...")
        processed_data = detector.preprocess_data(df)
        results_df = detector.detect_anomalies(processed_data, train=True)

        st.write("### Anomaly Detection Results")
        st.write(results_df.head())
        
        years = sorted(results_df['Year'].unique())
        selected_year = st.selectbox("Select Year for Visualization", years)

        fig = detector.visualize_anomalies(results_df, selected_year)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
