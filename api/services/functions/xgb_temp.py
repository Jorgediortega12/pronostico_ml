
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import joblib

@dataclass
class ScaleParams:
    scaler: RobustScaler

class DemandPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scale_params: Dict[str, ScaleParams] = {}
        
    def _scale_feature(self, series: pd.Series, column: str, params: ScaleParams = None) -> Tuple[pd.Series, ScaleParams]:
        """Escala una serie usando RobustScaler para manejar mejor los outliers"""
        if params is None:
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            params = ScaleParams(scaler=scaler)
        else:
            scaled_values = params.scaler.transform(series.values.reshape(-1, 1)).flatten()
        
        return pd.Series(scaled_values, index=series.index), params

    def prepare_training_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepara los datos históricos para el entrenamiento o predicción
        Adaptado para formato simplificado con columnas Fecha, P, t"""
        df = df.copy()
        
        # Convertir fechas
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Procesar demanda (ahora es un solo valor P)
        if df['P'].dtype == object:
            df['P'] = df['P'].str.replace(',', '.').astype(float, errors='coerce')
        else:
            df['P'] = pd.to_numeric(df['P'], errors='coerce')
        
        # Renombrar P a demanda_promedio para mantener compatibilidad con el código existente
        df['demanda_promedio'] = df['P']
        
        # Limpiar valores extremos
        df = df.dropna(subset=['demanda_promedio'])
        df = df[~np.isinf(df['demanda_promedio'])]
        df = df[df['demanda_promedio'] < df['demanda_promedio'].quantile(0.999)]  # Eliminar outliers extremos
        
        # Procesar temperatura (ahora es un solo valor t)
        if df['t'].dtype == object:
            df['t'] = df['t'].str.replace(',', '.').astype(float, errors='coerce')
        else:
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
        
        # Renombrar t a temp_promedio para mantener compatibilidad
        df['temp_promedio'] = df['t']
        
        # Ya no tenemos valores horarios, así que no podemos calcular máximos y mínimos diarios directamente
        # Podemos aproximar estos valores utilizando ventanas de tiempo
        ventana_temp = 3  # días para calcular variaciones de temperatura
        df['temp_max'] = df['temp_promedio'].rolling(window=ventana_temp, min_periods=1).max()
        df['temp_min'] = df['temp_promedio'].rolling(window=ventana_temp, min_periods=1).min()
        df['temp_rango'] = df['temp_max'] - df['temp_min']
        
        # Características temporales
        df['mes'] = df['Fecha'].dt.month
        df['dia_semana'] = df['Fecha'].dt.dayofweek
        
        # Ya no tenemos la hora pico, así que eliminamos esa característica
        
        # Variables categóricas (clasificación por temperatura)
        if is_training:
            self.label_encoder = LabelEncoder()
            df['tipo_dia'] = pd.cut(df['temp_promedio'],
                                  bins=[-np.inf, df['temp_promedio'].quantile(0.33),
                                       df['temp_promedio'].quantile(0.66), np.inf],
                                  labels=['fresco', 'templado', 'caluroso'])
            df['tipo_dia_encoded'] = self.label_encoder.fit_transform(df['tipo_dia'])
        else:
            # Si 'tipo_dia' no está en el dataframe y estamos en modo predicción,
            # tenemos que asegurarnos de que este paso no falle
            if 'tipo_dia' not in df.columns:
                # Clasificamos según los umbrales generados durante el entrenamiento
                bins = self.temp_bins if hasattr(self, 'temp_bins') else [-np.inf, 15, 25, np.inf]
                df['tipo_dia'] = pd.cut(df['temp_promedio'], bins=bins, labels=['fresco', 'templado', 'caluroso'])
            df['tipo_dia_encoded'] = self.label_encoder.transform(df['tipo_dia'])
        
        # Características de demanda histórica con diferentes ventanas
        ventanas = [1, 2, 3, 7, 14, 30]
        for ventana in ventanas:
            df[f'demanda_{ventana}d_mean'] = df['demanda_promedio'].shift(ventana)
            df[f'demanda_{ventana}d_max'] = df['demanda_promedio'].rolling(window=ventana).max().shift(1)
            df[f'demanda_{ventana}d_min'] = df['demanda_promedio'].rolling(window=ventana).min().shift(1)
            df[f'demanda_{ventana}d_std'] = df['demanda_promedio'].rolling(window=ventana).std().shift(1)
        
        # Escalar variables numéricas
        numeric_features = [
            'demanda_promedio',
            'temp_promedio', 'temp_max', 'temp_min', 'temp_rango',
        ] + [f'demanda_{v}d_{stat}' for v in ventanas for stat in ['mean', 'max', 'min', 'std']]
        
        for col in numeric_features:
            if col in df.columns:
                # Llenar NaN con la media
                mean_value = df[col].mean()
                if pd.isna(mean_value):  # Si toda la columna es NaN
                    mean_value = 0
                series = df[col].fillna(mean_value)
                
                if is_training:
                    df[col], self.scale_params[col] = self._scale_feature(series, col)
                    # Guardar los bins de temperatura para usarlos en predicción
                    if col == 'temp_promedio':
                        self.temp_bins = [-np.inf, 
                                        df['temp_promedio'].quantile(0.33), 
                                        df['temp_promedio'].quantile(0.66), 
                                        np.inf]
                else:
                    if col in self.scale_params:
                        df[col], _ = self._scale_feature(series, col, self.scale_params[col])
        
        return df

    def train(self, df: pd.DataFrame):
        """Entrena el modelo con configuración mejorada"""
        df_processed = self.prepare_training_data(df, is_training=True)
        
        features = [col for col in df_processed.columns if col not in 
                   ['Fecha', 'tipo_dia', 'P', 't', 'demanda_promedio_raw']]
        
        X = df_processed[features]
        y = df_processed['demanda_promedio']
        
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )
        
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame, tipo_dia: str) -> Tuple[float, pd.Timestamp]:
        """Realiza predicción para el siguiente día"""
        df_copy = df.copy()
        df_copy['tipo_dia'] = tipo_dia
        df_processed = self.prepare_training_data(df_copy, is_training=False)
        
        features = [col for col in df_processed.columns if col not in 
                   ['Fecha', 'tipo_dia', 'P', 't', 'demanda_promedio_raw']]
        
        ultima_fecha = df_processed['Fecha'].max()
        fecha_siguiente = ultima_fecha + timedelta(days=1)
        
        ultimo_registro = df_processed.iloc[-1].copy()
        ultimo_registro['Fecha'] = fecha_siguiente
        ultimo_registro['tipo_dia'] = tipo_dia
        ultimo_registro['tipo_dia_encoded'] = self.label_encoder.transform([tipo_dia])[0]
        ultimo_registro['mes'] = fecha_siguiente.month
        ultimo_registro['dia_semana'] = fecha_siguiente.dayofweek
        
        X_pred = pd.DataFrame([ultimo_registro[features]], columns=features)
        
        prediccion = self.model.predict(X_pred)[0]
        prediccion = self.scale_params['demanda_promedio'].scaler.inverse_transform(
            prediccion.reshape(-1, 1))[0][0]
        
        return prediccion, fecha_siguiente

    def evaluate(self, df: pd.DataFrame, test_days: int = 90):
        """Evalúa el modelo con métricas detalladas"""
        df_processed = self.prepare_training_data(df, is_training=True)
        df_processed = df_processed.sort_values('Fecha')
        
        test_start = df_processed['Fecha'].max() - pd.Timedelta(days=test_days)
        train_data = df_processed[df_processed['Fecha'] < test_start].copy()
        test_data = df_processed[df_processed['Fecha'] >= test_start]
        
        self.train(train_data)
        
        predictions = []
        for idx in range(len(test_data)):
            current_date = test_data.iloc[idx]['Fecha']
            df_until_current = df_processed[df_processed['Fecha'] <= current_date].copy()
            tipo_dia = test_data.iloc[idx]['tipo_dia']
            
            try:
                pred, _ = self.predict(df_until_current, tipo_dia)
                real_value = self.scale_params['demanda_promedio'].scaler.inverse_transform(
                    test_data.iloc[idx]['demanda_promedio'].reshape(-1, 1))[0][0]
                
                predictions.append({
                    'Fecha': current_date,
                    'Real': real_value,
                    'Prediccion': pred
                })
            except Exception as e:
                print(f"Error en predicción para {current_date}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No se pudieron generar predicciones")
        
        results = pd.DataFrame(predictions)
        
        metrics = {
            'R2': r2_score(results['Real'], results['Prediccion']),
            'MAPE': np.mean(np.abs((results['Real'] - results['Prediccion']) / results['Real'])) * 100
        }
        
        # Visualizaciones
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Gráfico de series temporales
        ax1.plot(results['Fecha'], results['Real'], 'b-', label='Real')
        ax1.plot(results['Fecha'], results['Prediccion'], 'r-', label='Predicción', alpha=0.7)
        ax1.set_title('Predicciones vs Valores Reales')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Demanda')
        ax1.legend()
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico de dispersión
        ax2.scatter(results['Real'], results['Prediccion'], c='blue', alpha=0.5)
        min_val = min(results['Real'].min(), results['Prediccion'].min())
        max_val = max(results['Real'].max(), results['Prediccion'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea perfecta')
        ax2.set_xlabel('Valores Reales')
        ax2.set_ylabel('Predicciones')
        ax2.set_title('Predicciones vs Reales (Dispersión)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\nMétricas de Evaluación:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return results