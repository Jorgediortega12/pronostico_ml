import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import joblib

@dataclass
class ScaleParams:
    scaler: RobustScaler

class DemandOnlyPredictor:
    def __init__(self):
        self.model = None
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
        """Prepara los datos históricos usando solo información de demanda pasada"""
        df = df.copy()
        
        # Convertir fechas de manera más flexible
        try:
            # Primero intentamos con formato explícito para fechas cortas
            df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%y')
        except ValueError:
            try:
                # Si falla, intentamos con formato de ano completo
                df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
            except ValueError:
                # Si ambos fallan, usamos el parsing automático con dayfirst=True
                df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
        df = df.sort_values('Fecha')
        
        # Procesar columna de demanda
        if 'P' in df.columns:
            df['demanda_promedio_raw'] = df['P']
        else:
            raise ValueError("La columna 'P' no existe en el DataFrame")
        
        # Limpiar valores extremos
        df = df.dropna(subset=['demanda_promedio_raw'])
        df = df[~np.isinf(df['demanda_promedio_raw'])]
        df = df[df['demanda_promedio_raw'] < df['demanda_promedio_raw'].quantile(0.999)]
        
        # Características temporales
        df['mes'] = df['Fecha'].dt.month
        df['dia_semana'] = df['Fecha'].dt.dayofweek
        
        # Escalar demanda_promedio
        if is_training:
            df['demanda_promedio'], self.scale_params['demanda_promedio'] = self._scale_feature(
                df['demanda_promedio_raw'], 'demanda_promedio')
        else:
            df['demanda_promedio'], _ = self._scale_feature(
                df['demanda_promedio_raw'], 'demanda_promedio', self.scale_params['demanda_promedio'])
        
        # Características de demanda histórica con diferentes ventanas
        ventanas = [7, 14, 30]
        for ventana in ventanas:
            df[f'demanda_{ventana}d_mean'] = df['demanda_promedio_raw'].rolling(window=ventana).mean().shift(1)
            df[f'demanda_{ventana}d_max'] = df['demanda_promedio_raw'].rolling(window=ventana).max().shift(1)
            df[f'demanda_{ventana}d_min'] = df['demanda_promedio_raw'].rolling(window=ventana).min().shift(1)
            df[f'demanda_{ventana}d_std'] = df['demanda_promedio_raw'].rolling(window=ventana).std().shift(1)
        
        # Características de demanda por día de la semana
        for dia in range(1, 8):
            df[f'demanda_dia_{dia}_anterior'] = df['demanda_promedio_raw'].shift(dia)
        
        # Variables adicionales
        df['demanda_promedio_anterior'] = df['demanda_promedio_raw'].shift(1)
        df['demanda_semana_anterior'] = df['demanda_promedio_raw'].shift(7)
        df['diff_semana_anterior'] = df['demanda_promedio_raw'] - df['demanda_semana_anterior']
        
        # Escalar variables numéricas
        numeric_features = ['demanda_promedio_anterior', 'demanda_semana_anterior', 'diff_semana_anterior'] + \
                         [f'demanda_{v}d_{stat}' for v in ventanas for stat in ['mean', 'max', 'min', 'std']] + \
                         [f'demanda_dia_{d}_anterior' for d in range(1, 8)]
        
        for col in numeric_features:
            if col in df.columns:
                series = df[col].fillna(df[col].mean())
                if is_training:
                    df[col], self.scale_params[col] = self._scale_feature(series, col)
                else:
                    df[col], _ = self._scale_feature(series, col, self.scale_params[col])
        
        return df

    def train(self, df: pd.DataFrame):
        """Entrena el modelo usando solo características de demanda"""
        df_processed = self.prepare_training_data(df, is_training=True)
        
        features = ['mes', 'dia_semana', 'demanda_promedio_anterior', 'demanda_semana_anterior', 
                   'diff_semana_anterior']
        
        ventanas = [7, 14, 30]
        for ventana in ventanas:
            features.extend([
                f'demanda_{ventana}d_mean',
                f'demanda_{ventana}d_max',
                f'demanda_{ventana}d_min',
                f'demanda_{ventana}d_std'
            ])
        
        for dia in range(1, 8):
            features.append(f'demanda_dia_{dia}_anterior')
        
        # Filtrar datos y remover filas con NaN
        df_processed = df_processed.dropna(subset=features + ['demanda_promedio'])
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

    def predict(self, df: pd.DataFrame) -> Tuple[float, pd.Timestamp]:
        """Realiza predicción para el siguiente día"""
        df_processed = self.prepare_training_data(df, is_training=False)
        
        features = ['mes', 'dia_semana', 'demanda_promedio_anterior', 'demanda_semana_anterior', 
                   'diff_semana_anterior']
        
        ventanas = [7, 14, 30]
        for ventana in ventanas:
            features.extend([
                f'demanda_{ventana}d_mean',
                f'demanda_{ventana}d_max',
                f'demanda_{ventana}d_min',
                f'demanda_{ventana}d_std'
            ])
        
        for dia in range(1, 8):
            features.append(f'demanda_dia_{dia}_anterior')
        
        ultima_fecha = df_processed['Fecha'].max()
        fecha_siguiente = ultima_fecha + timedelta(days=1)
        
        ultimo_registro = df_processed.iloc[-1][features].copy()
        ultimo_registro['mes'] = fecha_siguiente.month
        ultimo_registro['dia_semana'] = fecha_siguiente.dayofweek
        
        X_pred = pd.DataFrame([ultimo_registro])
        
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
            
            try:
                pred, _ = self.predict(df_until_current)
                real_value = test_data.iloc[idx]['demanda_promedio_raw']
                
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