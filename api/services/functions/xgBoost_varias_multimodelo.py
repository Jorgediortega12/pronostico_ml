import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
import joblib

@dataclass
class ScaleParams:
    scaler: RobustScaler

class DemandVariasMultiModelo:
    """
    Predictor de demanda con modelos separados por tipo de clima.
    Versión para múltiples temperaturas (CTG, MON, VDP).
    """
    def __init__(self):
        self.models = {
            'fresco': None,
            'templado': None,
            'caluroso': None
        }
        self.scale_params: Dict[str, Dict[str, ScaleParams]] = {
            'fresco': {},
            'templado': {},
            'caluroso': {}
        }
        self.temp_bins = None
        self.ciudades = ['CTG', 'MON', 'VDP']
        # Factores de ajuste por tipo de clima
        self.climate_adjustment_factors = {
            'fresco': 1.0,
            'templado': 1.0,
            'caluroso': 1.0
        }

    def _scale_feature(self, series: pd.Series, column: str, tipo_dia: str, params: ScaleParams = None) -> Tuple[pd.Series, ScaleParams]:
        """Escala una serie usando RobustScaler"""
        if params is None:
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            params = ScaleParams(scaler=scaler)
        else:
            scaled_values = params.scaler.transform(series.values.reshape(-1, 1)).flatten()

        return pd.Series(scaled_values, index=series.index), params

    def prepare_training_data(self, df: pd.DataFrame, tipo_dia: str, is_training: bool = True) -> pd.DataFrame:
        """Prepara los datos para múltiples ciudades"""
        df = df.copy()

        # Convertir fechas y separar columnas si es necesario
        columns = df.columns[0].split(';') if ';' in str(df.columns[0]) else df.columns.tolist()

        if len(df.columns) == 1:
            df_splitted = df[df.columns[0]].str.split(';', expand=True)
            df_splitted.columns = columns
            df = df_splitted

        df['Fecha'] = pd.to_datetime(df['Fecha'])

        # Procesar demanda
        if df['P'].dtype == object:
            df['P'] = df['P'].str.replace(',', '.').astype(float, errors='coerce')
        else:
            df['P'] = pd.to_numeric(df['P'], errors='coerce')

        df['demanda_promedio'] = df['P']

        # Limpiar valores extremos
        df = df.dropna(subset=['demanda_promedio'])
        df = df[~np.isinf(df['demanda_promedio'])]
        df = df[df['demanda_promedio'] < df['demanda_promedio'].quantile(0.999)]

        # Procesar temperaturas de múltiples ciudades
        for ciudad in self.ciudades:
            col_temp = f't_{ciudad}'

            if df[col_temp].dtype == object:
                df[col_temp] = df[col_temp].str.replace(',', '.').astype(float, errors='coerce')
            else:
                df[col_temp] = pd.to_numeric(df[col_temp], errors='coerce')

            # Variables derivadas por ciudad
            ventana_temp = 3
            df[f'temp_max_{ciudad}'] = df[col_temp].rolling(window=ventana_temp, min_periods=1).max()
            df[f'temp_min_{ciudad}'] = df[col_temp].rolling(window=ventana_temp, min_periods=1).min()
            df[f'temp_rango_{ciudad}'] = df[f'temp_max_{ciudad}'] - df[f'temp_min_{ciudad}']

        # Temperatura promedio de todas las ciudades
        df['temp_promedio'] = df[[f't_{ciudad}' for ciudad in self.ciudades]].mean(axis=1)

        # Características temporales
        df['mes'] = df['Fecha'].dt.month
        df['dia_semana'] = df['Fecha'].dt.dayofweek

        # Características de demanda histórica
        ventanas = [1, 2, 3, 7, 14, 30]
        for ventana in ventanas:
            df[f'demanda_{ventana}d_mean'] = df['demanda_promedio'].shift(ventana)
            df[f'demanda_{ventana}d_max'] = df['demanda_promedio'].rolling(window=ventana).max().shift(1)
            df[f'demanda_{ventana}d_min'] = df['demanda_promedio'].rolling(window=ventana).min().shift(1)
            df[f'demanda_{ventana}d_std'] = df['demanda_promedio'].rolling(window=ventana).std().shift(1)

        # FEATURES DE INTERACCIÓN (AMPLIFICACIÓN)
        # Interacciones con temperatura promedio
        df['temp_prom_x_demanda_1d'] = df['temp_promedio'] * df['demanda_1d_mean']
        df['temp_prom_x_demanda_7d'] = df['temp_promedio'] * df['demanda_7d_mean']
        df['temp_prom_x_demanda_14d'] = df['temp_promedio'] * df['demanda_14d_mean']

        # Interacciones específicas por ciudad
        for ciudad in self.ciudades:
            df[f'temp_{ciudad}_x_demanda'] = df[f't_{ciudad}'] * df['demanda_promedio']

        df['temp_prom_cuadrada'] = df['temp_promedio'] ** 2

        # Escalar variables numéricas
        numeric_features = [
            'demanda_promedio',
            'temp_promedio', 'temp_prom_cuadrada',
            'temp_prom_x_demanda_1d', 'temp_prom_x_demanda_7d', 'temp_prom_x_demanda_14d'
        ]

        # Agregar features de temperatura por ciudad
        for ciudad in self.ciudades:
            numeric_features.extend([
                f't_{ciudad}',
                f'temp_max_{ciudad}',
                f'temp_min_{ciudad}',
                f'temp_rango_{ciudad}',
                f'temp_{ciudad}_x_demanda'
            ])

        # Agregar features de demanda histórica
        numeric_features.extend([
            f'demanda_{v}d_{stat}' for v in ventanas for stat in ['mean', 'max', 'min', 'std']
        ])

        for col in numeric_features:
            if col in df.columns:
                mean_value = df[col].mean()
                if pd.isna(mean_value):
                    mean_value = 0
                series = df[col].fillna(mean_value)

                if is_training:
                    df[col], self.scale_params[tipo_dia][col] = self._scale_feature(series, col, tipo_dia)
                else:
                    if col in self.scale_params[tipo_dia]:
                        df[col], _ = self._scale_feature(series, col, tipo_dia, self.scale_params[tipo_dia][col])

        return df

    def train(self, df: pd.DataFrame):
        """Entrena modelos separados para cada tipo de día"""
        print("\n" + "="*60)
        print("ENTRENANDO MODELOS MULTI-CLIMA (VARIAS TEMPERATURAS)")
        print("="*60)

        df = df.copy()

        # Procesar columnas si vienen en formato CSV con ';'
        columns = df.columns[0].split(';') if ';' in str(df.columns[0]) else df.columns.tolist()

        if len(df.columns) == 1:
            df_splitted = df[df.columns[0]].str.split(';', expand=True)
            df_splitted.columns = columns
            df = df_splitted

        df['Fecha'] = pd.to_datetime(df['Fecha'])

        if df['P'].dtype == object:
            df['P'] = df['P'].str.replace(',', '.').astype(float, errors='coerce')
        else:
            df['P'] = pd.to_numeric(df['P'], errors='coerce')

        # Procesar temperaturas
        for ciudad in self.ciudades:
            col_temp = f't_{ciudad}'
            if df[col_temp].dtype == object:
                df[col_temp] = df[col_temp].str.replace(',', '.').astype(float, errors='coerce')
            else:
                df[col_temp] = pd.to_numeric(df[col_temp], errors='coerce')

        # Temperatura promedio para clasificación
        df['temp_promedio'] = df[[f't_{ciudad}' for ciudad in self.ciudades]].mean(axis=1)

        # Bins de temperatura AJUSTADOS para equilibrio
        # Ajustados para que templado quede entre fresco y caluroso
        self.temp_bins = [-np.inf, 29.0, 30.5, np.inf]

        print(f"\nBins de temperatura (AJUSTADOS - promedio de {', '.join(self.ciudades)}):")
        print(f"  Fresco:    < {self.temp_bins[1]:.1f}°C")
        print(f"  Templado:  {self.temp_bins[1]:.1f}°C - {self.temp_bins[2]:.1f}°C (rango de 3°C)")
        print(f"  Caluroso:  > {self.temp_bins[2]:.1f}°C")

        # Clasificar días
        df['tipo_dia'] = pd.cut(df['temp_promedio'], bins=self.temp_bins, labels=['fresco', 'templado', 'caluroso'])

        # FACTORES FIJOS para equilibrio perfecto
        print(f"\nAplicando factores de ajuste FIJOS (equilibrio perfecto):")

        self.climate_adjustment_factors = {
            'fresco': 0.90,
            'templado': 1.05,
            'caluroso': 1.20
        }

        for tipo in ['fresco', 'templado', 'caluroso']:
            pct_change = (self.climate_adjustment_factors[tipo] - 1.0) * 100
            print(f"  {tipo.capitalize():8s}: Factor = {self.climate_adjustment_factors[tipo]:.2f} ({pct_change:+.0f}%)")

        # Entrenar modelos
        for tipo in ['fresco', 'templado', 'caluroso']:
            print(f"\n--- Entrenando modelo para dias {tipo.upper()} ---")

            df_tipo = df[df['tipo_dia'] == tipo].copy()
            print(f"Datos disponibles: {len(df_tipo)} dias")

            if len(df_tipo) < 50:
                print(f"[!] ADVERTENCIA: Pocos datos para tipo '{tipo}'")

            df_processed = self.prepare_training_data(df_tipo, tipo, is_training=True)

            features = [col for col in df_processed.columns if col not in
                       ['Fecha', 'tipo_dia', 'P', 'demanda_promedio_raw'] +
                       [f't_{ciudad}' for ciudad in self.ciudades]]

            df_processed = df_processed.dropna(subset=features + ['demanda_promedio'])

            if len(df_processed) < 20:
                print(f"[!] ERROR: Quedan muy pocos datos ({len(df_processed)})")
                continue

            X = df_processed[features]
            y = df_processed['demanda_promedio']

            print(f"Datos para entrenamiento: {len(X)} registros, {len(features)} features")

            self.models[tipo] = xgb.XGBRegressor(
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

            self.models[tipo].fit(X, y)
            print(f"[OK] Modelo entrenado para dias {tipo}")

        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*60)

    def predict(self, df: pd.DataFrame, tipo_dia: str) -> Tuple[float, pd.Timestamp]:
        """Realiza predicción usando el modelo específico del tipo de día"""
        if tipo_dia not in self.models:
            raise ValueError(f"tipo_dia debe ser 'fresco', 'templado' o 'caluroso', no '{tipo_dia}'")

        if self.models[tipo_dia] is None:
            raise ValueError(f"El modelo para tipo '{tipo_dia}' no ha sido entrenado")

        df_processed = self.prepare_training_data(df, tipo_dia, is_training=False)

        features = [col for col in df_processed.columns if col not in
                   ['Fecha', 'tipo_dia', 'P', 'demanda_promedio_raw'] +
                   [f't_{ciudad}' for ciudad in self.ciudades]]

        ultima_fecha = df_processed['Fecha'].max()
        fecha_siguiente = ultima_fecha + timedelta(days=1)

        ultimo_registro = df_processed.iloc[-1].copy()
        ultimo_registro['Fecha'] = fecha_siguiente
        ultimo_registro['mes'] = fecha_siguiente.month
        ultimo_registro['dia_semana'] = fecha_siguiente.dayofweek

        # Obtener el orden de features que el modelo espera
        model_features = self.models[tipo_dia].get_booster().feature_names

        # Crear X_pred con el orden correcto de features
        X_pred = pd.DataFrame([ultimo_registro[model_features]], columns=model_features)

        prediccion = self.models[tipo_dia].predict(X_pred)[0]

        prediccion = self.scale_params[tipo_dia]['demanda_promedio'].scaler.inverse_transform(
            prediccion.reshape(-1, 1))[0][0]

        # APLICAR FACTOR DE AJUSTE
        prediccion = prediccion * self.climate_adjustment_factors[tipo_dia]

        return prediccion, fecha_siguiente

    def save_models(self, base_path='models/xgboost_varias_multimodelo'):
        """Guarda los modelos y parámetros"""
        import os
        os.makedirs(base_path, exist_ok=True)

        for tipo in ['fresco', 'templado', 'caluroso']:
            if self.models[tipo] is not None:
                joblib.dump(self.models[tipo], f'{base_path}/model_{tipo}.joblib')
                joblib.dump(self.scale_params[tipo], f'{base_path}/scale_params_{tipo}.joblib')

        joblib.dump(self.temp_bins, f'{base_path}/temp_bins.joblib')
        joblib.dump(self.climate_adjustment_factors, f'{base_path}/adjustment_factors.joblib')
        print(f"\n[OK] Modelos guardados en {base_path}/")

    def load_models(self, base_path='models/xgboost_varias_multimodelo'):
        """Carga los modelos y parámetros"""
        for tipo in ['fresco', 'templado', 'caluroso']:
            self.models[tipo] = joblib.load(f'{base_path}/model_{tipo}.joblib')
            self.scale_params[tipo] = joblib.load(f'{base_path}/scale_params_{tipo}.joblib')

        self.temp_bins = joblib.load(f'{base_path}/temp_bins.joblib')
        self.climate_adjustment_factors = joblib.load(f'{base_path}/adjustment_factors.joblib')
        print(f"\n[OK] Modelos cargados desde {base_path}/")
