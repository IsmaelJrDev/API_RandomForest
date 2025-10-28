from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import time
import traceback
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report  # accuracy_score y confusion_matrix no se usan en la respuesta final
from sklearn.tree import plot_tree
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
# import seaborn as sns # Seaborn no se está usando actualmente
import base64
from io import BytesIO
from django.shortcuts import render
# import math # math no se está usando actualmente
from pymongo import MongoClient
import os
from matplotlib.colors import ListedColormap # Importar ListedColormap

# --- Constantes MongoDB (mejor si vienen de settings.py) ---
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/') # Default a local si no está en env
DB_NAME = 'malware_data'
COLLECTION_NAME = 'datasets'
# -----------------------------------------------------------------

@api_view(['GET'])
def test_api(request):
    """
    Endpoint de prueba para verificar que la API está funcionando
    """
    return Response({
        'message': 'API de Random Forest funcionando correctamente',
        'status': 'online',
        'endpoints': {
            'test': '/api/test/',
            'predict': '/api/predict/ (POST con dataset CSV y train_size)',
        }
    })

def image_to_base64(fig):
    """
    Convierte una figura de matplotlib a base64 para enviar en JSON
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig) # Cierra la figura para liberar memoria
    return f"data:image/png;base64,{image_base64}"

def remove_labels(df, label_name):
    """
    Separa features y etiquetas
    """
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def predict_malware(request):
    """
    Endpoint principal para entrenar Random Forest y hacer predicciones.
    Lee datos de un CSV, los guarda en MongoDB, los lee de MongoDB,
    entrena el modelo y devuelve métricas e imágenes.

    Parámetros:
    - dataset: archivo CSV
    - train_size: porcentaje de datos para entrenamiento (entero de 10 a 90)

    Retorna:
    - Resultados del análisis o un mensaje de error.
    """
    start_time = time.time()
    client = None # Inicializa el cliente fuera del try para el finally

    try:
        # --- Validaciones iniciales ---
        if 'dataset' not in request.FILES:
            return Response(
                {'success': False, 'error': 'No se proporcionó archivo CSV'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # train_size es un porcentaje (ej. 60)
            train_percent = int(request.data.get('train_size', 60))
            if not (10 <= train_percent <= 90):
                raise ValueError("El porcentaje de entrenamiento (train_size) debe estar entre 10 y 90.")
            # Calcular el test_size como fracción (ej. 60% train -> 0.4 test)
            test_size_fraction = 1.0 - (train_percent / 100.0)
        except (ValueError, TypeError) as e:
            return Response(
                {'success': False, 'error': f'train_size inválido: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        dataset_file = request.FILES['dataset']

        # 1. Conectar a MongoDB Atlas
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # Timeout de conexión
            # Forzar conexión para verificarla
            client.server_info()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            # Limpiar la colección antes de insertar nuevos datos
            collection.delete_many({})
        except Exception as e:
             return Response({'success': False, 'error': f"Error conectando a MongoDB: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 2. Leer CSV y Insertar en MongoDB
        try:
            df_upload = pd.read_csv(dataset_file)
            if df_upload.empty:
                raise ValueError("El archivo CSV está vacío.")
            if 'calss' not in df_upload.columns:
                 raise ValueError('El CSV debe tener una columna llamada "calss". Verifica la ortografía.')

            # Convertir a formato BSON (lista de diccionarios) y manejar tipos no serializables si es necesario
            df_upload = df_upload.replace([np.inf, -np.inf], np.nan) # Reemplazar infinitos antes de insertar
            # Considera convertir tipos específicos si causan problemas con BSON
            data_dict = df_upload.to_dict("records")
            collection.insert_many(data_dict)

        except Exception as e:
            return Response({'success': False, 'error': f"Error procesando o insertando CSV en MongoDB: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # 3. Leer Datos DESDE MongoDB para Entrenar
        try:
            cursor = collection.find({})
            df_from_db = pd.DataFrame(list(cursor))
            if '_id' in df_from_db.columns:
                df_from_db = df_from_db.drop('_id', axis=1)
            if df_from_db.empty:
                 raise ValueError("No se pudieron recuperar datos de MongoDB para el entrenamiento.")

        except Exception as e:
             return Response({'success': False, 'error': f"Error leyendo datos desde MongoDB: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


        # --- Procesamiento y Entrenamiento con df_from_db ---
        try:
            # Separar features y labels
            X, y = remove_labels(df_from_db, 'calss')

            # Manejar NaN/infinitos en X antes de dividir
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean()) # O usa otra estrategia de imputación

            # Dividir en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_fraction, random_state=42, stratify=y
            )

            # Escalar datos (después de dividir para evitar data leakage)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Entrenar RandomForestClassifier
            rf_model = RandomForestClassifier(
                n_estimators=10,  # Mantener reducido por velocidad
                max_depth=10,
                random_state=42,
                n_jobs=-1 # Usar todos los cores disponibles
            )
            rf_model.fit(X_train_scaled, y_train)

            # Predicciones
            y_pred = rf_model.predict(X_test_scaled)

            # Calcular métricas
            f1 = f1_score(y_test, y_pred, average='weighted')
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # Añadir zero_division

        except Exception as model_e:
             return Response({'success': False, 'error': f"Error durante el entrenamiento/predicción: {str(model_e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # --- Generación de Imágenes ---
        regression_tree_image = None
        decision_boundary_image = None
        try:
            required_cols = ['min_flowpktl', 'flow_fin']
            if not all(col in X_train.columns for col in required_cols):
                 print(f"Advertencia: Faltan las columnas {required_cols} para generar gráficas.")
            else:
                X_reduced_train = X_train[required_cols].copy()
                # Limpieza explícita
                X_reduced_train = X_reduced_train.apply(pd.to_numeric, errors='coerce')
                X_reduced_train = X_reduced_train.replace([np.inf, -np.inf], np.nan)
                X_reduced_train = X_reduced_train.fillna(X_reduced_train.mean())

                uniques = pd.unique(y_train)
                mapping = {val: i for i, val in enumerate(uniques)}
                y_train_reg = y_train.map(mapping)
                # Manejar posibles NaNs introducidos por el mapeo si y_train tuviera valores no en uniques
                y_train_reg = y_train_reg.fillna(-1).astype(int) # O alguna otra estrategia

                # Asegurarse de que no haya NaNs antes de entrenar el regresor
                if X_reduced_train.isnull().values.any() or y_train_reg.isnull().values.any():
                    raise ValueError("NaNs encontrados en datos reducidos o etiquetas mapeadas antes de entrenar RandomForestRegressor.")

                # Entrenar RandomForestRegressor
                rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_reg.fit(X_reduced_train.values, y_train_reg.values)

                # Generar imagen del árbol de regresión
                fig_reg, ax_reg = plt.subplots(figsize=(20, 12))
                plot_tree(
                    rf_reg.estimators_[0],
                    feature_names=required_cols,
                    filled=True, rounded=True, ax=ax_reg,
                    max_depth=5 # Limitar profundidad para visualización
                )
                ax_reg.set_title('Árbol de Regresión (Primer Estimador)', fontsize=16)
                regression_tree_image = image_to_base64(fig_reg)
                plt.close(fig_reg) # Cerrar explícitamente

                # Generar límite de decisión
                fig_db, ax_db = plt.subplots(figsize=(12, 6))
                mins = X_reduced_train.min(axis=0) - 1
                maxs = X_reduced_train.max(axis=0) + 1
                # Asegurarse de que mins y maxs sean finitos
                mins = mins.fillna(0)
                maxs = maxs.fillna(1)
                x1_linspace = np.linspace(mins['min_flowpktl'], maxs['min_flowpktl'], 100) # Reducir resolución
                x2_linspace = np.linspace(mins['flow_fin'], maxs['flow_fin'], 100)

                # Verificar si linspace generó NaNs (si min=max)
                if np.isnan(x1_linspace).any() or np.isnan(x2_linspace).any():
                     raise ValueError("Linspace generó NaN, posiblemente min=max en alguna característica reducida.")

                x1, x2 = np.meshgrid(x1_linspace, x2_linspace)
                X_new = np.c_[x1.ravel(), x2.ravel()]

                # Verificar NaNs en X_new
                if np.isnan(X_new).any():
                    raise ValueError("NaNs encontrados en la malla (X_new) antes de predecir.")

                y_mesh = rf_reg.predict(X_new)
                y_mesh_round = np.round(y_mesh).astype(int)
                n_classes = len(uniques)
                y_mesh_round = np.clip(y_mesh_round, 0, n_classes - 1)
                y_mesh_round = y_mesh_round.reshape(x1.shape)

                custom_cmap = ListedColormap(['#FAFAB0', '#9898FF', '#A0FAA0'][:n_classes])
                ax_db.contourf(x1, x2, y_mesh_round, cmap=custom_cmap, alpha=0.5)

                markers = ['o', 's', '^', 'v', 'D']
                colors = ['y', 'b', 'g', 'r', 'm']
                y_train_mapped = y_train.map(mapping).values # Calcular y_train_mapped

                for i, class_val in enumerate(uniques):
                    idx = y_train_mapped == i
                    # Asegurar que los datos para plotear sean finitos
                    plot_data_x = X_reduced_train.values[idx, 0]
                    plot_data_y = X_reduced_train.values[idx, 1]
                    finite_mask = np.isfinite(plot_data_x) & np.isfinite(plot_data_y)
                    ax_db.plot(plot_data_x[finite_mask], plot_data_y[finite_mask],
                               color=colors[i % len(colors)], marker=markers[i % len(markers)],
                               linestyle='none', label=str(class_val))

                ax_db.set_xlabel('min_flowpktl', fontsize=14)
                ax_db.set_ylabel('flow_fin', fontsize=14)
                ax_db.set_title('Límite de Decisión (Regresión)', fontsize=14)
                ax_db.legend()
                decision_boundary_image = image_to_base64(fig_db)
                plt.close(fig_db) # Cerrar explícitamente

        except ValueError as ve:
             print(f"Error de valor durante la generación de imágenes: {ve}")
             # No retornar error, solo dejar imágenes como None
        except Exception as img_e:
            print(f'Error inesperado generando imágenes: {img_e}')
            # No retornar error, solo dejar imágenes como None


        # --- Preparar Respuesta Final ---
        execution_time = time.time() - start_time
        response_data = {
            'success': True,
            'f1_score': round(float(f1), 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'classes': [str(c) for c in rf_model.classes_], # Asegurar que sean strings
            'classification_report': class_report,
            'regression_tree_image': regression_tree_image,
            'decision_boundary_image': decision_boundary_image,
            'execution_time': round(execution_time, 2)
        }
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        # Captura cualquier error no manejado específicamente
        error_traceback = traceback.format_exc()
        print(f"Error general en predict_malware: {error_traceback}")
        # Considera no enviar 'details' en producción a menos que sea necesario para debug
        return Response(
            {
                'success': False,
                'error': f'Ocurrió un error inesperado: {str(e)}',
                'details': error_traceback if request.data.get('debug') else None # Opcional: añadir flag debug
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    finally:
        # Asegurarse de cerrar la conexión a MongoDB
        if client:
            client.close()
            print("Conexión a MongoDB cerrada.")

# --- Vista para la página principal ---
def home(request):
    """Renderiza la página HTML principal."""
    return render(request, 'index.html')