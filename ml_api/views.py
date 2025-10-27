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
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from django.shortcuts import render
import math  # { changed code }


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
    plt.close(fig)
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
    Endpoint principal para entrenar Random Forest y hacer predicciones
    
    Parámetros:
    - dataset: archivo CSV
    - train_size: porcentaje de datos para entrenamiento (entero de 10 a 90)
    
    Retorna:
    - f1_score: métrica F1
    - tree_image: imagen del primer árbol (base64)
    - decision_boundary_image: límite de decisión (base64)
    - classification_report: reporte detallado
    """
    start_time = time.time()
    
    try:
        # Validar que se recibió el archivo
        if 'dataset' not in request.FILES:
            return Response(
                {'error': 'No se proporcionó archivo CSV'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validar train_size (ahora es porcentaje)
        try:
            # Ahora train_size es un porcentaje (ej. 60)
            train_percent = int(request.data.get('train_size', 60)) 
            if not (10 <= train_percent <= 90):
                raise ValueError("El porcentaje de entrenamiento (train_size) debe estar entre 10 y 90.")
            
            # Calcular el test_size como fracción (ej. 60% train -> 0.4 test)
            test_size_fraction = 1.0 - (train_percent / 100.0)

        except ValueError as e:
            return Response(
                {'error': f'train_size inválido: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Leer el archivo CSV
        dataset_file = request.FILES['dataset']
        try:
            df = pd.read_csv(dataset_file)
        except Exception as e:
            return Response(
                {'error': f'Error al leer CSV: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validar que el CSV tenga datos
        if len(df) == 0:
            return Response(
                {'error': 'El archivo CSV está vacío'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validar que tenga la columna de clase
        if 'calss' not in df.columns:
            return Response(
                {'error': 'El CSV debe tener una columna llamada "calss"'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Ya no limitamos por tamaño fijo, usamos todo el dataset con el porcentaje
        
        # Separar features y labels
        X, y = remove_labels(df, 'calss')
        
        # Dividir en train/test usando la fracción calculada (ej. 0.4 para 60%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_fraction, random_state=42, stratify=y
        )
        
        # Escalar datos
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=10,  # Reducido para velocidad
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = rf_model.predict(X_test_scaled)
        
        # Calcular métricas
        f1 = f1_score(y_test, y_pred, average='weighted')
        # La precisión (accuracy) se elimina del cálculo final
        acc = accuracy_score(y_test, y_pred) # Se mantiene temporalmente para la lógica si fuera necesario
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Generar imagen del árbol de regresión completo (usando las 2 características reducidas)
        try:
            X_reduced = X[['min_flowpktl', 'flow_fin']].copy()
            X_reduced = X_reduced.apply(pd.to_numeric, errors='coerce')
            X_reduced = X_reduced.replace([np.inf, -np.inf], np.nan)
            X_reduced = X_reduced.fillna(X_reduced.mean())

            # Preparar conjuntos reducidos usando los mismos índices que X_train/X_test
            X_reduced_train = X_train[['min_flowpktl', 'flow_fin']].copy()
            X_reduced_train = X_reduced_train.apply(pd.to_numeric, errors='coerce')
            X_reduced_train = X_reduced_train.replace([np.inf, -np.inf], np.nan)
            X_reduced_train = X_reduced_train.fillna(X_reduced_train.mean())

            # Codificar las etiquetas a enteros para entrenar el regresor
            uniques = pd.unique(y)
            mapping = {val: i for i, val in enumerate(uniques)}
            y_train_reg = y_train.map(mapping)

            # Entrenar RandomForestRegressor sobre las dos características (sin escalado para mantener las unidades originales)
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_reg.fit(X_reduced_train.values, y_train_reg.values)

            # Generar la imagen del árbol completo (primer estimador del regresor)
            fig_reg, ax_reg = plt.subplots(figsize=(20, 12))
            plot_tree(
                rf_reg.estimators_[0],
                feature_names=['min_flowpktl', 'flow_fin'],
                filled=True,
                rounded=True,
                ax=ax_reg,
            )
            ax_reg.set_title('Árbol de Regresión (completo)', fontsize=16)
            regression_tree_image = image_to_base64(fig_reg)

            # Generar límite de decisión similar al notebook
            fig_db, ax_db = plt.subplots(figsize=(12, 6))
            mins = X_reduced_train.min(axis=0) - 1
            maxs = X_reduced_train.max(axis=0) + 1
            x1, x2 = np.meshgrid(
                np.linspace(mins['min_flowpktl'], maxs['min_flowpktl'], 500),
                np.linspace(mins['flow_fin'], maxs['flow_fin'], 500)
            )
            X_new = np.c_[x1.ravel(), x2.ravel()]

            # Predecir (regresión) y discretizar para colorear regiones por clase
            y_mesh = rf_reg.predict(X_new)
            # Redondear y asegurar índices válidos
            y_mesh_round = np.round(y_mesh).astype(int)
            # Mapear valores para que estén dentro del rango de clases
            n_classes = len(uniques)
            y_mesh_round = np.clip(y_mesh_round, 0, n_classes - 1)
            y_mesh_round = y_mesh_round.reshape(x1.shape)

            from matplotlib.colors import ListedColormap
            custom_cmap = ListedColormap(['#FAFAB0', '#9898FF', '#A0FAA0'][:n_classes])
            ax_db.contourf(x1, x2, y_mesh_round, cmap=custom_cmap, alpha=0.5)

            # Graficar puntos de entrenamiento con sus etiquetas mapeadas
            y_train_mapped = y_train.map(mapping).values
            ax_db.plot(X_reduced_train.values[y_train_mapped==0, 0], X_reduced_train.values[y_train_mapped==0, 1], 'yo', label=str(uniques[0]) if len(uniques)>0 else '0')
            if n_classes>1:
                ax_db.plot(X_reduced_train.values[y_train_mapped==1, 0], X_reduced_train.values[y_train_mapped==1, 1], 'bs', label=str(uniques[1]))
            if n_classes>2:
                ax_db.plot(X_reduced_train.values[y_train_mapped==2, 0], X_reduced_train.values[y_train_mapped==2, 1], 'g^', label=str(uniques[2]))

            ax_db.set_xlabel('min_flowpktl', fontsize=14)
            ax_db.set_ylabel('flow_fin', fontsize=14)
            ax_db.set_title('Límite de Decisión (Regresión → clases discretizadas)', fontsize=14)
            ax_db.legend()
            decision_boundary_image = image_to_base64(fig_db)
        except Exception as e:
            print('Error generando imágenes de regresión/decisión:', e)
            regression_tree_image = None
            decision_boundary_image = None

        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'f1_score': round(float(f1), 4),
            # 'accuracy': round(float(acc), 4), # Eliminado de la respuesta
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'classes': list(rf_model.classes_),
            'classification_report': class_report,
            'regression_tree_image': regression_tree_image,
            'decision_boundary_image': decision_boundary_image,
            'execution_time': round(execution_time, 2)
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        # Log del error completo
        error_traceback = traceback.format_exc()
        print(f"Error en predict_malware: {error_traceback}")
        
        return Response(
            {
                'success': False,
                'error': str(e),
                'details': error_traceback if request.data.get('debug') else None
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
def home(request):
    return render(request, 'index.html')