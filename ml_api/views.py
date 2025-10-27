from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import time
import traceback
from sklearn.ensemble import RandomForestClassifier
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
    - train_size: cantidad de datos para entrenamiento (entero)
    
    Retorna:
    - f1_score: métrica F1
    - accuracy: precisión del modelo
    - tree_image: imagen del primer árbol (base64)
    - confusion_matrix_image: matriz de confusión (base64)
    - feature_importance_image: importancia de características (base64)
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
        
        # Validar train_size
        try:
            train_size = int(request.data.get('train_size', 1000))
            if train_size <= 0:
                raise ValueError("train_size debe ser mayor a 0")
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
        
        # Limitar el dataset si se especificó
        if train_size < len(df):
            # Mantener balance de clases
            df_sample = df.groupby('calss', group_keys=False).apply(
                lambda x: x.sample(min(len(x), train_size // df['calss'].nunique()))
            )
            df = df_sample
        
        # Separar features y labels
        X, y = remove_labels(df, 'calss')
        
        # Dividir en train/test (60/40)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
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
        acc = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Generar imagen del primer árbol
        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
        plot_tree(
            rf_model.estimators_[0],
            feature_names=X.columns,
            class_names=rf_model.classes_,
            filled=True,
            rounded=True,
            ax=ax_tree,
            max_depth=3  # Limitar profundidad para visualización
        )
        ax_tree.set_title('Primer Árbol del Random Forest', fontsize=16)
        tree_image_base64 = image_to_base64(fig_tree)
        
        # Generar matriz de confusión
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=rf_model.classes_,
            yticklabels=rf_model.classes_,
            ax=ax_cm
        )
        ax_cm.set_title('Matriz de Confusión', fontsize=14)
        ax_cm.set_xlabel('Predicción', fontsize=12)
        ax_cm.set_ylabel('Real', fontsize=12)
        cm_image_base64 = image_to_base64(fig_cm)
        
        # Generar gráfica de importancia de características
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        fig_fi, ax_fi = plt.subplots(figsize=(12, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax_fi)
        ax_fi.set_title('Top 20 Características Más Importantes', fontsize=14)
        ax_fi.set_xlabel('Importancia', fontsize=12)
        ax_fi.set_ylabel('Característica', fontsize=12)
        fi_image_base64 = image_to_base64(fig_fi)
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'f1_score': round(float(f1), 4),
            'accuracy': round(float(acc), 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'classes': list(rf_model.classes_),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'tree_image': tree_image_base64,
            'confusion_matrix_image': cm_image_base64,
            'feature_importance_image': fi_image_base64,
            'top_features': feature_importance.to_dict('records'),
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