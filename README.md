# 🤖 API de Predicción con RandomForest

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Django](https://img.shields.io/badge/Django-4.x-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange)
![TailwindCSS](https://img.shields.io/badge/Tailwind-CSS-06B6D4)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248)

Un sistema de **entrenamiento y predicción de Machine Learning** basado en
el algoritmo **Random Forest**. Esta aplicación web permite a los usuarios
subir un dataset en formato CSV y entrenar un modelo para realizar
clasificaciones.

La aplicación genera visualizaciones del modelo, como el árbol de decisión y los límites de clasificación, y guarda un historial de los entrenamientos en **MongoDB** para acelerar futuras peticiones con los mismos datos.



## 🚀 Características Principales

-   **Modelo Dinámico:** Entrena un clasificador **RandomForest** con el dataset y el porcentaje de entrenamiento que el usuario especifique.
-   **Visualización de Resultados:** Genera y muestra gráficos en tiempo real:
    -   **Árbol de Regresión:** Visualiza la estructura de uno de los árboles del bosque.
    -   **Límite de Decisión:** Muestra cómo el modelo clasifica el espacio de características.
-   **Caché Inteligente:** Guarda los resultados de cada entrenamiento en **MongoDB** usando un hash del archivo. Si se sube el mismo archivo con la misma configuración, devuelve el resultado cacheado al instante.
-   **Métricas de Rendimiento:** Calcula y muestra el F1-Score, reporte de clasificación y tiempo de ejecución.
-   **Interfaz Moderna:** Frontend construido con **TailwindCSS** para una experiencia de usuario limpia.



## 🛠️ Stack Tecnológico

-   **Backend:** Django, Django REST Framework
-   **IA / ML:** Scikit-Learn, Pandas, NumPy
-   **Visualización:** Matplotlib, Seaborn
-   **Base de Datos:** MongoDB
-   **Frontend:** HTML5, TailwindCSS (CDN)
-   **Exposición:** Ngrok (opcional)



## 📂 Estructura del Proyecto (Simplificada)

```bash
API_RandomForest/
├── ml_api/
│   ├── templates/
│   │   └── index.html
│   ├── models/           # Carpeta para modelos cacheados
│   └── views.py          # Lógica principal de la API
├── core/                 # Configuración de Django
│   └── settings.py
├── manage.py
└── requirements.txt
```



## ⚙️ Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/IsmaelJrDev/API_RandomForest.git
cd API_RandomForest
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar MongoDB (Opcional)

Crea un archivo `.env` en la raíz del proyecto y añade tu cadena de conexión de MongoDB Atlas.

```
MONGO_URI="mongodb+srv://<usuario>:<password>@cluster..."
```

Si no se configura, la API funcionará pero no guardará los resultados en la base de datos.


## ▶️ Ejecución

### Modo Local

```bash
python manage.py runserver
```

Abre tu navegador en `http://127.0.0.1:8000/`.

### Modo Público (con Ngrok)

```bash
ngrok http 8000
```



## ⚠️ Nota

Esta herramienta es un prototipo con fines de demostración. El rendimiento y la precisión dependen enteramente de la calidad y características del dataset proporcionado.



## 👨‍💻 Autor

**IsmaelJrDev**
GitHub: https://github.com/IsmaelJrDev
