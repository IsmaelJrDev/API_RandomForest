from django.urls import path
from . import views

# Define el nombre de la aplicación para el enrutamiento de URLs
app_name = 'apiRF'

# Define las rutas URL y las asigna a las vistas correspondientes
urlpatterns = [
    path('', views.formulario, name='formulariorf'),
]