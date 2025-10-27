from django.urls import path
from . import views

urlpatterns = [
    path('api/test/', views.test_api, name='test_api'),
    path('api/predict/', views.predict_malware, name='predict_malware'),
    path('', views.home, name='home'),
]