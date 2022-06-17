from django.urls import path
from . import views

urlpatterns=[
    path('', views.home, name="text-home"),
    path('test/', views.test, name="text-prediction")
]
