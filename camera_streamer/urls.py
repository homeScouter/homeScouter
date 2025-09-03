# camera_streamer/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('run-model/', views.run_model, name='run_model'),
    path('stop-model/', views.stop_model, name='stop_model'),
]
