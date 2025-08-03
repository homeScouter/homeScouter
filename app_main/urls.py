# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('latest-frame/', views.get_latest_frame, name='latest_frame'),
]
