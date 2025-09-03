# config/urls.py

from django.urls import path, include

urlpatterns = [
    path('api/', include('app_main.urls')),  # app_main 앱의 urls.py에서 정의된 경로를 포함
    path('api/', include('abnormalevents.urls')),
    path('camera_streamer/', include('camera_streamer.urls')), # ai모델 돌리는 url
]