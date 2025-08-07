from django.apps import AppConfig

# 장고 설정 파일 기본값으로 사용

class AppMainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app_main'
