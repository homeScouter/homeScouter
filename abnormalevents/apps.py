# abnormalevents/apps.py

import os
from django.apps import AppConfig
import firebase_admin
from firebase_admin import credentials, firestore
from django.conf import settings  # settings를 임포트하여 설정값 사용
import logging  # 로깅 추가

logger = logging.getLogger(__name__)

# 전역 Firestore 클라이언트 변수 (초기화 전에는 None)
db = None


class AbnormaleventsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'abnormalevents'
    verbose_name = 'Abnormal Events'

    def ready(self):
        # Django 관리 명령어 (runserver, migrate 등)가 실행될 때마다 ready()가 호출됩니다.
        # 따라서 스레드가 중복 실행되지 않도록, runserver 명령어일 때만 실행하는 것이 중요합니다.
        if os.environ.get('RUN_MAIN') == 'true' or os.environ.get('DJANGO_AUTORELOAD_MODE') == 'development':
            logger.info("Django AppConfig ready() for abnormalevents. Attempting Firebase initialization.")

            # Firebase Admin SDK가 이미 초기화되었는지 확인하여 중복 초기화 방지
            if not firebase_admin._apps:
                cred_path = settings.FIREBASE_CREDENTIAL_PATH  # settings.py에서 경로 가져옴

                if not cred_path:
                    logger.critical("❌ 오류: 환경변수 GOOGLE_APPLICATION_CREDENTIALS가 설정되어 있지 않습니다. Firebase 초기화 실패.")
                    return  # 초기화 실패 시 함수 종료

                if not os.path.isfile(cred_path):
                    logger.critical(f"❌ 오류: 서비스 계정 키 파일을 찾을 수 없습니다: {cred_path}. Firebase 초기화 실패.")
                    return  # 초기화 실패 시 함수 종료

                try:
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': settings.FIREBASE_STORAGE_BUCKET,  # settings.py에서 버킷 이름 가져옴
                        # 'projectId': 'home-scouter-50835', # storageBucket이 설정되면 projectID는 필수는 아니지만, 명시적 설정 가능
                    })
                    logger.info("✅ Firebase Admin SDK 초기화 완료.")
                except Exception as e:
                    logger.critical(f"❌ Firebase Admin SDK 초기화 중 오류 발생: {e}", exc_info=True)
                    return  # 초기화 실패 시 함수 종료
            else:
                logger.info("Firebase Admin SDK는 이미 초기화되어 있습니다.")

            # Firestore 클라이언트 인스턴스를 전역 db 변수에 할당
            global db
            db = firestore.client()
            logger.info("✅ Firestore client initialized.")

            # --- (선택 사항) StreamProcessor 스레드 시작 로직 ---
            # 만약 StreamProcessor 스레드를 Django와 함께 실행하려면 이 부분을 활성화합니다.
            # 이 로직은 `abnormalevents/stream_processor.py` 파일이 있다고 가정합니다.
            # try:
            #     from . import stream_processor
            #     stream_processor.set_db_client(db) # Firestore 클라이언트 전달
            #     # stream_processor_thread는 이곳에 정의되어야 함
            #     global stream_processor_thread
            #     if stream_processor_thread is None or not stream_processor_thread.is_alive():
            #         logger.info("Starting RTSP StreamProcessor thread...")
            #         stream_processor_thread = stream_processor.StreamProcessor(
            #             rtsp_url=settings.RTSP_URL,
            #             interval_sec=settings.CAPTURE_INTERVAL,
            #             num_frames_per_interval=16
            #         )
            #         stream_processor_thread.daemon = True
            #         stream_processor_thread.start()
            #         logger.info("RTSP StreamProcessor thread started.")
            #     else:
            #         logger.info("RTSP StreamProcessor thread is already running.")
            # except ImportError:
            #     logger.warning("StreamProcessor module not found. Skipping RTSP stream processing.")
            # except Exception as e:
            #     logger.critical(f"Failed to start StreamProcessor thread: {e}", exc_info=True)
        else:
            logger.info(
                "Skipping Firebase initialization/StreamProcessor thread start (not in RUN_MAIN/development mode).")

