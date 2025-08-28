# abnormalevents/views.py

import logging
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from datetime import datetime, timedelta
import pytz

# Firebase Admin SDK 관련 임포트 (views.py에서도 필요)
import firebase_admin
from firebase_admin import storage
from firebase_admin import firestore

# settings를 임포트하여 FIREBASE_STORAGE_BUCKET 값을 직접 사용
from django.conf import settings

# abnormalevents/apps.py에서 초기화된 Firestore 클라이언트 임포트
from .apps import db as firestore_db  # 여기서 db 객체를 가져옵니다.

logger = logging.getLogger(__name__)


@require_GET
def get_abnormal_events_api(request):
    # Firestore 클라이언트가 제대로 초기화되었는지 확인
    if firestore_db is None:
        logger.error("Firebase Admin SDK (Firestore client) is not initialized in apps.py. Check logs.")
        return JsonResponse({"error": "Server error: Firebase connection not ready."}, status=500)

    # Firebase Storage 버킷 참조 시 버킷 이름을 명시적으로 전달
    try:
        bucket = storage.bucket(name=settings.FIREBASE_STORAGE_BUCKET)
        logger.info(f"Successfully accessed Firebase Storage bucket: {settings.FIREBASE_STORAGE_BUCKET}")
    except Exception as e:
        logger.critical(f"Error accessing Firebase Storage bucket. Details: {e}", exc_info=True)
        return JsonResponse({"error": f"Server error: Could not access Storage bucket. Details: {e}"}, status=500)

    # 쿼리 파라미터 'limit' 처리
    limit = request.GET.get('limit', 20)
    try:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("Limit value must be a positive integer.")
    except ValueError as e:
        logger.warning(f"Invalid 'limit' parameter received: {request.GET.get('limit')} - {e}")
        return JsonResponse({"error": f"Invalid 'limit' parameter: {e}"}, status=400)

    # 쿼리 파라미터 'since' 처리
    since_dt_str = request.GET.get('since')
    since_datetime = None
    if since_dt_str:
        try:
            since_datetime = datetime.fromisoformat(since_dt_str)
            if since_datetime.tzinfo is None:
                since_datetime = pytz.utc.localize(since_datetime)
            else:
                since_datetime = since_datetime.astimezone(pytz.utc)
        except ValueError:
            logger.warning(f"Invalid 'since' parameter format: {since_dt_str}")
            return JsonResponse(
                {"error": "Invalid 'since' parameter format. Use ISO 8601 (e.g., 2023-10-26T10:00:00Z)."}, status=400)

    events_data_list = []
    try:
        # Firestore 클라이언트 (firestore_db)를 사용하여 쿼리
        query = firestore_db.collection('abnormal_events')

        if since_datetime:
            query = query.where('timestamp', '>', since_datetime)

        query = query.order_by('timestamp', direction='DESCENDING').limit(limit)

        docs = query.stream()

        # ⭐️⭐️⭐️ bucket.name이 정확히 무엇인지 확인합니다. ⭐️⭐️⭐️
        logger.debug(f"DEBUG VIEWS: Initialized bucket name for blob extraction: {bucket.name}")

        for doc in docs:
            event = doc.to_dict()
            event['id'] = doc.id

            if 'timestamp' in event and hasattr(event['timestamp'], 'isoformat'):
                event['timestamp'] = event['timestamp'].isoformat()

            # ⭐️ 이미지 URL 재성성 로직 ⭐️
            if 'image_url' in event and event['image_url']:
                try:
                    original_image_url = event['image_url']  # 원본 URL 저장
                    parsed_url_without_query = original_image_url.split('?')[0]  # 쿼리 파라미터 제거

                    # ⭐️⭐️⭐️ 이 변수들의 값을 정확히 확인해야 합니다. ⭐️⭐️⭐️
                    logger.debug(f"DEBUG VIEWS: Processing image_url for event {event['id']}: {original_image_url}")
                    logger.debug(f"DEBUG VIEWS: Parsed URL without query: {parsed_url_without_query}")
                    logger.debug(f"DEBUG VIEWS: Bucket name being searched for: {bucket.name}")

                    # bucket.name (예: "home-scouter-50835.firebasestorage.app") 뒤의 경로를 blob_name으로 간주합니다.
                    blob_name_start_index = parsed_url_without_query.find(bucket.name)
                    logger.debug(f"DEBUG VIEWS: Result of find (blob_name_start_index): {blob_name_start_index}")

                    if blob_name_start_index != -1:
                        # 버킷 이름과 그 뒤의 슬래시를 건너뛰어 실제 blob_name을 추출
                        blob_name_start_index += len(bucket.name) + 1
                        blob_name = parsed_url_without_query[blob_name_start_index:]

                        logger.debug(f"DEBUG VIEWS: Extracted blob_name: {blob_name}")

                        image_blob = bucket.blob(blob_name)
                        # 새로운 서명된 URL 생성 (만료 시간: 7일 = 604800초)
                        event['image_url'] = image_blob.generate_signed_url(version="v4",
                                                                            expiration=604800)  # <--- 여기를 수정했습니다.
                        logger.info(f"Regenerated image_url for {event['id']}.")
                    else:
                        logger.warning(
                            f"Could not extract valid blob name from image_url: {original_image_url} for event {event['id']}. Setting to None.")
                        event['image_url'] = None
                except Exception as e:
                    logger.error(f"Error regenerating image_url for event {event.get('id', 'N/A')}: {e}", exc_info=True)
                    event['image_url'] = None

            # ⭐️ 비디오 URL 재성성 로직 ⭐️
            if 'video_url' in event and event['video_url']:
                try:
                    original_video_url = event['video_url']
                    parsed_video_url_without_query = original_video_url.split('?')[0]

                    logger.debug(f"DEBUG VIEWS: Processing video_url for event {event['id']}: {original_video_url}")
                    logger.debug(f"DEBUG VIEWS: Parsed video URL without query: {parsed_video_url_without_query}")
                    logger.debug(f"DEBUG VIEWS: Bucket name being searched for: {bucket.name}")

                    blob_name_start_index = parsed_video_url_without_query.find(bucket.name)
                    logger.debug(f"DEBUG VIEWS: Result of find (video_blob_name_start_index): {blob_name_start_index}")

                    if blob_name_start_index != -1:
                        blob_name_start_index += len(bucket.name) + 1
                        video_blob_name = parsed_video_url_without_query[blob_name_start_index:]

                        logger.debug(f"DEBUG VIEWS: Extracted video_blob_name: {video_blob_name}")

                        video_blob = bucket.blob(video_blob_name)
                        # 비디오 URL도 새로운 서명된 URL로 재성성 (만료 시간: 7일 = 604800초)
                        event['video_url'] = video_blob.generate_signed_url(version="v4",
                                                                            expiration=604800)  # <--- 여기도 수정했습니다.
                        logger.info(f"Regenerated video_url for {event['id']}.")
                    else:
                        logger.warning(
                            f"Could not extract valid blob name from video_url: {original_video_url} for event {event['id']}. Setting to None.")
                        event['video_url'] = None
                except Exception as e:
                    logger.error(f"Error regenerating video_url for event {event.get('id', 'N/A')}: {e}", exc_info=True)
                    event['video_url'] = None

            events_data_list.append(event)

    except Exception as e:
        logger.exception("Error fetching abnormal events from Firestore.")
        return JsonResponse({"error": f"Failed to retrieve abnormal events: {e}"}, status=500)

    logger.info(f"Successfully retrieved {len(events_data_list)} abnormal events from Firestore.")
    return JsonResponse({"events": events_data_list}, status=200)

