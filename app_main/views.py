# app_main/views.py

import json # JSON 파싱 및 생성에 필요 (DRF 사용 시 request.data가 자동 파싱하지만, 직접 파싱할 때 대비)
import asyncio # 비동기 서비스 함수 호출을 위해 필요

# Django Rest Framework 관련 임포트
from rest_framework.decorators import api_view # API 뷰 데코레이터
from rest_framework.response import Response # DRF 응답 객체
from rest_framework import status # HTTP 상태 코드

# 기존 카메라 스트리밍 관련 임포트
from django.http import HttpResponse, JsonResponse
from camera_streamer.capture_api import capture_single_frame

# 우리가 만든 Firebase 서비스 클래스 임포트
from .firestore_service import UserAuthService

# UserAuthService 인스턴스 생성 (모듈 레벨에서 한 번만 생성)
# 이 인스턴스는 두 뷰에서 모두 공유하여 사용할 수 있습니다.
user_auth_service = UserAuthService()

# --- 기존 get_latest_frame 뷰 ---
@api_view(['GET'])
def get_latest_frame(request):
    """
    카메라에서 최신 프레임을 캡처하여 JPEG 이미지로 반환합니다.
    """
    jpeg_bytes = capture_single_frame()
    if jpeg_bytes is None:
        # 프레임 캡처 실패 시 JSON 오류 응답
        return JsonResponse({'error': 'Failed to capture frame'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 캡처 성공 시 JPEG 이미지 바이트를 HTTP 응답으로 반환
    return HttpResponse(jpeg_bytes, content_type='image/jpeg')

# --- 새로 추가하는 signup_api 뷰 ---
@api_view(['POST']) # 이 뷰는 POST 요청만 처리하도록 설정합니다.
def signup_api(request):
    """
    사용자 회원가입 API 엔드포인트.
    Flutter 앱에서 사용자 정보를 받아 Firebase Authentication에 계정을 생성하고
    Firestore에 추가 프로필 정보를 저장합니다.
    """
    try:
        # Django Rest Framework의 request.data는 POST 요청의 JSON 본문을 자동으로 파싱합니다.
        name = request.data.get('name')
        email = request.data.get('id') # Flutter에서 'id'로 보냈지만, Firebase Auth의 email로 사용
        password = request.data.get('password')
        phone = request.data.get('phone')
        tapo_code = request.data.get('tapoCode') # Flutter에서 보낸 TAPO 기기 코드

        # 입력 데이터의 기본적인 유효성 검사
        if not all([name, email, password, phone, tapo_code]):
            # 필요한 모든 필드가 제공되지 않은 경우 에러 응답
            return Response(
                {'message': '이름, 아이디(이메일), 비밀번호, 전화번호, TAPO 기기 코드는 필수 입력 항목입니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # UserAuthService의 비동기 함수 호출
        # Django 뷰가 동기식인 경우, asyncio.run()을 사용하여 비동기 함수를 실행합니다.
        # 주의: 이 방법은 짧고 독립적인 비동기 작업에 적합하며,
        # 긴 비동기 작업을 처리하는 경우에는 Django의 async view (async def)를 사용하는 것이 더 효율적입니다.
        user_data = asyncio.run(user_auth_service.signup_user(name, email, password, phone, tapo_code))

        # 회원가입 성공 시 201 Created 응답 반환
        return Response(
            {
                'message': '회원가입이 성공적으로 완료되었습니다.',
                'uid': user_data['uid'], # Firebase Auth에서 생성된 UID
                'email': user_data['email'] # 등록된 이메일
            },
            status=status.HTTP_201_CREATED
        )

    except ValueError as e:
        # firestore_service.py에서 정의된 사용자 정의 에러 처리
        error_code = str(e)
        if error_code == "email-already-in-use":
            return Response(
                {'message': '이미 사용 중인 이메일(아이디)입니다. 다른 아이디를 사용해주세요.'},
                status=status.HTTP_409_CONFLICT # 409 Conflict: 요청 충돌
            )
        elif error_code == "invalid-email":
            return Response(
                {'message': '유효하지 않은 이메일(아이디) 형식입니다. 올바른 이메일 주소를 입력해주세요.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        elif error_code == "weak-password":
            return Response(
                {'message': '비밀번호가 너무 약합니다. 최소 6자 이상으로 설정해주세요.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        else:
            # 알 수 없는 사용자 정의 에러
            print(f"알 수 없는 회원가입 관련 오류: {e}")
            return Response(
                {'message': '회원가입 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    except Exception as e:
        # Django 서버 내부에서 발생한 예상치 못한 오류
        print(f"서버 내부 오류 발생: {e}")
        return Response(
            {'message': '서버 처리 중 오류가 발생했습니다. 관리자에게 문의해주세요.'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

