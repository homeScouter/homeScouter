from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST  # POST 요청만 받도록 제한
import subprocess
import os

# 프로세스를 추적할 글로벌 변수
current_process = None

@csrf_exempt
@require_POST
def run_model(request):
    """
    Flutter에서 호출될 API
    'E:\\my_home\\camera_streamer\\realtime_check_save_firebase.py' 경로의 Python 스크립트를 실행하고 종료 처리
    """
    global current_process  # 현재 실행 중인 프로세스를 추적할 변수

    try:
        # 파일 경로 지정
        script_path = r'E:\my_home\camera_streamer\realtime_check_save_firebase.py'

        # 가상환경 Python 경로 (수정 부분)
        python_path = r'E:\my_home\.venv\Scripts\python.exe'

        # subprocess를 사용하여 외부 Python 스크립트 실행
        current_process = subprocess.Popen(
            [python_path, script_path],  # python 경로를 명시적으로 지정
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 프로세스 실행 결과 읽기
        stdout, stderr = current_process.communicate()

        # stderr의 인코딩 오류 처리
        stderr_message = ""
        try:
            stderr_message = stderr.decode("utf-8")
        except UnicodeDecodeError:
            stderr_message = stderr.decode("utf-8", errors="ignore")  # 오류를 무시하고 디코딩

        # 스크립트가 끝난 후, 종료 코드 확인
        if current_process.returncode == 0:
            return JsonResponse({"status": "success", "message": "모델 실행 완료"})
        else:
            return JsonResponse({"status": "error", "message": stderr_message})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})


@csrf_exempt
@require_POST
def stop_model(request):
    """
    실행 중인 프로세스를 종료하는 API (서버 종료)
    """
    global current_process  # 현재 실행 중인 프로세스를 추적할 변수

    try:
        if current_process is None or current_process.poll() is not None:
            return JsonResponse({"status": "error", "message": "실행 중인 프로세스가 없습니다."})

        # 프로세스를 종료
        current_process.terminate()
        current_process.wait()  # 종료될 때까지 기다림
        current_process = None  # 프로세스 정보 초기화

        return JsonResponse({"status": "success", "message": "모델 프로세스 종료 완료"})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})
