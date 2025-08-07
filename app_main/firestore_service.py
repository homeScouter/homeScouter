from config.settings import db
from firebase_admin import auth
from firebase_admin import firestore
from firebase_admin import exceptions as fb_exceptions

class UserAuthService:
    def __init__(self):
        # Firestore의 'users' 컬렉션에 대한 참조를 미리 얻어둡니다.
        self.users_collection = db.collection('users')

    async def signup_user(self, name: str, email: str, password: str, phone: str, tapo_code: str) -> dict:
        """
        Firebase Authentication으로 사용자 계정을 생성하고,
        성공적으로 생성되면 Firestore에 사용자 프로필 데이터를 저장합니다.

        Args:
            name (str): 사용자의 이름 (표시 이름)
            email (str): 사용자의 이메일 (Firebase Authentication ID로 사용)
            password (str): 사용자의 비밀번호
            phone (str): 사용자의 전화번호
            tapo_code (str): TAPO 기기 코드

        Returns:
            dict: 생성된 사용자 계정의 UID, 이메일, 이름, 전화번호, TAPO 코드를 포함하는 딕셔너리.

        Raises:
            ValueError: 'email-already-in-use', 'invalid-email', 'weak-password', 'unknown-error' 등의 에러 코드.
        """
        try:
            # 1. Firebase Authentication에 새로운 사용자 계정 생성
            # email이 곧 로그인 ID 역할을 합니다.
            user_record = auth.create_user(
                email=email,
                password=password,
                display_name=name,  # Firebase Auth의 표시 이름 필드
                # email_verified=False, # 필요에 따라 이메일 인증 플래그 설정
            )
            # Firebase Auth에서 생성된 사용자의 고유 ID (UID)
            uid = user_record.uid

            print(f"Firebase Auth에 사용자 계정 생성됨. UID: {uid}, Email: {email}")

            # 2. Firestore에 사용자 프로필 데이터 저장
            # 문서 ID로 Firebase Auth에서 받은 UID를 사용하여, 인증과 프로필을 연결합니다.
            user_profile_data = {
                'name': name,
                'email': email,
                'phone': phone,
                'tapoCode': tapo_code,
                'createdAt': firestore.SERVER_TIMESTAMP,  # 서버 타임스탬프 (생성 시각)
                'updatedAt': firestore.SERVER_TIMESTAMP   # 서버 타임스탬프 (업데이트 시각)
            }

            # Firestore의 'users' 컬렉션에 UID를 문서 ID로 하여 프로필 데이터 저장
            self.users_collection.document(uid).set(user_profile_data)
            print(f"Firestore에 사용자 프로필 저장됨. UID: {uid}")

            # 성공 시 클라이언트에 전달할 정보 반환
            return {
                'uid': uid,
                'email': email,
                'name': name,
                'phone': phone,
                'tapoCode': tapo_code
            }

        except fb_exceptions.FirebaseError as e:
            error_message = str(e)
            print(f"FirebaseError 발생: {error_message}")

            # 에러 메시지 내 키워드 기준으로 세부 에러 분기
            if 'email-already-exists' in error_message:
                print(f"회원가입 실패: {email}은 이미 존재하는 이메일입니다.")
                raise ValueError("email-already-in-use")
            elif 'INVALID_EMAIL' in error_message or 'invalid email' in error_message.lower():
                print(f"회원가입 실패: {email}은 유효하지 않은 이메일 형식입니다.")
                raise ValueError("invalid-email")
            elif 'WEAK_PASSWORD' in error_message or 'password should be at least' in error_message.lower():
                print("회원가입 실패: 비밀번호가 너무 약합니다.")
                raise ValueError("weak-password")
            else:
                print(f"회원가입 중 알 수 없는 FirebaseError 발생: {error_message}")
                raise ValueError("unknown-error")

        except Exception as e:
            # 위에 명시되지 않은 다른 모든 예외 처리
            print(f"회원가입 중 알 수 없는 오류 발생: {e}")
            raise ValueError("unknown-error")

    async def get_user_profile(self, uid: str) -> dict | None:
        """
        주어진 UID로 Firestore에서 사용자 프로필 데이터를 가져옵니다.
        """
        doc = self.users_collection.document(uid).get()
        if doc.exists:
            profile_data = doc.to_dict()
            profile_data['uid'] = doc.id  # 문서 ID (UID)도 데이터에 포함시켜 반환
            return profile_data
        return None

    # async def update_user_profile(self, uid: str, update_data: dict):
    #     """
    #     주어진 UID의 사용자 프로필을 업데이트합니다.
    #     """
    #     update_data['updatedAt'] = firestore.SERVER_TIMESTAMP
    #     self.users_collection.document(uid).update(update_data)
    #     print(f"User {uid} profile updated.")

    # async def delete_user_and_profile(self, uid: str):
    #     """
    #     Firebase Authentication에서 사용자를 삭제하고, Firestore 프로필도 삭제합니다.
    #     """
    #     auth.delete_user(uid)  # Firebase Auth 사용자 삭제
    #     self.users_collection.document(uid).delete()  # Firestore 프로필 삭제
    #     print(f"User {uid} and their profile deleted.")
