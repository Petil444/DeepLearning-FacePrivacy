import cv2
import face_recognition
import numpy as np
import faiss
import os
import time

# =========================
# 0. 화면 크기 설정
# =========================a

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# =========================
# 1. 얼굴 DB 로드
# =========================

known_face_names = []
known_face_encodings = []

faces_path = "faces"

for file in os.listdir(faces_path):

    img_path = os.path.join(faces_path, file)

    image = cv2.imread(img_path)

    if image is None:
        continue

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # uint8 보장
    image = np.array(image, dtype=np.uint8)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:

        encoding = encodings[0]

        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file)[0])

        print(f"등록 완료: {file}")

# =========================
# 2. FAISS 인덱스 생성
# =========================

dimension = 128

index = faiss.IndexFlatL2(dimension)

db_vectors = np.array(
    known_face_encodings,
    dtype=np.float32
)

index.add(db_vectors)

print("FAISS DB 구축 완료")

# =========================
# 3. 웹캠 시작
# =========================

video_capture = cv2.VideoCapture(0)

# 해상도 설정
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# 실제 적용된 해상도 확인
actual_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"현재 해상도: {actual_width} x {actual_height}")

# =========================
# FPS 계산용
# =========================

prev_time = 0

# =========================
# 4. 메인 루프
# =========================

while True:

    ret, frame = video_capture.read()

    if not ret:
        break

    # =========================
    # FPS 계산
    # =========================

    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    # =========================
    # RGB 변환
    # =========================

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.array(rgb_frame, dtype=np.uint8)

    # =========================
    # 얼굴 탐지 (HOG)
    # =========================

    face_locations = face_recognition.face_locations(
        rgb_frame,
        model="hog"
    )

    # =========================
    # 얼굴 인코딩
    # =========================

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    # =========================
    # 얼굴 비교
    # =========================

    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):

        query = np.array(
            [face_encoding],
            dtype=np.float32
        )

        # =========================
        # FAISS 검색
        # =========================

        distances, indices = index.search(query, 1)

        distance = distances[0][0]
        idx = indices[0][0]

        # =========================
        # threshold
        # =========================

        if distance < 0.45:

            name = known_face_names[idx]

            # 초록 박스
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0,255,0),
                2
            )

            # 이름 출력
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        else:

            # =========================
            # 블라인드 처리
            # =========================

            face = frame[top:bottom, left:right]

            if face.size != 0:

                face = cv2.GaussianBlur(
                    face,
                    (51,51),
                    30
                )

                frame[top:bottom, left:right] = face

    # =========================
    # FPS 출력
    # =========================

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # =========================
    # 해상도 출력
    # =========================

    cv2.putText(
        frame,
        f"SIZE : {actual_width} x {actual_height}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )

    # =========================
    # 화면 출력
    # =========================

    cv2.imshow("FAISS Face Recognition", frame)

    # q 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# 종료
# =========================

video_capture.release()
cv2.destroyAllWindows()