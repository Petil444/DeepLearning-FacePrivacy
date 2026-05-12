import cv2
import face_recognition

# 웹캠 시작
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    # BGR → RGB 변환
    rgb_frame = frame[:, :, ::-1]

    # 얼굴 위치 찾기
    face_locations = face_recognition.face_locations(rgb_frame)

    # 얼굴 박스 그리기
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Webcam Face Recognition", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows() 