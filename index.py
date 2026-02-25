import cv2
import mediapipe as mp
import os
import sounddevice as sd
import soundfile as sf

audio_path = "triggering_breakup.wav" 
data, fs = sf.read(audio_path, dtype='float32')

def play_audio():
    sd.play(data, fs, loop=True)

def stop_audio():
    sd.stop()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path = "triggering_breakup.mov"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found")
    exit()

video_cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

video_window_name = "GET BACK TO WORK"
window_open = False
audio_playing = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    looking_away = True 

    if results.multi_face_landmarks:
        # Landmark 4 is the nose tip
        nose = results.multi_face_landmarks[0].landmark[4]
        if 0.35 < nose.x < 0.65 and 0.35 < nose.y < 0.65:
            looking_away = False

    if looking_away:
        # Toggle audio ON
        if not audio_playing:
            play_audio()
            audio_playing = True

        ret, v_frame = video_cap.read()
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, v_frame = video_cap.read()

        if ret and v_frame is not None:
            v_frame = cv2.resize(v_frame, (360, 640)) 
            cv2.imshow(video_window_name, v_frame)
            window_open = True
    else:
        # Toggle audio OFF
        if audio_playing:
            stop_audio()
            audio_playing = False

        if window_open:
            try:
                cv2.destroyWindow(video_window_name)
                window_open = False
            except:
                pass

    # Status UI
    status_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(frame, f"STATUS: {'AWAY' if looking_away else 'DISTRACTED'}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Webcam Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_cap.release()
sd.stop()
cv2.destroyAllWindows()