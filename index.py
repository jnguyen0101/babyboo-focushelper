import cv2
import mediapipe as mp
import sounddevice as sd
import soundfile as sf
import time

# Define the playlist
playlist = [
    {"video": "media/triggering_breakup.mov", "audio": "media/triggering_breakup.wav"},
    {"video": "media/pinky_up.mov", "audio": "media/pinky_up.wav"},
    {"video": "media/squat.mov", "audio": "media/squat.wav"}
]

current_idx = 0
audio_data, fs = None, None
video_fps = 30.0
distraction_start_time = 0

def load_media(idx):
    global audio_data, fs, video_fps
    item = playlist[idx]
    
    # Load audio
    audio_data, fs = sf.read(item["audio"], dtype='float32')
    
    # Load video and get native FPS
    v_cap = cv2.VideoCapture(item["video"])
    video_fps = v_cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30.0
    
    return v_cap

video_cap = load_media(current_idx)
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_window_name = "GET BACK TO WORK"
audio_playing = False
was_looking_away = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    looking_away = True 
    if results.multi_face_landmarks:
        nose = results.multi_face_landmarks[0].landmark[4]
        if 0.35 < nose.x < 0.65 and 0.35 < nose.y < 0.65:
            looking_away = False

    # Just looked away
    if looking_away and not was_looking_away:
        distraction_start_time = time.time()
        sd.play(audio_data, fs, loop=True)
        audio_playing = True

    # While looking away
    if looking_away:
        # Calculate exactly which frame we should be on based on elapsed time
        elapsed_time = time.time() - distraction_start_time
        target_frame = int(elapsed_time * video_fps)

        # Jump to that specific frame
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, v_frame = video_cap.read()
        
        # If video ends, loop it
        if not ret:
            distraction_start_time = time.time()
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, v_frame = video_cap.read()

        if ret:
            v_frame = cv2.resize(v_frame, (360, 640)) 
            cv2.imshow(video_window_name, v_frame)
            
    # Stopped looking away
    else:
        if audio_playing:
            sd.stop()
            audio_playing = False

            # Increment playlist for the next time you look away
            current_idx = (current_idx + 1) % len(playlist)
            video_cap.release()
            video_cap = load_media(current_idx)
            try: cv2.destroyWindow(video_window_name)
            except: pass

    was_looking_away = looking_away

    # Status UI
    status_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(frame, f"STATUS: {'DISTRACTED' if looking_away else 'FOCUSED'}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Webcam Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
video_cap.release()
sd.stop()
cv2.destroyAllWindows()