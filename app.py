import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import pyttsx3
import threading
import queue
import time
import os
import streamlit as st
from PIL import Image
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Initialize TTS engine
engine = pyttsx3.init()

# Sound files initialization
try:
    sound = pyglet.media.load("sound.wav", streaming=False)
    siren_sound = pyglet.media.load("siren.wav", streaming=False)
except:
    print("Warning: Some sound files not found. Audio feedback will be limited.")

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    st.error("Error: shape_predictor_68_face_landmarks.dat not found. Please download it.")
    st.stop()

class VisionBoardApp:
    def __init__(self):
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        # Initialize session state variables
        if 'text' not in st.session_state:
            st.session_state.text = ""
        if 'blink_pattern' not in st.session_state:
            st.session_state.blink_pattern = []
        if 'last_blink_time' not in st.session_state:
            st.session_state.last_blink_time = time.time()
        if 'blink_warning_shown' not in st.session_state:
            st.session_state.blink_warning_shown = False
        if 'no_face_frames' not in st.session_state:
            st.session_state.no_face_frames = 0
        if 'frames' not in st.session_state:
            st.session_state.frames = 0
        if 'blinking_frames' not in st.session_state:
            st.session_state.blinking_frames = 0
        if 'pattern_start_time' not in st.session_state:
            st.session_state.pattern_start_time = 0
        if 'cooldown_counter' not in st.session_state:
            st.session_state.cooldown_counter = 0
        
        # Constants
        self.frames_to_blink = 3
        self.cooldown_frames = 25
        self.max_no_face_frames = 150
        self.pattern_timeout = 3  # seconds to complete a pattern
        
        # Image names and paths
        self.image_paths = {
            "water": "water.jpeg",
            "washroom": "washroom.jpeg",
            "food": "food.jpeg",
            "emergency": "emergency.jpeg",
            "medicine": "medicine.jpeg",
            "clothing": "cloth.webp",
            "position": "stature.webp",
            "lights": "lights.jpeg",
            "environment": "un.png"
        }
        
        self.image_names = {
            "water": "Water",
            "washroom": "Washroom",
            "food": "Food",
            "emergency": "Emergency",
            "medicine": "Medicine",
            "clothing": "Change my clothes",
            "position": "Change my position",
            "lights": "Lights",
            "environment": "Unpleasant environment"
        }
        
        # Load images with error handling
        self.images = {}
        for img_key, img_path in self.image_paths.items():
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError
                self.images[img_key] = cv2.resize(img, (200, 200))
            except:
                st.warning(f"Image {img_path} not found. Using placeholder.")
                self.images[img_key] = np.zeros((200, 200, 3), np.uint8)
                cv2.putText(self.images[img_key], img_key, (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recognized blink patterns
        self.recognized_patterns = {
            '111': 'emergency',
            '11': 'water',
            '101': 'washroom',
            '1001': 'food'
        }
        
        # Setup Streamlit UI
        self.setup_ui()
    
    def setup_ui(self):
        st.title("Vision Board for Impaired Persons")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            self.voice_enabled = st.checkbox("Enable Voice Feedback", True)
            self.blink_detection = st.checkbox("Enable Blink Detection", True)
            self.emergency_mode = st.checkbox("Emergency Mode (Siren)", False)
            
            if st.button("Reset Message Board"):
                self.reset_board()
            
            st.header("Quick Commands")
            cols = st.columns(3)
            for i, (key, name) in enumerate(self.image_names.items()):
                with cols[i % 3]:
                    if st.button(name):
                        self.add_to_message(name)
                        self.speak(name)
                        if key == "emergency":
                            self.play_siren()
                        else:
                            self.play_sound()
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header("Live Camera Feed")
            self.webrtc_ctx = webrtc_streamer(
                key="vision-board",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_frame_callback=self.process_video,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            st.header("Message Board")
            self.message_board = st.empty()
            self.update_message_board()
        
        with col2:
            st.header("Quick Actions")
            for key, name in self.image_names.items():
                if st.button(name, key=f"btn_{key}"):
                    self.add_to_message(name)
                    self.speak(name)
                    if key == "emergency":
                        self.play_siren()
                    else:
                        self.play_sound()
    
    def process_video(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = detector(gray_frame)
        if len(faces) == 0:
            st.session_state.no_face_frames += 1
            if st.session_state.no_face_frames >= self.max_no_face_frames:
                st.warning("No face detected for too long. Please check your camera.")
        else:
            st.session_state.no_face_frames = 0
            
            # Process single face
            face = faces[0]
            landmarks = predictor(gray_frame, face)
            
            # Get eye contours
            left_eye, right_eye = self.eyes_contour_points(landmarks)
            
            # Draw eye contours
            cv2.polylines(img, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(img, [right_eye], True, (0, 0, 255), 2)
            
            if self.blink_detection:
                # Detect blinking
                left_eye_ratio = self.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                right_eye_ratio = self.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                
                # Check for blink patterns
                if blinking_ratio > 5:  # Eyes closed
                    current_time = time.time()
                    if len(st.session_state.blink_pattern) == 0 or current_time - st.session_state.pattern_start_time > 0.5:
                        st.session_state.blink_pattern.append('1')
                        st.session_state.pattern_start_time = current_time
                        st.session_state.last_blink_time = current_time
                        st.session_state.blinking_frames += 1
                else:  # Eyes open
                    current_time = time.time()
                    if len(st.session_state.blink_pattern) > 0 and current_time - st.session_state.pattern_start_time > 0.5:
                        st.session_state.blink_pattern.append('0')
                
                # Check for completed patterns
                pattern_str = ''.join(st.session_state.blink_pattern)
                for pattern, action in self.recognized_patterns.items():
                    if pattern_str.endswith(pattern):
                        self.handle_pattern_action(action)
                        st.session_state.blink_pattern = []
                        break
                
                # Check for blink timeout
                if time.time() - st.session_state.last_blink_time > 10 and not st.session_state.blink_warning_shown:
                    st.warning("No blinking detected for 10 seconds. Are you okay?")
                    st.session_state.blink_warning_shown = True
                    if self.emergency_mode:
                        self.play_siren()
                elif time.time() - st.session_state.last_blink_time <= 10:
                    st.session_state.blink_warning_shown = False
                
                # Show green eyes when closed
                if blinking_ratio > 5:
                    cv2.polylines(img, [left_eye], True, (0, 255, 0), 2)
                    cv2.polylines(img, [right_eye], True, (0, 255, 0), 2)
                    
                    # Check if blink is long enough to trigger action
                    if st.session_state.blinking_frames >= self.frames_to_blink and st.session_state.cooldown_counter == 0:
                        # Here you could add action for single blink if needed
                        st.session_state.cooldown_counter = self.cooldown_frames
                        st.session_state.blinking_frames = 0
                else:
                    st.session_state.blinking_frames = 0
                
                # Handle cooldown
                if st.session_state.cooldown_counter > 0:
                    st.session_state.cooldown_counter -= 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def handle_pattern_action(self, action):
        """Handle recognized blink patterns"""
        name = self.image_names.get(action, action)
        self.add_to_message(name)
        self.speak(name)
        
        if action == 'emergency':
            self.play_siren()
        else:
            self.play_sound()
    
    def add_to_message(self, text):
        """Add text to the message board"""
        st.session_state.text += " " + text
        self.update_message_board()
    
    def update_message_board(self):
        """Update the message board display"""
        self.message_board.text_area("Current Message", 
                                  value=st.session_state.text, 
                                  height=200,
                                  key="message_board_area")
    
    def midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)
    
    def get_blinking_ratio(self, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        ratio = hor_line_length / ver_line_length
        return ratio
    
    def eyes_contour_points(self, facial_landmarks):
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = facial_landmarks.part(n).x
            y = facial_landmarks.part(n).y
            left_eye.append([x, y])
        for n in range(42, 48):
            x = facial_landmarks.part(n).x
            y = facial_landmarks.part(n).y
            right_eye.append([x, y])
        left_eye = np.array(left_eye, np.int32)
        right_eye = np.array(right_eye, np.int32)
        return left_eye, right_eye
    
    def speak(self, text):
        """Speak text using TTS"""
        if self.voice_enabled:
            self.tts_queue.put(text)
    
    def play_sound(self):
        """Play standard sound"""
        if self.voice_enabled:
            try:
                sound.play()
            except:
                pass
    
    def play_siren(self):
        """Play emergency siren"""
        if self.voice_enabled:
            try:
                siren_sound.play()
            except:
                pass
    
    def reset_board(self):
        """Reset the message board"""
        st.session_state.text = ""
        self.update_message_board()
    
    def _tts_worker(self):
        """Background worker for TTS to avoid blocking"""
        while True:
            text = self.tts_queue.get()
            engine.say(text)
            engine.runAndWait()
            self.tts_queue.task_done()

# Main application
if __name__ == "__main__":
    app = VisionBoardApp()