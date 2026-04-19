import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from deepface import DeepFace
from transformers import pipeline
import warnings

# Suppress some specific HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
AUDIO_DURATION = 3  # Seconds to record
SAMPLE_RATE = 16000 # 16kHz is standard for HuggingFace audio models
AUDIO_FILE = "last_capture.wav"
IMAGE_FILE = "last_capture.jpg"

print("Loading models... ")

# Load Audio Emotion Model (Hugging Face)
audio_classifier = pipeline(
    "audio-classification", 
    model="superb/hubert-large-superb-er"
)

print("Models loaded successfully!\n")

def capture_multimodal_data():
    """Captures audio from the mic and a frame from the webcam."""
    print(f"🔴 Recording audio for {AUDIO_DURATION} seconds... Speak now!")
    
    # Start capturing audio
    audio_data = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Wait for the audio recording to finish
    sd.wait()
    print("✅ Audio recording complete.")
    
    # Grab a frame from the webcam
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to grab webcam frame. Check your camera permissions.")
        return None, None

    # --- NEW: Save the image so you can look at it ---
    cv2.imwrite(IMAGE_FILE, frame)
    print(f"📸 Image saved as: {IMAGE_FILE}")

    # --- NEW: Play the audio back to you ---
    print("🔊 Playing back what the microphone heard...")
    sd.play(audio_data, SAMPLE_RATE)
    sd.wait() # Wait until the playback finishes

    # --- NEW: Save the audio permanently ---
    write(AUDIO_FILE, SAMPLE_RATE, audio_data)
    print(f"💾 Audio saved as: {AUDIO_FILE}")
    
    return frame, AUDIO_FILE

def detect_emotions():
    frame, audio_file = capture_multimodal_data()
    
    if frame is None:
        return

    print("\n--- Analyzing Data ---")
    
    # 1. Facial Emotion Analysis (DeepFace)
    try:
        # enforce_detection=False prevents crashes if no face is perfectly aligned
        face_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_face_emotion = face_result[0]['dominant_emotion']
        face_confidence = face_result[0]['emotion'][dominant_face_emotion]
        print(f"📷 Face Emotion:   {dominant_face_emotion.capitalize()} (Confidence: {face_confidence:.2f}%)")
    except Exception as e:
        print("📷 Face Emotion:   No face detected clearly.")

    # 2. Speech Emotion Analysis (Hugging Face)
    try:
        audio_result = audio_classifier(audio_file)
        # The model returns a list of dictionaries; the first is the highest confidence
        top_audio_emotion = audio_result[0]['label']
        audio_confidence = audio_result[0]['score'] * 100
        print(f"🎤 Speech Emotion: {top_audio_emotion.capitalize()} (Confidence: {audio_confidence:.2f}%)")
    except Exception as e:
        print(f"🎤 Speech Emotion: Error analyzing audio - {e}")

if __name__ == "__main__":
    try:
        while True:
            input("\nPress ENTER to start emotion capture (or CTRL+C to quit)...")
            detect_emotions()
            print("-" * 30)
    except KeyboardInterrupt:
        print("\nExiting Emotion Detector. Goodbye!")