# Multimodal Emotion Detector 🎭

This project is a real-time, multimodal Artificial Intelligence script that detects human emotions using both facial expressions and speech. It captures a 3-second audio clip and a webcam frame, analyzing both simultaneously to provide confidence scores for your current emotional state.

## 🛠️ Tools and Technologies Used
* **Python 3.11**
* **DeepFace:** For facial emotion recognition (Computer Vision).
* **OpenCV:** For webcam frame capture and image processing.
* **Hugging Face Transformers (Hubert):** Fine-tuned model (`superb/hubert-large-superb-er`) for Speech Emotion Recognition (SER).
* **PyTorch & SciPy:** For underlying tensor operations and audio array handling.

## ⚙️ System Prerequisites
Before running the Python environment, you must have **FFmpeg** installed on your system to process the audio files.
* **macOS (via Homebrew):** `brew install ffmpeg`
* **Linux (Ubuntu/Debian):** `sudo apt install ffmpeg`

## 🚀 Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv emotion_env
   source emotion_env/bin/activate  # On Windows use: emotion_env\Scripts\activate
3. **Install the required Python libraries:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

## 🎮 How to Use
Press ENTER when prompted. Speak into your microphone and look at your webcam for 3 seconds. The script will save your capture locally (last_capture.jpg and last_capture.wav), play the audio back to you, and print the AI's emotional analysis to the console.
