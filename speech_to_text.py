import tempfile
import os
from pydub import AudioSegment
import speech_recognition as sr

def transcribe_audio_file(audio_file):
    """
    Transcribe audio file to text using local speech recognition (Google Web Speech API).
    Accepts a file-like object.
    """
    try:
        # Convert audio to WAV format if needed
        audio = AudioSegment.from_file(audio_file)
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        os.unlink(wav_path)
        return text
    except Exception as e:
        raise Exception(f"Audio transcription failed: {str(e)}")