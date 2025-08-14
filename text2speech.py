import whisper
import os

def transcribe_audio(file_path):
    # Check if file exists before transcribing
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Load Whisper model (base is faster, large is more accurate)
    model = whisper.load_model("base")
    print(f"✅ Transcribing: {file_path}")

    result = model.transcribe(file_path)
    print("\n--- TRANSCRIPTION ---\n")
    print(result["text"])

if __name__ == "__main__":
    # Your exact path
    audio_file = r"C:\aud to text\eshaudio.MP3"
    transcribe_audio(audio_file)
