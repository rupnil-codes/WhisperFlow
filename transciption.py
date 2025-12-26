import groq
from config import GROQ_API

def transcribe_audio(chunk_file, audio_file):
    client = groq.Client(
        api_key=GROQ_API,
    )

    print("chunk:" + chunk_file)

    # Passing the audio file (as a file-like object) to the transcription API
    result = client.audio.transcriptions.create(
        file=(chunk_file, audio_file),
        model="whisper-large-v3",
        language="en",
        response_format="verbose_json",
        timestamp_granularities = [
            "word",
            "segment",
        ],
        temperature=0
    )

    print(result)
    return result