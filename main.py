import json
import os
import time

import pyaudio
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import wave

from transciption import transcribe_audio
import config

## Parameters

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

SILENCE_DURATION = 1.25 # Silence duration in seconds
NOISE_SENSITIVITY = 2000

FOLDER = "recordings"
SETTINGS_FILE = "settings.json"
SESSION_DIR = f"{FOLDER}\\sessions"
SESSION_FILE = "sessions.json"
VOICE_FILE = f"{FOLDER}\\voice.wav"

## Initializations

p = pyaudio.PyAudio()

try:
    path = os.path.join(FOLDER, SETTINGS_FILE)
    with open(path, 'r') as f:
        settings = json.load(f)
    DEVICE_INDEX = settings["deviceIndex"]

except FileNotFoundError:
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    DEVICE_INDEX = int(input("Select Devic Index: "))
    while True:
        doSave = input("Save Selection? (y/n): ")
        if doSave.lower() == "y":
            path = os.path.join(FOLDER, SETTINGS_FILE)
            os.makedirs(FOLDER, exist_ok=True)
            with open(path, "w", encoding="utf-8") as doSaveFile:
                json.dump(
                    {"deviceIndex": DEVICE_INDEX},
                    doSaveFile,
                    indent=4,
                )
            break
        elif doSave.lower() == "n":
            break
        else:
            print("Invalid Selection")

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=DEVICE_INDEX
)
model = load_silero_vad(onnx=True)

def analyse_ambient_noise(duration=3):
    print("Analysing ambient noise...")
    noise_levels = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        audio_data = stream.read(CHUNK)
        noise_levels.append(np.max(np.frombuffer(audio_data, dtype=np.int16)))

    print("Finished analysing ambient noise: ", format(np.mean(noise_levels)))

    ambient_noise_level = np.mean(noise_levels) + NOISE_SENSITIVITY
    print("Ambient noise level: ", ambient_noise_level)
    return ambient_noise_level

def is_silent(data, threshold):
    return np.mean(np.max(np.frombuffer(data, dtype=np.int16)) < threshold)

def detect_voice(audio_chunk):
    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)

    # Opening a new WAV file for writing
    with wave.open(VOICE_FILE, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(p.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(audio_int16.tobytes())

    wav = read_audio(VOICE_FILE)
    speech_timestamps = get_speech_timestamps(wav, model)

    os.remove(VOICE_FILE)

    print(speech_timestamps)
    return speech_timestamps

def save_to_wav(frames, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def append_audio(base_file, new_frames):
    if os.path.exists(base_file):
        with wave.open(base_file, 'rb') as wf:
            existing_frames = wf.readframes(wf.getnframes())
        frames = existing_frames + b''.join(new_frames)
    else:
        frames = b''.join(new_frames)
    save_to_wav([frames], base_file)

class Session:
    def __init__(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(SESSION_DIR, exist_ok=True) # To ensure the session directory exists

        self.file = os.path.join(SESSION_DIR, SESSION_FILE)
        self.path = os.path.join(SESSION_DIR, f"session_{timestamp}")

        if not os.path.exists(self.file):
            with open(self.file, "w") as f:
                json.dump([], f)

    def load_sessions(self):
        with open(self.file, "r") as f:
            return json.load(f)

    def save_session(self, session_data):
        sessions = self.load_sessions()
        sessions.append(session_data)
        with open(self.file, "w") as f:
            json.dump(sessions, f, indent=4)

    def create_session(self):
        os.makedirs(self.path, exist_ok=True)
        return self.path

def main():
    prev_frames = []
    all_frames = []

    session = Session()

    print("Starting session...")

    session_path = session.create_session()
    complete_voice_file = os.path.join(session_path, "complete_voice.wav")
    chunk_metadata_file = os.path.join(session_path, "chunks.json")
    complete_file = os.path.join(session_path, "complete.wav")

    if not os.path.exists(chunk_metadata_file):
        with open(chunk_metadata_file, "w") as f:
            json.dump([], f)

    ambient_noise_level = analyse_ambient_noise()

    while True:
        if len(prev_frames) > 2:
            prev_frames = prev_frames[1:]

        audio_data: bytes = stream.read(CHUNK)
        prev_frames.append(audio_data)

        all_frames.append(audio_data)
        save_to_wav(all_frames, complete_file)

        if not is_silent(audio_data, ambient_noise_level):
            print("Audio detected, recording...")
            frames = [prev_frames[0], audio_data]

            silence_start = None
            print("Listening for audio...")
            while True:
                audio_data = stream.read(CHUNK)
                frames.append(audio_data)
                all_frames.append(audio_data)

                if is_silent(audio_data, ambient_noise_level):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        print("Silence detected, stopping recording...")
                        break
                else:
                    silence_start = None

            append_audio(complete_file, frames)

            chunk_file = os.path.join(session_path, f"chunk_{int(time.time())}.wav")
            save_to_wav(frames, chunk_file)

            with open(chunk_file, 'rb') as f:
                audio_bytes = f.read()

            if detect_voice(audio_bytes):
                transciption = transcribe_audio(chunk_file, audio_bytes)
                text = transciption.text
                print(transciption.text)

                with open(chunk_metadata_file, "r+") as f:
                    chunk_metadata = json.load(f)
                    chunk_metadata.append({
                        "chunk_file": chunk_file,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transcription": text,
                    })
                    f.seek(0)
                    json.dump(chunk_metadata, f, indent=4)

if __name__ == "__main__":
    print(config.NAME + " " + config.VERSION)
    input("Press ENTER to start session: ")
    print("\n")
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        stream.stop_stream()
        stream.close()
        p.terminate()

