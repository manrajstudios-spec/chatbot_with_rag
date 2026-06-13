import os
import pyaudio
import webrtcvad
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openwakeword.model import Model
from sympy.physics.vector import frame
from faster_whisper import WhisperModel, transcribe

whisper = WhisperModel("base", device="cuda")

rhaspy = Model(["hey rhasspy"])

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def transcribe(file_name):
    segs,_ = whisper.transcribe(file_name)
    os.remove(file_name)
    return " ".join([s.text for s in segs]).strip()

def record_until_silence():
    check_voice = webrtcvad.Vad(2)
    print("Listening...")

    max_silence = int(2*16000/480)
    frames = []
    silence_frames = 0
    speaking = False
    with sd.InputStream(samplerate=16000,channels=1,dtype='int16',blocksize=480) as stream:
        while True:
            if not speaking:
                silence_frames =0
            audio_chunk,_ = stream.read(480)
            frame = audio_chunk.tobytes()
            is_skeaking = check_voice.is_speech(frame,16000)

            if is_skeaking:
                speaking = True
                frames.append(audio_chunk)
            else:
                silence_frames += 1

                if silence_frames > max_silence:
                    print("Silencing...")
                    break

    f_np = np.concatenate(frames,dtype=np.int16)
    wav.write("in.wav",16000,f_np)
    return "in.wav"

def wake_word():
    print("Listning For Wake_word")
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frame = np.frombuffer(audio_data, dtype=np.int16)
        prediction = rhaspy.predict(audio_frame)

        if rhaspy.prediction_buffer['hey rhasspy'][-1] > 0.5:
            print("Waking UP")
            return True

def stop_wake_word():
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()

def get_query():
    if wake_word():
        path = record_until_silence()
        return transcribe(path)

if __name__ == "__main__":
    a = record_until_silence()
    print(transcribe(a))