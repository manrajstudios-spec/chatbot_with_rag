import os
import pyaudio
import webrtcvad
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openwakeword.model import Model
from faster_whisper import WhisperModel
from loader import console
from rich.panel import Panel

whisper = WhisperModel("base", device="cuda")
alexa = Model()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def transcribe(file_name):
    if not os.path.exists(file_name): return ""
    segs, _ = whisper.transcribe(file_name)
    os.remove(file_name)
    return " ".join([s.text for s in segs]).strip()

def record_until_silence():
    check_voice = webrtcvad.Vad(2)
    console.print("[dim]Listening...[/dim]")
    max_silence = int(2 * 16000 / 480)
    frames = []
    silence_frames = 0
    speaking = False

    with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=480) as stream:
        while True:
            if not speaking:
                silence_frames = 0
            audio_chunk, _ = stream.read(480)
            frame = audio_chunk.tobytes()
            is_speaking = check_voice.is_speech(frame, 16000)
            if is_speaking:
                speaking = True
                frames.append(audio_chunk)
            else:
                silence_frames += 1
                if silence_frames > max_silence:
                    console.print("[dim]Silence detected[/dim]")
                    break

    f_np = np.concatenate(frames, dtype=np.int16)
    wav.write("in.wav", 16000, f_np)
    return "in.wav"

def wake_word():
    console.print("[dim]Listening for wake word...[/dim]")
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frame = np.frombuffer(audio_data, dtype=np.int16)
        prediction = alexa.predict(audio_frame)
        if prediction["alexa"] > 0.5:
            return True

def stop_wake_word():
    console.print("[dim]\nStopping...[/dim]")
    stream.stop_stream()
    stream.close()
    p.terminate()

def get_query():
    if wake_word():
        path = record_until_silence()
        t = transcribe(path)
        console.print(Panel(t, title="[bold cyan]You[/bold cyan]", border_style="cyan"))
        return t

if __name__ == "__main__":
    wake_word()