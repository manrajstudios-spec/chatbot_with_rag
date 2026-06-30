import webrtcvad
import numpy as np
import sounddevice as sd
from loader import console
from openwakeword import Model
from faster_whisper import WhisperModel

alexa = Model()
whisper_model = WhisperModel("base", device="cuda")

def record_voice():
    check_voice = webrtcvad.Vad(1)
    sample_rate = 16000
    block_size = 480
    channels = 1

    frames = []
    silence_threshold = int(2 * sample_rate / block_size)
    silence_frames = 0
    is_speaking = False
    speech_started = False

    with console.status("[dim]recording...[/dim]",spinner="moon"):
        with sd.InputStream(device=0,channels=channels,samplerate=sample_rate,blocksize=block_size,dtype="int16") as stream:
            while True:
                audio_chunk,_ = stream.read(block_size)
                frame = audio_chunk.tobytes()

                is_speaking = check_voice.is_speech(frame,sample_rate)

                if is_speaking:
                    speech_started = True
                    frames.append(audio_chunk)
                    silence_frames = 0
                elif speech_started:
                    silence_frames += 1
                    frames.append(audio_chunk)
                    if silence_frames >= silence_threshold:
                        break

        audio = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
        segments, info = whisper_model.transcribe(audio)

        text = " ".join(segment.text for segment in segments).strip()
        return text

def wake_word():
    with console.status("[dim]Listening for wake word...[/dim]",spinner="moon"):
        with sd.InputStream(device=0,channels=1,samplerate=16000,dtype="int16",blocksize=1280) as stream:
            while True:
                audio_data,_ = stream.read(1280)
                prediction = alexa.predict(audio_data.flatten())
                if prediction["alexa"] > 0.5:
                    return True

def get_query():
    if wake_word():
        return record_voice()


