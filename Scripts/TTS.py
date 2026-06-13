import os
import asyncio
import edge_tts
import threading
from playsound3 import playsound

sound = None

def interrupt():
    input()
    if sound:
        sound.stop()
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")


async def play_sound(text):
    global sound

    cc = edge_tts.Communicate(text, voice="ja-JP-NanamiNeural")
    await cc.save("temp.mp3")

    tt = threading.Thread(target=interrupt, daemon=True)
    tt.start()

    sound = playsound("temp.mp3", block=False)
    sound.wait()

    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")


def make_sound(text):
    asyncio.run(play_sound(text))

if __name__ == "__main__":
    make_sound("Hello how are you")