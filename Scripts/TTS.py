import os
import asyncio
import edge_tts
import subprocess

proc = None

async def play_sound(text):
    global proc
    cc = edge_tts.Communicate(text, voice="ja-JP-NanamiNeural")
    await cc.save("temp.mp3")
    proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "temp.mp3"])
    proc.wait()
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")

def make_sound(text):
    asyncio.run(play_sound(text))

if __name__ == "__main__":
    make_sound("Hello how are you")