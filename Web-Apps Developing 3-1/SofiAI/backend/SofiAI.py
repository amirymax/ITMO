import vosk
import sys
import sounddevice as sd
import queue
import json
import config
from fuzzywuzzy import fuzz
import torch
import time


class Sofi:
    def __init__(self) -> None:

        self.torch_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                             model='silero_tts',
                                             language='en',
                                             speaker='v3_en')
        self.torch_model.to(torch.device('cpu'))

        self.vosk_model = vosk.Model(
            "C:\\Users\\Sofia\\Desktop\\ITMO\\Web-Apps Developing 3-1\\SofiAI\\backend\\model-small")

    def listen(self) -> None:
        self.say('I am listening to you sir. What did you want?')
        print('Listening')
        q = queue.Queue()
        samplerate = 16000
        device = 1

        def q_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        def recognize_cmd(cmd: str):
            rc = {'cmd': '', 'percent': 0}
            for c, v in config.VA_CMD_LIST.items():

                for x in v:
                    vrt = fuzz.ratio(cmd, x)
                    if vrt > rc['percent']:
                        rc['cmd'] = c
                        rc['percent'] = vrt

            return rc

        def filter_cmd(raw_voice: str):
            cmd = raw_voice

            for x in config.VA_ALIAS:
                cmd = cmd.replace(x, "").strip()

            for x in config.VA_TBR:
                cmd = cmd.replace(x, "").strip()
            print(f'CMD: {cmd}')
            return cmd
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16',
                               channels=1, callback=q_callback):

            rec = vosk.KaldiRecognizer(self.vosk_model, samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    voice = json.loads(rec.Result())["text"]
                    print(f'Voice: {voice}')
                    if voice.startswith(config.VA_ALIAS):
                        # обращаются к ассистенту
                        cmd = recognize_cmd(filter_cmd(voice))
                        print(cmd)
                        if cmd['cmd'] not in config.VA_CMD_LIST.keys():
                            self.say("What?")
                        else:
                            self.execute(cmd['cmd'])

    def say(self, what: str):
        sample_rate = 48000
        audio = self.torch_model.apply_tts(text=what+"..",
                                           speaker="en_5",
                                           sample_rate=sample_rate,
                                           put_accent=True,
                                           put_yo=True)

        sd.play(audio, sample_rate * 1.05)
        time.sleep((len(audio) / sample_rate) + 0.5)
        sd.stop()

    def execute(self, cmd: str) -> None:
        pass

    def open(self, program_name) -> None:
        pass

    def volume(self, degree) -> None:
        pass

    def google(self, request) -> None:
        pass

    def restart_pc(self) -> None:
        pass

    def turn_off_pc(self) -> None:
        pass

    def talk(self) -> None:
        pass


Sofi()
