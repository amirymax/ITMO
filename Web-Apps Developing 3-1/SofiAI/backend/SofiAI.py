import vosk
import sys
import sounddevice as sd
import queue
import json


class Sofi:
    def __init__(self) -> None:
        self.text: str = ''
        self.vosk_model = vosk.Model("model_small")

    def listen(self) -> None:
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

        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16',
                               channels=1, callback=q_callback):

            rec = vosk.KaldiRecognizer(self.vosk_model, samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    voice = json.loads(rec.Result())["text"]
                    if voice.startswith(config.VA_ALIAS):
                        # обращаются к ассистенту
                        cmd = recognize_cmd(filter_cmd(voice))

                        if cmd['cmd'] not in config.VA_CMD_LIST.keys():
                            tts.va_speak("Что?")
                        else:
                            execute_cmd(cmd['cmd'])

    def execute(self) -> None:
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
