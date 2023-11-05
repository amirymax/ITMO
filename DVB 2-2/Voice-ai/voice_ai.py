import openai
import pyaudio
import pyttsx3
import vosk
from apikey import API_KEY
import json

openai.api_key = API_KEY
tts = pyttsx3.init()

voices = tts.getProperty('voices')
tts.setProperty('voices', 'en')

for voice in voices:
    print(voice.name)
    if voice.name == 'Microsoft Zira Desktop - English (United States)':
        tts.setProperty('voice', voice.id)
model = vosk.Model('vosk-model-small-en-us-0.15')
record = vosk.KaldiRecognizer(model, 16000)
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16,
                 channels=1,
                 rate=16000,
                 input=True,
                 frames_per_buffer=8000)
stream.start_stream()


def listen():
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if record.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(record.Result())
            if answer['text']:
                yield answer['text']


def speak(say: str):
    tts.say(say)
    tts.runAndWait()


def chat_with_model(message):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()


def insert_line_breaks(string, max_length):
    parts = []
    current_part = ''
    words = string.split()

    for word in words:
        if len(current_part) + len(word) <= max_length:
            current_part += word + ' '
        else:
            parts.append(current_part.strip())
            current_part = word + ' '

    if current_part:
        parts.append(current_part.strip())

    return parts


# Example conversation
def main():
    # Greeting the model

    user_input = 'Hi sir, how can i help YOU?'
    print('ChatGPT: ', user_input)
    speak(user_input)
    for user_input in listen():
        print('YOU: -', user_input)
        if user_input.lower() in 'exitstop':
            print('ChatGPT: It was good to work with you sir. Bye')
            speak('It was good to work with you sir. Bye')
            break
        response = chat_with_model(user_input)
        parts = insert_line_breaks(response, 100)
        print('ChatGPT: ', end='')
        for i in parts:
            print(i)
        # print('ChatGPT:',response)
        speak(response)


if __name__ == '__main__':
    main()
