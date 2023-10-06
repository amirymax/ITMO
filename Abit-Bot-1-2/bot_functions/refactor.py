from random import shuffle


def get_question(a):
    s = list(a[0])
    question = s[0]
    s.remove(s[0])
    shuffle(s)
    shuffle(s)
    text = question
    return text, s


def get_answer(a):
    return a[0][1]


def configure(text):
    if len(text) <= 19:
        return text
    chars_count = 0
    text_to_return = ''
    for i in text.split():
        chars_count += len(i)
        if chars_count >= 19:
            text_to_return += '\n'
            chars_count = 0
        text_to_return += i + ' '
        chars_count += len(i)
    return text_to_return
