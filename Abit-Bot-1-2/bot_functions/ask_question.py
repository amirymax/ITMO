from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from bot_functions import detect_question
from bot_functions.refactor import configure


def new_question(user):
    if user.question_number > 10:
        return

    question, shuffled_choices, correct_answer, question_id = detect_question.get_question(user.subject,
                                                                                           user.question_number,
                                                                                           user.passed_questions)

    user.correct_answer = correct_answer
    user.answered_question(question_id)
    user.test = {
        'a': shuffled_choices[0],
        'b': shuffled_choices[1],
        'c': shuffled_choices[2],
        'd': shuffled_choices[3]
    }

    variants = InlineKeyboardMarkup(resize_keyboard=False)
    a = InlineKeyboardButton(configure(shuffled_choices[0]), callback_data='variant a')
    b = InlineKeyboardButton(configure(shuffled_choices[1]), callback_data='variant b')
    c = InlineKeyboardButton(configure(shuffled_choices[2]), callback_data='variant c')
    d = InlineKeyboardButton(configure(shuffled_choices[3]), callback_data='variant d')
    # print(configure(shuffled_choices[0]))
    # print(configure(shuffled_choices[1]))
    # print(configure(shuffled_choices[2]))
    # print(configure(shuffled_choices[3]))

    variants.add(a)
    variants.add(b)
    variants.add(c)
    variants.add(d)

    return question, variants
