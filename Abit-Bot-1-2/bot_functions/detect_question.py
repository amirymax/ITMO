import sqlite3
from bot_functions import levels, refactor


def get_question(subject_name, question_number, answered_questions):
    connect = sqlite3.connect('bot_functions/questions.db')
    if question_number < 3:
        question_id = levels.one()
        while question_id in answered_questions:
            question_id = levels.one()
    elif question_number < 8:
        question_id = levels.two()
        while question_id in answered_questions:
            question_id = levels.two()
    else:
        question_id = levels.three()
        while question_id in answered_questions:
            question_id = levels.three()
    print(subject_name)
    random_question = connect.execute(
        f'SELECT question, v1,v2,v3,v4 from {subject_name} WHERE id={question_id}').fetchall()
    connect.close()
    question, shuffled_choices = refactor.get_question(random_question)
    correct_answer = refactor.get_answer(random_question)

    return question, shuffled_choices, correct_answer, question_id
