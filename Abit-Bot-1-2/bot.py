from aiogram import types, executor, Dispatcher, Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from bot_functions.Person import Student
from bot_functions.ask_question import new_question
from TOKEN import TOKEN

bot = Bot(token=TOKEN)  # @itmo_abit_help_bot
dp = Dispatcher(bot)
user = Student()


def get_from_call(call):
    return call.data.split()[1]


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    user.area_markup_status = True
    user.subjects_markup_status = True

    markup = InlineKeyboardMarkup()
    but1 = InlineKeyboardButton('Выбор мегафакультета', callback_data='area')
    but2 = InlineKeyboardButton('Тест по предметам', callback_data='subjects')
    markup.insert(but1)
    markup.insert(but2)
    user.action_markup = markup

    user.name = message.chat.first_name
    await bot.send_message(message.chat.id,
                           f'Привет, {user.name}\nЕсли выбор профессии или направления кажется сложным, то не '
                           f'стоит беспокоиться! Данный бот поможет вам проверить свои знания в школьных '
                           f'предметах, а так же определить подходящее вам направление. С чего бы вы хотели '
                           f'начать?',
                           reply_markup=markup)


@dp.callback_query_handler(lambda c: c.data == 'area')
async def area(call: types.callback_query):
    # user.subjects_markup_status=False
    user.choose_area = True
    megas = {
        'КТУ': 'mega1',
        'ФТМФ': 'mega2',
        'ТИНТ': 'mega3',
        'ФТМИ': 'mega4',
        'БиоТех': 'mega5'
    }
    markup = InlineKeyboardMarkup(resize_keyboard=True)

    for i, j in megas.items():
        markup.insert(InlineKeyboardButton(i, callback_data='area ' + j))
    user.area_markup = markup
    await bot.answer_callback_query(call.id)
    await bot.send_message(call.message.chat.id, 'Выберите мегафакультет:', reply_markup=markup)


subjects = {
    'Английский': 'english',
    'Биология': 'biology',
    'География': 'geography',
    'ИКТ': 'ict',
    'История': 'history',
    'Литература': 'literature',
    'Математика': 'math',
    'Обществознание': 'social_science',
    'Русский': 'russian',
    'Физика': 'physics',
    'Химия': 'chemistry'
}


@dp.callback_query_handler(lambda c: c.data == 'subjects')
async def subject(call: types.callback_query):
    # user.subjects_markup_status = True
    user.choose_subject = True
    # if user.subjects_markup_status:

    markup = InlineKeyboardMarkup(resize_keyboard=True)

    for i, j in subjects.items():
        markup.insert(InlineKeyboardButton(i, callback_data='subject ' + j))
    user.subject_markup = markup
    await bot.answer_callback_query(call.id)
    await bot.send_message(call.message.chat.id, 'Выберите предмет:', reply_markup=markup)


@dp.callback_query_handler(text_contains='area')
async def area_choose_subject(call: types.callback_query):
    user.testing_area = True
    if user.choose_area:
        subjects_names = []
        user.choose_area = False
        user.area = get_from_call(call)
        user.detect_subjects()
        print(user.subjects_of_area)
        for i in user.subjects_of_area:
            subjects_names.append('**' + list(subjects.keys())[list(subjects.values()).index(i)] + '**')
        subjects_joined = ', '.join(i for i in subjects_names)
        user.joined_subjects = subjects_joined
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton('Продолжить', callback_data='continue'))
        user.subject = user.subjects_of_area[user.subjects_of_area_index]
        # question, variants = new_question(user)
        await bot.send_message(call.message.chat.id, 'Тестирование проходит по данным предметам:\n' + subjects_joined,
                               reply_markup=markup,
                               parse_mode='markdown')
        # user.question_number += 1


@dp.callback_query_handler(text_contains='subject')
async def subject_testing(call: types.callback_query):
    if user.choose_subject:
        user.choose_subject = False
        user.subject = get_from_call(call)

        question, variants = new_question(user)
        await bot.send_message(call.message.chat.id, f'{user.question_number}. ' + question, reply_markup=variants)
        user.question_number += 1


@dp.callback_query_handler(text_contains='continue')
async def continuee(call: types.callback_query):
    print(user.subjects_of_area)
    if len(user.subjects_of_area):
        await bot.send_message(call.message.chat.id,
                               'Тест по предмету: ' + '**' + list(subjects.keys())[list(subjects.values()).index(
                                   user.subjects_of_area[0])] + '**',
                               parse_mode='markdown')

        user.choose_subject = False
        print(user.subject)

        question, variants = new_question(user)
        await bot.send_message(call.message.chat.id, question, reply_markup=variants)
        user.question_number += 1

        user.subjects_of_area.remove(user.subjects_of_area[0])


@dp.callback_query_handler(text_contains='variant')
async def testing(call: types.callback_query):
    answer = get_from_call(call)

    if user.test[answer] == user.correct_answer:
        user.correct_answers += 1

    edited_message = call.message.text + '\n'
    variant_letter_index = 0
    variant_letters = 'abcd'
    for i in user.test.values():
        edited_message += variant_letters[variant_letter_index] + '. ' + i
        if i == user.correct_answer:
            edited_message += '✅'
        elif i == user.test[answer]:
            edited_message += '❌'
        edited_message += '\n'
        variant_letter_index += 1
    edited_message = edited_message.strip()
    await call.message.edit_text(edited_message)
    try:
        question, variants = new_question(user)
        # if user.question_number==10:print(question)
        await bot.send_message(call.message.chat.id, f'{user.question_number}. ' + question, reply_markup=variants)
        user.question_number += 1
    except TypeError:
        if user.testing_area:
            user.all_points += user.correct_answers
            if len(user.subjects_of_area):
                markup = InlineKeyboardMarkup()
                markup.add(InlineKeyboardButton('Продолжить', callback_data='continue'))
                user.choose_subject = True
                this_subject = user.subject
                user.subject = user.subjects_of_area[0]
                await bot.send_message(call.message.chat.id,
                                       f'Тест по предмету **{this_subject}** окончен.'
                                       f'Вы набрали **{user.correct_answers} баллов**.'
                                       f' В сумме у вас **{user.all_points} баллов**. *Продолжить?*',
                                       reply_markup=markup, parse_mode='markdown')
                user.question_number = 1
                user.correct_answers = 0
                user.passed_questions = []
            else:
                await bot.send_message(call.message.chat.id,
                                       f'Вы проходили тестирование по предметам **{user.joined_subjects}** '
                                       f'и набрали **{user.all_points} баллов**, что составляет '
                                       f'**{user.all_points * 100 / 40}%**. *Выберите следующее действие:*',
                                       parse_mode='markdown', reply_markup=user.action_markup)

                user.__init__(user.name)
        else:
            await bot.send_message(call.message.chat.id,
                                   f'Тест окончен.\nВаш результать {user.correct_answers * 10}%. '
                                   f'({user.correct_answers} правильных ответов) '
                                   f'\nХотите пройти тест еще раз или выбрать направление?',
                                   reply_markup=user.action_markup)

            user.__init__(user.name)


@dp.message_handler(content_types=['text'])
async def text(message: types.Message):
    await bot.send_message(message.chat.id, "Yooo Dude! нажми на кнопки, это тебе не чатбот😡😡😡")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
