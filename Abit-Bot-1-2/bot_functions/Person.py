class Student:
    def __init__(self, name=''):
        self.choose_subject = True
        self.choose_area = True
        self.area = None
        self.subject = None
        self.name = name
        self.passed_questions = []
        self.correct_answers = 0
        self.test = {}
        self.correct_answer = None
        self.question_number = 1
        self.action_markup = None
        self.subjects_markup = None
        self.area_markup = None
        self.subjects_markup_status = True
        self.area_markup_status = True
        self.subjects_of_area = []
        self.subjects_of_area_index = 0
        self.testing_area = False
        self.all_points = 0
        self.joined_subjects = ''

    def answered_question(self, question_id):
        self.passed_questions.append(question_id)

    def answered_questions(self):
        return self.passed_questions

    def detect_subjects(self):
        if self.area == 'mega1':
            self.subjects_of_area = ['math', 'ict', 'physics', 'russian']
        if self.area == 'mega2':
            self.subjects_of_area = ['physics', 'math', 'ict', 'russian']
        if self.area == 'mega3':
            self.subjects_of_area = ['ict', 'math', 'english', 'russian']
        if self.area == 'mega4':
            self.subjects_of_area = ['physics', 'ict', 'math', 'english']
        if self.area == 'mega5':
            self.subjects_of_area = ['biology', 'ict', 'english', 'russian']
