#!/usr/bin/env python
# coding: utf-8

# ## Амири Зикрулло
# ### Лабораторная 7.  Логистическая регрессия
# #### Датасет о диабете : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database 

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv("diabetes.csv")
df


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


X = df.drop('Outcome', axis=1)  # X содержит признаки, за исключением целевой переменной
y = df['Outcome']  # y содержит целевую переменную
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[11]:


import numpy as np

# Функция для вычисления гипотезы (sigmoid function)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция для вычисления функции потерь (log loss)
def log_loss(y, y_pred):
    epsilon = 1e-15
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1 - epsilon, y_pred)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Метод обучения с градиентным спуском
def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)  # Инициализация весов нулевыми значениями
    
    for iteration in range(num_iterations):
        z = np.dot(X, theta)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / m
        theta -= learning_rate * gradient
    
        if iteration % 1000 == 0:
            cost = log_loss(y, y_pred)
            print(f"Iteration {iteration}: Cost = {cost}")
    
    return theta

# Гиперпараметры
learning_rate = 0.01
num_iterations = 10000

# Обучение логистической регрессии на тренировочных данных
theta = logistic_regression(X_train, y_train, learning_rate, num_iterations)

# Функция для прогнозирования
def predict(X, theta):
    z = np.dot(X, theta)
    y_pred = sigmoid(z)
    return y_pred

# Получение прогнозов для тестовых данных
y_pred = predict(X_test, theta)

# Вычисление точности модели (опционально)
accuracy = np.mean((y_pred >= 0.5) == y_test)
print(f"Accuracy: {accuracy}")


# In[12]:


import numpy as np

# Функция для вычисления гипотезы (sigmoid function)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция для вычисления функции потерь (log loss)
def log_loss(y, y_pred):
    epsilon = 1e-15
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1 - epsilon, y_pred)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Метод обучения с градиентным спуском
def logistic_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)  # Инициализация весов нулевыми значениями
    
    for iteration in range(num_iterations):
        z = np.dot(X, theta)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / m
        theta -= learning_rate * gradient
    
        if iteration % 1000 == 0:
            cost = log_loss(y, y_pred)
            print(f"Iteration {iteration}: Cost = {cost}")
    
    return theta

# Функция для прогнозирования
def predict(X, theta):
    z = np.dot(X, theta)
    y_pred = sigmoid(z)
    return y_pred

# Функция для вычисления точности
def accuracy(y_true, y_pred):
    return np.mean((y_pred >= 0.5) == y_true)

learning_rates = [0.01, 0.001]
num_iterations = [100, 1000, 10000]

best_accuracy = 0
best_lr = None
best_iter = None

# Проведём исследование производительности гиперпараметров
for lr in learning_rates:
    for iter in num_iterations:
        # Используем градиентный спуск
        theta = logistic_regression(X_train, y_train, lr, iter)

        # Сделаем прогноз на тестовых данных
        y_pred = predict(X_test, theta)

        # Оцениваем точность модели
        acc = accuracy(y_test, y_pred)

        # Выведём результат и сохраним лучшие гиперпараметры
        print(f"Learning Rate: {lr}, Iterations: {iter}, Accuracy: {acc}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_lr = lr
            best_iter = iter

print(f"Best Learning Rate: {best_lr}, Best Iterations: {best_iter}, Best Accuracy: {best_accuracy}")


# In[13]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Сделайте прогноз на тестовых данных
y_pred = predict(X_test, theta)

# Бинаризация прогнозов с порогом 0.5
y_pred_binary = (y_pred >= 0.5).astype(int)

# Оцените точность модели
acc = accuracy_score(y_test, y_pred_binary)
prec = precision_score(y_test, y_pred_binary)
rec = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Выведите результаты
print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1-Score: {f1}")


#  ### Лучшая комбинация гиперпараметров для данной модели и набора данных 
# * Accuracy:  66.23%
# * Precision:  53.13%
# * Recall:  21.25%
# * F1-Score:  30.36%
# 
# Из этого следует, что модель показывает неплохую точность **(Accuracy)** и точность предсказаний **(Precision)**, что означает, что она правильно классифицирует многие положительные случаи, которые она предсказывает. Однако у модели низкий **Recall** и **F1-Score**, что свидетельствует о том, что она упускает множество действительных положительных случаев и не слишком хорошо сбалансирована в отношении точности и полноты.
# 
# Для улучшения **Recall** и **F1-Score**, можно попробовать изменить гиперпараметры, такие как скорость обучения **(learning rate)** и количество итераций обучения, или рассмотреть другие методы оптимизации. При этом следует учитывать баланс между точностью и способностью обнаруживать положительные случаи, чтобы определить оптимальные гиперпараметры для конкретной задачи классификации
#  

# In[ ]:




