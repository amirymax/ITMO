#!/usr/bin/env python
# coding: utf-8

# ### Лабораторная 5.  Метод k-ближайших соседей

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv("diabetes.csv")
df


# In[7]:


from sklearn.preprocessing import StandardScaler

# Разделим признаки и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создадим новый DataFrame с отмасштабированными признаками
df_scaled = pd.DataFrame(data=X_scaled, columns=X.columns)
df_scaled['Outcome'] = y

# Выведем первые несколько строк для проверки
df_scaled.head()


# In[9]:


from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Реализация функции для вычисления матрицы ошибок
def confusion_matrix(true_labels, predicted_labels):
    unique_labels = np.unique(np.concatenate((true_labels, predicted_labels)))
    matrix = np.zeros((len(unique_labels), len(unique_labels)))

    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[label_to_index[true_label]][label_to_index[predicted_label]] += 1
    
    return matrix

# Реализация функции для вычисления евклидового расстояния между двумя точками
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Реализация функции k-ближайших соседей
def k_nearest_neighbors(train_data, test_point, k):
    distances = []
    for _, train_row in train_data.iterrows():
        train_point = train_row[:-1]
        label = train_row[-1]
        distance = euclidean_distance(test_point, train_point)
        distances.append((label, distance))
    
    distances.sort(key=lambda x: x[1])
    nearest_neighbors = distances[:k]
    class_votes = {}
    for neighbor in nearest_neighbors:
        label = neighbor[0]
        class_votes[label] = class_votes.get(label, 0) + 1
    
    predicted_class = max(class_votes, key=class_votes.get)
    return predicted_class

# Чтение данных и предварительная обработка
df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(data=X_scaled, columns=X.columns)
df_scaled['Outcome'] = y

# Разделение данных на обучающий и тестовый наборы
train_data, test_data = train_test_split(df_scaled, test_size=0.2, random_state=42)

# Зададим значения k, которые хотим проверить
k_values = [3, 5, 10]

# Модель 1: Признаки случайно отбираются
print("Модель 1: Признаки случайно отбираются")
for k in k_values:
    predicted_labels = []
    for _, row in test_data.iterrows():
        test_point = row[:-1]
        predicted_class = k_nearest_neighbors(train_data, test_point, k)
        predicted_labels.append(predicted_class)
    
    true_labels = test_data["Outcome"].values
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    print(f"При k = {k}, точность: {accuracy}")
    print(f"Матрица ошибок (Confusion Matrix) при k = {k}:\n{conf_matrix}")

# Модель 2: Фиксированный набор признаков
selected_features = ["Pregnancies", "Glucose", "BMI"]
train_data_subset = train_data[selected_features + ["Outcome"]]
test_data_subset = test_data[selected_features + ["Outcome"]]
print("Модель 2: Фиксированный набор признаков")
for k in k_values:
    predicted_labels = []
    for _, row in test_data_subset.iterrows():
        test_point = row[:-1]
        predicted_class = k_nearest_neighbors(train_data_subset, test_point, k)
        predicted_labels.append(predicted_class)
    
    true_labels = test_data_subset["Outcome"].values
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    print(f"При k = {k}, точность: {accuracy}")
    print(f"Матрица ошибок (Confusion Matrix) при k = {k}:\n{conf_matrix}")


# Исходя из проведенной оценки производительности двух моделей k-ближайших соседей (k-NN) при 
# 
# разных значениях k и с различными наборами признаков, можно сделать следующие выводы:
# 
# Модель 1 (Признаки случайно отбираются) и Модель 2 (Фиксированный набор признаков) показали схожую производительность с точки зрения точности классификации:
# 
# При различных значениях k (k=3, k=5, k=10) обе модели достигли точности, колеблющейся в пределах примерно 71.90% - 75.82%. Это означает, что примерно 72% - 76% объектов классифицированы правильно.
# 
# Матрица ошибок (Confusion Matrix) позволяет лучше понять, где модели совершают ошибки:
# 
# В обеих моделях при более низком значении k (например, k=3), количество ложно-положительных и ложно-отрицательных случаев сравнительно высоко. Это может быть вызвано чувствительностью моделей к выбросам или шумам в данных.
# 
# При увеличении значения k до 10, количество ложных положительных и ложных отрицательных случаев снижается, что указывает на более стабильную классификацию.
# 
# 

#                                 Матрица ощибок при k = 3
#                                 
# 83 объекта были верно классифицированы как положительные.
# 
# 27 объектов были верно классифицированы как отрицательные.
# 
# 21 объект был неверно классифицирован как положительный (ложное положительное).
# 
# 22 объекта были неверно классифицированы как отрицательные (ложное отрицательное).

# In[ ]:




