import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('pulsar_stars_new.csv')

data = data[(data.MIP >= 10) & (data.MIP <= 100)]
# print('Число строк:', len(data))
# print('Выборочнее среднее столбца MIP:', data.MIP.mean())
# print('Минимальное значение столбца MIP:', data.MIP.min())

data = data.sort_values(by = 'SIP')
X = data.drop('TG', axis=1)
y = data['TG']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=33, stratify=y)
# print('Максимальное значение STDC:', X_train['STDC'].max())
scaler = MinMaxScaler()

X_train_normilized = scaler.fit_transform(X_train)
X_test_normilized = scaler.transform(X_test)

# print('Выборочнее среднее для столбца STDIP:', X_train_normilized[:, X_train.columns.get_loc('STDIP')].mean())

model=LogisticRegression()
model.fit(X_train_normilized, y_train)

y_pred = model.predict(X_test_normilized)
accuracy = accuracy_score(y_test, y_pred)

confusion = confusion_matrix(y_test, y_pred)

# print('Значение True Positive:',confusion[1][1])

f1 = f1_score(y_test, y_pred)
# print('F1 Score:', f1)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_normilized, y_train)

k_pred = knn_model.predict(X_test_normilized)

# k_accuracy = accuracy_score(k_pred, y_test)

k_confusion = confusion_matrix(y_test, k_pred)
# print('KNN True Positive:',k_confusion[1,1])
print('KNN F1 score:',f1_score(y_test, k_pred))
