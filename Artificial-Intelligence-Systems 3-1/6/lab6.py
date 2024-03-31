
# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt, ceil, log2
from collections import Counter


# In[2]:


df = pd.read_csv('./students-performance-evaluation.csv')
df.head()


# ### Предобработка данных

# #### Общие проверки

# In[3]:


df.info()


# In[4]:


df.describe()


# Видим, что пустых значений нету.

# In[5]:


null_counts = pd.DataFrame([df.isnull().sum(), df.isna().sum()], index=['isnull', 'isna'])
null_counts = null_counts.loc[:, (null_counts != 0).any(axis=0)]
print(null_counts)


# #### Визуализация 

# In[6]:


X = df.drop(['GRADE', 'STUDENT ID'], axis=1)
y = df['GRADE']


# In[7]:


fig, axs = plt.subplots(ceil(sqrt(len(X.columns))), ceil(sqrt(len(X.columns))), figsize=(15, 15), sharex=False, sharey=False, gridspec_kw={'hspace': 0.5})
for ax, col in zip(axs.flatten(), X.columns):
    X[col].value_counts().plot(kind="bar", ax=ax).set_title(col)
plt.show()


# In[8]:


y.value_counts().plot(kind="bar")


# In[9]:


y = pd.Series([1 if i >=4 else 0 for i in y])
y.value_counts().plot(kind="bar")


# ### Decision Tree

# In[10]:


cols = X.columns
cols = np.random.choice(cols, ceil(sqrt(len(cols))), replace=False)
X = X[cols]
X.head()


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)


# In[12]:


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


# In[13]:


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=60, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
            leaf_value, leaf_proba = self._most_common_label(y)
            return Node(value=leaf_value, proba=leaf_proba)

        # find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if round(best_gain, 5) == 0:
            leaf_value, leaf_proba = self._most_common_label(y)
            return Node(value=leaf_value, proba=leaf_proba)
        # create child notes and call _grow_tree() recursively
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:

                gain = self._information_gain(y, X_column, threshold)

                if gain >= best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh, best_gain

    def _information_gain(self, y, X_column, split_thresh):
        # parent node entropy
        parent_entropy = self._entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted average child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # return information gain
        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        proba = [counter[0] / len(y), counter[1] / len(y)]
        value = counter.most_common(1)[0][0]
        return value, proba

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root)[0] for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_tree(x, self.root)[1] for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value, node.proba

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# In[14]:


dt = DecisionTree()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy(y_test, y_pred)


# ##### Проверка 

# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier(max_features=ceil(sqrt(X_train.shape[1])))
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier( random_state=0)
clf.fit(X_train, y_train)
print(accuracy(y_test,clf.predict(X_test)))


# ### Провести оценку реализованного алгоритма с использованием Accuracy, precision и recall

# In[16]:


print(y_pred)
print(y_test)


# 1 - Positive, 0 - Negative [[TN, FP], [FN, TP]]

# In[17]:


def confusion_matrix(pred_y, true_y):
    res = np.zeros((2, 2))
    for pred, true in zip(pred_y, true_y):
        pred = 1 if pred == 1 else 0
        true = 1 if true == 1 else 0
        res[pred][true] += 1
    return res


# In[18]:


print(confusion_matrix(y_pred, y_test))


# In[19]:


def confusion_matrix_prob(pred_probs, true_y, threshold):
    res = np.zeros((2, 2))

    for pred_prob, true in zip(pred_probs, true_y):
        pred = 1 if pred_prob >= threshold else 0
        true = 1 if true == 1 else 0
        res[pred][true] += 1

    return res

def accuracy(conf):
    return (conf[1][1] + conf[0][0]) / (conf[1][1] + conf[0][0] + conf[1][0] + conf[0][1])
def precision(conf):
    return conf[1][1] / (conf[1][1] + conf[1][0])
def recall(conf):
    return conf[1][1] / (conf[1][1] + conf[0][1])
def tpr(conf):
    return recall(conf)
def fpr(conf):
    return conf[1][0] / (conf[1][0] + conf[0][0])
print("Accuracy: ", accuracy(confusion_matrix(y_pred, y_test)))
print("Precision: ", precision(confusion_matrix(y_pred, y_test)))
print("Recall: ", recall(confusion_matrix(y_pred, y_test)))
    


# ##### Проверка

# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# ### Построим AUC-ROC и AUC-PR

# #### AUC-ROC

# In[21]:


y_prob = dt.predict_proba(X_test)


# In[22]:


def auc_roc_plot(y_prob):
    sns.set(font_scale=1)
    sns.set_color_codes("muted")
    plt.figure(figsize=(8, 8))
    tpr_arr = []
    fpr_arr = []
    prob = np.sort(np.unique(y_prob[:, 1]))[::-1]
    for th in np.arange(1, 0, -0.01):
        conf = confusion_matrix_prob(y_prob[:, 1], y_test, th)
        tpr_arr.append(tpr(conf))
        fpr_arr.append(fpr(conf))
    print(pd.DataFrame({'tpr': tpr_arr, 'fpr': fpr_arr}))

    plt.plot([0] + fpr_arr + [1], [0] + tpr_arr + [1], lw=2, label='ROC')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


# In[23]:


auc_roc_plot(y_prob)


# ##### Проверка 

# In[24]:


from sklearn.metrics import roc_curve, auc
fpr_clf, tpr_clf, thresholds = roc_curve(y_test, y_prob[:, 1])
print(pd.DataFrame({'tpr': tpr_clf, 'fpr': fpr_clf, 'thresholds': thresholds}))
roc_auc = auc(fpr_clf, tpr_clf)
plt.figure(figsize=(5, 5))
plt.plot(fpr_clf, tpr_clf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.grid()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])


#  #### AUC-PR

# In[25]:


def auc_pr_plot(y_prob, y_test):
    sns.set(font_scale=1)
    sns.set_color_codes("muted")
    plt.figure(figsize=(8, 8))
    p_arr = [1]
    r_arr = [0]
    y_prob = y_prob[:, 1]
    prob = np.sort(np.unique(y_prob))[::-1]
    dtype = [('prob', 'float'), ('test', 'float')]
    array = [(prob, test) for prob, test in zip(y_prob, y_test)]
    a = np.array(array, dtype= dtype)
    a = np.sort(a, order='prob')
    y_prob, y_test = a['prob'], a['test'] 
    for th in prob:
        conf = confusion_matrix_prob(y_prob, y_test, th)
        p_arr.append(precision(conf))
        r_arr.append(recall(conf))
    print(pd.DataFrame({'Recall': r_arr, 'Precision': p_arr}))
    p_arr.append(0)
    r_arr.append(1)
    plt.plot(r_arr, p_arr, lw=2, label='PR')
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


# In[26]:


y_prob = dt.predict_proba(X_test)
auc_pr_plot(y_prob, y_test)


# In[27]:


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob[:,1])
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
     plt.plot(recalls, precisions, lw=2)
     plt.xlabel('Recall')
     plt.ylabel('Precision')
     plt.title('Precision-Recall curve')
     plt.show()

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

