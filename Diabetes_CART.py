
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile
import graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz, export_text
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

from helpers.eda import *
pd.set_option('display.max_columns', None)

######
# veriseti bilgileri
########

################################
# DATA PREPROCESSING
################################

# ağaç yöntemi kullanacağız bu sebeple data preprocessing ile işimiz yok

df_ = pd.read_csv("datasets/diabetes.csv")
df = df_.copy()
df.head()
df.iloc[:, 9:].head()
y = df["Outcome"]
X = df.iloc[:, 9:]
y = df["Outcome"]
X = df.drop("Outcome", axis=1)
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)

################################
# HOLDOUT YÖNTEMİ İLE MODEL DOĞRULAMA
################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
## test için roc değeri: 0.6739
## yeni değişkenler türetince yeni hata 0.76


########################################
# FEATURE ENGINEERING
########################################

# BMI
df.head()
df.describe()
df.loc[df["BMI"] < 30, "BMI_category"] = 0
df.loc[df["BMI"] >= 30, "BMI_category"] = 1

# Healthy persons are concentrate with an blood pressure <= 80
df.loc[df["BloodPressure"] < 80, "BloodPressure_category"] = 0
df.loc[df["BloodPressure"] >= 80, "BloodPressure_category"] = 1

# Healthy persons are concentrate with an glucose <= 105
df.loc[df["Glucose"] < 105, "Glucose_category"] = 0
df.loc[df["Glucose"] >= 105, "Glucose_category"] = 1

df.loc[(df['SkinThickness'] < 20), 'SkinThickness_category'] = 0
df.loc[(df['SkinThickness'] >= 20), 'SkinThickness_category'] = 1

df.loc[(df['Insulin'] < 200), "Insulin_category"] = 0
df.loc[(df['Insulin'] >= 200), "Insulin_category"] = 1

df.loc[(df['Pregnancies'] < 6), "Pregnancies_category"] = 0
df.loc[(df['Pregnancies'] >= 6), "Pregnancies_category"] = 1

################################
# HİPERPARAMETRE OPTİMİZASYONU
################################

cart_model = DecisionTreeClassifier(random_state=17)

# arama yapılacak hiperparametre setleri
cart_params = {'max_depth': range(1, 8),
               "min_samples_split": [2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)

# # train hatası
# y_pred = cart_tuned.predict(X_train)
# y_prob = cart_tuned.predict_proba(X_train)[:, 1]
# print(classification_report(y_train, y_pred))
# roc_auc_score(y_train, y_prob)

y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

