import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")

#Data analysis
print(train.head())
print(train.info())
print(train.describe())

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
train["IsAlone"] = (train["FamilySize"] == 1).astype(int)

#SexSurvived plot
sb.countplot(data=train, x="Survived", hue="Sex")
plt.savefig("sexsur_plot.png")
plt.figure()
#Age plot
sb.histplot(train["Age"].dropna())
plt.savefig("age_plot.png")
plt.figure()
#AgeSurvived plot
sb.histplot(data=train, x="Age", hue="Survived")
plt.savefig("agesur_plot.png")
plt.figure()
#Heatmap
print(train.corr(numeric_only=True))

target = "Survived"
features = ["Pclass","Sex", "Age",
            "SibSp", "Parch", "Fare",
            "Embarked", "FamilySize", "IsAlone"]

x = train[features]
y = train[target]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

num_feat = ["Age","SibSp","Parch","Fare","FamilySize", "IsAlone"]
cat_feat = ["Pclass","Sex","Embarked"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_feat),
        ("cat", cat_transformer, cat_feat)])

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, n_jobs=-1))])

clf.fit(x_train, y_train)
y_pred = clf.predict(x_valid)

val_accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation accuracy(LogisticRegression): {val_accuracy * 100:.2f}%")

clfF = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=0,
        n_jobs=-1))])

clfF.fit(x_train, y_train)
y_pred = clfF.predict(x_valid)

val_accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation accuracy(RandomForest): {val_accuracy * 100:.2f}%")

