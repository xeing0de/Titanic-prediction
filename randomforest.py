import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Data analysis
print(train.head())
print(train.info())
print(train.describe())

#SexSurvived plot
sb.countplot(data=train, x='Survived', hue='Sex')
plt.savefig('sexsur_plot.png')
plt.figure()
#Age plot
sb.histplot(train['Age'].dropna())
plt.savefig('age_plot.png')
plt.figure()
#AgeSurvived plot
sb.histplot(data=train, x='Age', hue='Survived')
plt.savefig('agesur_plot.png')
plt.figure()
#Heatmap
print(train.corr(numeric_only=True))
