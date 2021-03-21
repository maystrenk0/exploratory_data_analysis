import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('StudentsPerformance.csv', usecols=['math score'])
print(df['math score'].describe())
print('var: ' + str(float(df['math score'].var())))
std = float(df['math score'].std())
print('data range: ' + str(float(df['math score'].max()-df['math score'].min())))
print('var coef: ' + str(std*100/df['math score'].mean()) + '%')
iqr = float(df['math score'].quantile(0.75)-df['math score'].quantile(0.25))
print('IQR: ' + str(iqr))
print('Q10: ' + str(float(df['math score'].quantile(0.1))))
print('Q90: ' + str(float(df['math score'].quantile(0.9))))
print('skewness: ' + str(df['math score'].skew()))
print('kurtosis: ' + str(df['math score'].kurtosis()))
print("Shapiro-Wilk test: " + str(stats.shapiro(df['math score'])))
# >0.05 - normal

basic = int(round(1+math.log(df['math score'].count(), 2)))
scott = int(round(pow(df['math score'].count(), 1.0/3)/3.5*std))
freedmandiaconis = int(round(pow(df['math score'].count(), 1.0/3)/2*iqr))

sns.set(style='whitegrid')
boxplt = sns.boxplot(y=df['math score'])
plt.show()
distplt = sns.distplot(df)

mu = df['math score'].mean()
sigma = std
x = np.linspace(df['math score'].min(), df['math score'].max(), 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()
stats.probplot(df['math score'], dist="norm", plot=sns.mpl.pyplot)
plt.show()

