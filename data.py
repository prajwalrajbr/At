# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import missingno as msno
import seaborn as sns

"""## 1.2 ) Reading the data from a CSV file"""
r='WA_Fn-UseC_-HR-Employee-Attrition.csv'
df=pd.read_csv(r)

dat=input("enter the column name")
##msno.matrix(df) # just to visuali
##plt.savefig("graphs\matrixplot.jpg")
plt.figure(figsize=(20,10))
sns.countplot(x=df[dat],data=df,hue='Attrition',palette="twilight_shifted",saturation=2,dodge=True,)

plt.savefig("graphs\Age&attrition.png")

