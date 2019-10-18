import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


data = pd.read_csv("mobile.csv")
print(data.head(10))

x = data.iloc[:,0:20]
y = data.iloc[:,-1]


bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)

score = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(x.columns)


feature = pd.concat([columns , score] , axis=1)
feature.columns = ["Specification" , "Score"]
print(feature)

print("**************************")
#print top 10 best features
print(feature.nlargest(10 , "Score"))


