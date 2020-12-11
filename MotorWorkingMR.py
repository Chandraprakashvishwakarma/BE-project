
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv('Automobile.csv')

X = dataset.iloc[:, :]
Y = dataset.iloc[:,24:25]
X=X.drop('price',axis=1)

make=pd.get_dummies(X['make'])
aspiration=pd.get_dummies(X['aspiration'])
num_of_doors=pd.get_dummies(X['num_of_doors'])
body_style=pd.get_dummies(X['body_style'])
drive_wheels=pd.get_dummies(X['drive_wheels'])
engine_location=pd.get_dummies(X['engine_location'])
engine_type=pd.get_dummies(X['engine_type'])
num_of_cylinders=pd.get_dummies(X['num_of_cylinders'])
fuel_system=pd.get_dummies(X['fuel_system'])
horsepower_binned=pd.get_dummies(X['horsepower_binned'])


# Drop the state coulmn
X=X.drop('make',axis=1)
X=X.drop('aspiration',axis=1)
X=X.drop('num_of_doors',axis=1)
X=X.drop('body_style',axis=1)
X=X.drop('drive_wheels',axis=1)
X=X.drop('engine_location',axis=1)
X=X.drop('engine_type',axis=1)
X=X.drop('num_of_cylinders',axis=1)
X=X.drop('fuel_system',axis=1)
X=X.drop('horsepower_binned',axis=1)


# concat the dummy variables
X=pd.concat([X,make],axis=1)
X=pd.concat([X,aspiration],axis=1)
X=pd.concat([X,num_of_doors],axis=1)
X=pd.concat([X,body_style],axis=1)
X=pd.concat([X,drive_wheels],axis=1)
X=pd.concat([X,engine_location],axis=1)
X=pd.concat([X,engine_type],axis=1)
X=pd.concat([X,num_of_cylinders],axis=1)
X=pd.concat([X,fuel_system],axis=1)
X=pd.concat([X,horsepower_binned],axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
  
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

Intercept=regressor.intercept_
print("Intercept :",Intercept)
print()


coef= regressor.coef_
print("coefficient :" ,coef)
print()

score=r2_score(Y_test,Y_pred)
print("r2_score:" ,score)











