import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the house data into a data frame
data = pd.read_csv('data.csv')


x= data['Serial_No'].values
x1= data['timestamp'].values
y= data['Level_sensor'].values
y1= data['sensor_01'].values
y5=data['sensor_05'].values

#To plot original data
ax = plt.gca()
ax.scatter(x,y, color="b", s=5, marker="s", label='Level')
ax.scatter(x,y1, color="y",s=5, marker="*", label='Sensor 1')
ax.scatter(x,y5, color="r",s=5,marker="o", label='Sensor 5')

m=len(x)
x=x.reshape((m,1))

reg=LinearRegression()
reg1=LinearRegression()
reg2=LinearRegression()

reg=reg.fit(x, y1)
reg1=reg1.fit(x, y)
reg2=reg2.fit(x, y5)

y1_pred=reg.predict(x)
y_pred=reg1.predict(x)
y5_pred=reg2.predict(x)

mse=mean_squared_error(y1,y1_pred)
rmse=np.sqrt(mse)
r2_score=reg.score(x,y1)

ax = plt.gca()

ax.scatter(x,y, color="b", s=5, marker="s", label='Level')
ax.scatter(x,y1, color="y",s=5, marker="*", label='Sensor 1')
ax.scatter(x,y5, color="r",s=5,marker="o", label='Sensor 5')
 
plt.plot(x1, y1_pred, color ='k') 
plt.plot(x1, y_pred, color ='r') 
plt.plot(x1, y5_pred, color ='b')
plt.legend()
plt.show()

print(reg.intercept_)
print(reg.coef_)

print(reg2.intercept_)
print(reg2.coef_)

print(reg1.intercept_)
print(reg1.coef_)