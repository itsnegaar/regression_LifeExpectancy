from cgi import print_arguments
from operator import inv
from random import seed
from cv2 import LMEDS, dft
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sb
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import datetime as dt
from sklearn import linear_model as lm
from sklearn.metrics import mean_absolute_error , mean_squared_error
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d




#___________Load And Clear data_________________
pd.options.display.max_rows = 9999
data = pd.read_csv('/home/veunex/a/MachineLearning/regression/archive/Life Expectancy Data.csv',na_values ='')
df = pd.DataFrame(data)

#cleardata

#remove duplicates
df.duplicated() 
df.drop_duplicates(inplace = True)

#find and replace nan with mode 
# print(df.info())
for clm in df :
    x = df[clm].mode(dropna=True)[0]
    df[clm].fillna(x, inplace=True)

df.to_csv('new.csv')

# sb.pairplot(df)

# shuffle the DataFrame rows
df = df.sample(frac = 1)

#split dataset to train, validation and test
# train , test1 = train_test_split(df,test_size=0.2)
# test , valid = train_test_split(test1,test_size=0.5)


#____________Linear Regression________________
# dfcorr = df.corr()
# df.corr().to_csv('corr.csv') 
# df.plot.scatter(x='Life expectancy ', y = 'Schooling')
# plt.xlabel("Life expectancy")
# plt.ylabel('Schooling')
# plt.show()

# data_plot = df.loc[:,['Life expectancy ','Schooling']]
# data_plot.plot()
# data.plot(kind = "hist",y ='Life expectancy ',bins = 50,range= (0,50),normed = True)
# plt.figure(figsize=(15,10))
# plt.tight_layout()
# sb.histplot(df['Life expectancy '])

# reg = LinearRegression()

# x = train['Life expectancy '].values.reshape(-1,1)
# y = train['Schooling'].values.reshape(-1,1)
# skLmodel = reg.fit(x,y)
# yLpredict = skLmodel.predict(x,y)
# plt.scatter(x,y,color="lightblue")
# plt.plot(x,reg.predict(x),color="blue")
# plt.title('Linear Regression(training Set)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()


#_________________testing______________________
# xtest = test['Life expectancy '].values.reshape(-1,1)
# ytest = test['Schooling'].values.reshape(-1,1)
# model=reg.fit(x,y)
# predictions = model.predict(xtest)
# plt.scatter(ytest,predictions)
# plt.title('Linear Regression(testing Set)')
# plt.xlabel('True')
# plt.ylabel('Predict')
# plt.show()

#____________Multiple Regression_______________
# dfcorr = df.corr()['Life expectancy ']
# # print(dfcorr)
# LifeExpectancy = df['Life expectancy ']
# # print(df.columns)
# ListOfDroped=['Country', 'Year', 'Status', 'Life expectancy ','infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
#        'Measles ','under-five deaths ', 'Polio', 'Total expenditure',
#        'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
#        ' thinness  1-19 years', ' thinness 5-9 years']
# dfdrop = df.drop(ListOfDroped,axis =1)
# dfdrop['Life expectancy '] = LifeExpectancy
# # plt.scatter(dfdrop['Life expectancy '],dfdrop['Income composition of resources'])
# # plt.xlabel('Life expectancy ')
# # plt.ylabel('Income composition of resources')
# # plt.show()

# # plt.scatter(dfdrop['Life expectancy '],dfdrop['Status'])
# # plt.xlabel('Life expectancy ')
# # plt.ylabel('Status')
# # plt.show()
# reg = LinearRegression()
# dfnp = dfdrop.to_numpy()
# xtrain , ytrain = dfnp[:,:4],dfnp[:,-1] 
# skmodel = reg.fit(xtrain,ytrain)
# ypredict = skmodel.predict(xtrain)
# print(ypredict)
# coef = skmodel.coef_
# intercept = skmodel.intercept_
# #measure the error
# meansq = mean_squared_error(ypredict,ytrain)
# meanab = mean_absolute_error(ypredict,ytrain)
# print('meansq' , meansq)
# print('meanab' , meanab)
# print(dfdrop.columns)
# predicteddf = pd.DataFrame({'Adult Mortality':dfdrop['Adult Mortality'],
#                           ' BMI ':dfdrop[' BMI '] , 
#                           'Income composition of resources' : dfdrop['Income composition of resources'],
#                           'Schooling' : dfdrop['Schooling'] ,
#                           'Life expectancy ':dfdrop['Life expectancy '] ,
#                           'ypredict' : ypredict,
#                           'meanab' : meanab})

# print(predicteddf)

# def getpredict(model,X):
#     (n,pmin) = X.shape
#     p = pmin + 1
#     newX = np.ones(shape = (n,p))
#     newX[:, 1:] = X
#     return np.dot(newX , model)


# def GetBestModel(X,y):
#     (n,pmin) = X.shape
#     p = pmin + 1
#     newX = np.ones(shape = (n,p))
#     newX =[:,1:] = X
#     return np.dot(np.dot(inv(np.dot((newX , newX)), newx.T)),y)


# fig = plt.figure()
# ax = fig.gca(projection ='3d')
 
# ax.scatter(x[:, 1], x[:, 2], y, label ='y',
#                 s = 5, color ="dodgerblue")
 
# ax.scatter(x[:, 1], x[:, 2], c[0] + c[1]*x[:, 1] + c[2]*x[:, 2],
#                     label ='regression', s = 5, color ="orange")
 
# ax.view_init(45, 0)
# ax.legend()
# plt.show()
dfcorr = df.corr()['Life expectancy ']
# print(dfcorr)


LifeExpectancy = df['Life expectancy ']
LDroped = ['Country', 'Year', 'Status', 'Life expectancy ',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources']

dfdrop2 = df.drop(LDroped,axis =1)
dfdrop2['Life expectancy '] = LifeExpectancy
dfnp2 = dfdrop2.to_numpy()
train , test1 = train_test_split(dfnp2,test_size=0.2)
test , valid = train_test_split(test1,test_size=0.5)
xtrain2 , ytrain2 = dfnp2[:,:1],dfnp2[:,-1] 
xtest2 , ytest2 = dfnp2[:,:1],dfnp2[:,-1] 
reg = LinearRegression() 
skmodel2 = reg.fit(xtrain2,ytrain2)
ypredict2 = skmodel2.predict(xtrain2)   
# print(ypredict2)
# coef2 = skmodel2.coef_
# intercept2 = skmodel2.intercept_


#measure the error
meansq2 = mean_squared_error(ypredict2,ytest2)
meanab2 = mean_absolute_error(ypredict2,ytest2)
print('mean ab' ,meanab2)

predicteddf2 = pd.DataFrame({'Adult Mortality':dfdrop2['Adult Mortality'], 
                          'Schooling' : dfdrop2['Schooling'] ,
                          'Life expectancy ':dfdrop2['Life expectancy '] ,
                          'ypredict' : ypredict2,
                          'meanab' : meanab2})


fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.scatter(xtrain2, ytrain2, ypredict2, color = 'lightblue')
ax.plot_surface(xtrain2,ytrain2,ypredict2.reshape(xtrain2.shape), color='b', alpha=0.3)
plt.title('Matplot 3d scatter plot')
ax.set_xlabel('Adult Mortality')
ax.set_ylabel('schooling')
ax.set_zlabel('Life expectancy prediction - linear')
plt.legend(loc=2)
plt.show()


clf = Ridge(alpha=1.0)
skmodel3 = clf.fit(xtrain2,ytrain2)
ypredict3 = skmodel3.predict(xtrain2)   
meanab2 = mean_absolute_error(ypredict3,ytest2)
print('mean ab',meanab2)
fig2 = plt.figure()
ax = fig2.gca(projection='3d')
ax.scatter(xtrain2, ytrain2, ypredict3, color = 'lightblue')
ax.plot_surface(xtrain2,ytrain2,ypredict3.reshape(xtrain2.shape), color='b', alpha=0.3)
plt.title('Matplot 3d scatter plot - nonlinear')
ax.set_xlabel('Adult Mortality')
ax.set_ylabel('schooling')
ax.set_zlabel('Life expectancy prediction')
plt.show()
