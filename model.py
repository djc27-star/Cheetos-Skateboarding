#load libraries
from pandas import read_csv
import math
import numpy as np
from pandas import isna
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import get_dummies
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as smf 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# read dataset
data = read_csv("housing.csv")

# check for na 
isna(data).sum() # 207 missing values in total_bedrooms
data = data.dropna()

# create new columns based on house distance from major cities
# longitude and latitude need to be converted to radians for calculations
# calculations with haversine formula
data["longitude"] = math.pi*data["longitude"]/180
data["latitude"] = math.pi*data["latitude"]/180

#los angelos 0.5943455223 -2.063770159
data["dist to los angeles"] = 3959*2*np.arcsin(np.sqrt(np.sin((0.5943455223-data["latitude"])/2)**2
                            +np.cos(0.5943455223)*np.cos(data["latitude"])*
                            np.sin((-2.063770159-data["longitude"])/2)**2))

# san diego 0.5709844648 -2.044871385
data["dist to san diego"] = 3959*2*np.arcsin(np.sqrt(np.sin((0.5709844648-data["latitude"])/2)**2
                            +np.cos(0.5709844648)*np.cos(data["latitude"])*
                            np.sin((-2.044871385-data["longitude"])/2)**2))

# san jose 0.6515895641 -2.127556358
data["dist to san jose"] = 3959*2*np.arcsin(np.sqrt(np.sin((0.6515895641-data["latitude"])/2)**2
                            +np.cos(0.6515895641)*np.cos(data["latitude"])*
                            np.sin((-2.127556358-data["longitude"])/2)**2))

# san francisco 0.6593829479 -2.136790023
data["dist to san francisco"] = 3959*2*np.arcsin(np.sqrt(np.sin((0.6593829479-data["latitude"])/2)**2
                            +np.cos(0.6593829479)*np.cos(data["latitude"])*
                            np.sin((-2.136790023-data["longitude"])/2)**2))

# fresno 0.6415491736 -2.090555204
data["dist to fresno"] = 3959*2*np.arcsin(np.sqrt(np.sin((0.6415491736 -data["latitude"])/2)**2
                            +np.cos(0.6415491736 )*np.cos(data["latitude"])*
                            np.sin((-2.090555204-data["longitude"])/2)**2))

# sacramento 0.6733363317 -2.120416216
data["dist to sacramento"] = 3959*2*np.arcsin(np.sqrt(np.sin((0.6733363317 -data["latitude"])/2)**2
                            +np.cos(0.6733363317 )*np.cos(data["latitude"])*
                            np.sin((-2.120416216-data["longitude"])/2)**2))

data["dist to nearest city"] = data.loc[:, ["dist to los angeles",
                            "dist to san diego","dist to san jose",
                            "dist to san francisco","dist to fresno",
                            "dist to sacramento"]].min(axis=1)

del data["longitude"]
del data["latitude"]

# create new columns by combining other columns 
data['rooms per person'] = data['total_rooms']/data['population']

data['family size'] = data['population']/data['households']

data['bedrooms per person'] = data['total_bedrooms']/data['population']

data['rooms per bedroom'] = data['total_rooms']/data['total_bedrooms']

# go threw data checking values are correct and to explore features
col = data.columns
data.describe()

# plot graphs for each variable to visualise data

# housing_median_age
sns.distplot( data['housing_median_age'])
plt.show()
sns.scatterplot(x = 'housing_median_age', y = 'median_house_value',data =data)
plt.show()

# total_rooms
sns.distplot( data['total_rooms'])
plt.show()
sns.scatterplot(x = 'total_rooms', y = 'median_house_value',data =data)
plt.show()

# total_bedrooms
sns.distplot( data['total_bedrooms'])
plt.show()
sns.scatterplot(x = 'total_bedrooms', y = 'median_house_value',data =data)
plt.show()

# population
sns.distplot( data['population'])
plt.show()
sns.scatterplot(x = 'population', y = 'median_house_value',data =data)
plt.show()

# households
sns.distplot( data['households'])
plt.show()
sns.scatterplot(x = 'households', y = 'median_house_value',data =data)
plt.show()

# median_income
sns.distplot( data['median_income'])
plt.show()
sns.scatterplot(x = 'median_income', y = 'median_house_value',data =data)
plt.show()

# dist to los angeles
sns.distplot( data['dist to los angeles'])
plt.show()
sns.scatterplot(x = 'dist to los angeles', y = 'median_house_value',data =data)
plt.show()

# dist to san diego
sns.distplot( data['dist to san diego'])
plt.show()
sns.scatterplot(x = 'dist to san diego', y = 'median_house_value',data =data)
plt.show()

# dist to san jose
sns.distplot( data['dist to san jose'])
plt.show()
sns.scatterplot(x = 'dist to san jose', y = 'median_house_value',data =data)
plt.show()

# dist to san francisco
sns.distplot( data['dist to san francisco'])
plt.show()
sns.scatterplot(x = 'dist to san francisco', y = 'median_house_value',data =data)
plt.show()

# dist to fresno
sns.distplot( data['dist to fresno'])
plt.show()
sns.scatterplot(x = 'dist to fresno', y = 'median_house_value',data =data)
plt.show()

# dist to sacramento
sns.distplot( data['dist to sacramento'])
plt.show()
sns.scatterplot(x = 'dist to sacramento', y = 'median_house_value',data =data)
plt.show()

# dist to nearest city
sns.distplot( data['dist to nearest city'])
plt.show()
sns.scatterplot(x = 'dist to nearest city', y = 'median_house_value',data =data)
plt.show()

# rooms per person
sns.distplot( data['rooms per person'])
plt.show()
sns.scatterplot(x = 'rooms per person', y = 'median_house_value',data =data)
plt.show()

# family size
sns.distplot( data['family size'])
plt.show()
sns.scatterplot(x = 'family size', y = 'median_house_value',data =data)
plt.show()

# housing_median_age
sns.distplot( data['bedrooms per person'])
plt.show()
sns.scatterplot(x = 'bedrooms per person', y = 'median_house_value',data =data)
plt.show()

# rooms per bedroom
sns.distplot( data['rooms per bedroom'])
plt.show()
sns.scatterplot(x = 'rooms per bedroom', y = 'median_house_value',data =data)
plt.show()

# ocean_proximity
sns.boxplot(x = 'ocean_proximity', y = 'median_house_value',data =data)
plt.show()

# median_house_value
sns.distplot( data['median_house_value'])
plt.show()

# create dummy variables for ocean_proximity
dummy = get_dummies(data["ocean_proximity"]) 
dummy = dummy.reset_index()
del data["ocean_proximity"]

# standardise data
names = data.columns
st_x= StandardScaler()    
data= st_x.fit_transform(data)  
data = DataFrame(data,columns = names)

# add dummy variables and intercept
data['intercept'] = np.ones((20433,1))
data["INLAND"] = dummy["INLAND"]
data["NEAR BAY"] = dummy["NEAR BAY"]
data["NEAR OCEAN"] = dummy["NEAR OCEAN"]
data["<1H OCEAN"] = dummy["<1H OCEAN"]
data["ISLAND"] = dummy["ISLAND"]
del dummy

# seperate data into target and explanatory variables
x = data.drop('median_house_value',axis = 1)
y = data['median_house_value']

seed(81)
X_train, X_test, Y_train, Y_test = train_test_split(x,
         y, test_size=0.20, random_state=1, shuffle=True)

# lists for results
test_results = []

# average model
mu = np.mean(Y_train)
test_results.append(("average model",np.sqrt(mean_squared_error(np.ones((len(Y_test),
                                                                    1))*mu,Y_test))))

# linear regression with given variables
regressor_1= LinearRegression()  
regressor_1.fit(X_train[['housing_median_age', 'total_rooms', 'total_bedrooms', 
                        'population','households', 'median_income']], Y_train)
y_test_pred = regressor_1.predict(X_test[['housing_median_age', 'total_rooms',
                'total_bedrooms', 'population','households', 'median_income']])
test_results.append(("lm given variables",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# linear regression with all variables
regressor_2= LinearRegression()  
regressor_2.fit(X_train, Y_train)
y_test_pred = regressor_2.predict(X_test)
test_results.append(("lm all variables",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# linear regression by removing insignificant variables
X_train2 = X_train[:]
X_test2 = X_test[:]
regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # del total rooms
del X_train2['total_rooms']
del X_test2['total_rooms']

regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # NEAR OCEAN 
del X_train2['NEAR OCEAN']
del X_test2['NEAR OCEAN']

regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # family size
del X_train2['family size']
del X_test2['family size']

regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # rooms per person
del X_train2['rooms per person']
del X_test2['rooms per person']

regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # total bedrooms
del X_train2['total_bedrooms']
del X_test2['total_bedrooms']

regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # rooms per bedroom 
del X_train2['rooms per bedroom']
del X_test2['rooms per bedroom']

regressor_OLS=smf.OLS(endog = Y_train, exog=X_train2).fit()  
regressor_OLS.summary() # all variables significant

regressor_3= LinearRegression()  
regressor_3.fit(X_train2, Y_train)
y_test_pred = regressor_3.predict(X_test2)
test_results.append(("lm significant variables",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# try a ridge model to deal with multicollinarity
regressor_4 = Ridge()
regressor_4.fit(X_train,Y_train)
y_test_pred = regressor_4.predict(X_test)
test_results.append(("Ridge",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# optimise alpha
rs = {'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01]}
grid = GridSearchCV(estimator = Ridge(), param_grid = rs,
                    cv =10,scoring = 'neg_mean_squared_error')
grid.fit(X_train,Y_train)
grid.best_params_
regressor_5 = Ridge(alpha = 0.5)
regressor_5.fit(X_train,Y_train)
y_test_pred = regressor_5.predict(X_test)
test_results.append(("Ridge optimised",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# try random forest
regressor_6 = RandomForestRegressor()
regressor_6.fit(X_train,Y_train)
y_test_pred = regressor_6.predict(X_test)
test_results.append(("rf",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# remove features by feature importance
plt.bar(X_train.columns,regressor_6.feature_importances_)
plt.xticks(rotation=90)
plt.show()

variable_importance = [(X_train.columns[list(regressor_6.feature_importances_).index(i)],i) for i in 
sorted(regressor_6.feature_importances_, reverse = True)]

sorted_importances = [x[1] for x in variable_importance]
sorted_features = [x[0] for x in variable_importance]
cumulative_importances = np.cumsum(sorted_importances)
plt.plot(sorted_features, cumulative_importances, 'g-')
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r',
           linestyles = 'dashed')
plt.xticks(rotation = 90)
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance')
plt.title('Cumulative Importances')
plt.show()

# try model with strongest variables
X_train3 = X_train[sorted_features[0:13]]
X_test3 = X_test[sorted_features[0:13]]
regressor_7 = RandomForestRegressor()
regressor_7.fit(X_train3,Y_train)
y_test_pred = regressor_7.predict(X_test3)
test_results.append(("rf strong variables",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# not much information lost for a quicker model will now optimise
'''rs = {"n_estimators" : [100,300,500],
            "min_samples_split" : [2,4,8],            
            'max_depth': [2,4,6]
            }
grid = GridSearchCV(estimator = RandomForestRegressor(), param_grid = rs,
                    cv =3,scoring = 'neg_mean_squared_error')
grid.fit(X_train3,Y_train)
grid.best_params_'''

regressor_8 = RandomForestRegressor(max_depth = 6, min_samples_split =2,
                                    n_estimators = 500)
regressor_8.fit(X_train3,Y_train)
y_test_pred = regressor_8.predict(X_test3)
test_results.append(("rf optimise",np.sqrt(mean_squared_error(y_test_pred,Y_test))))


# #try boosting
regressor_9 = GradientBoostingRegressor()
regressor_9.fit(X_train3,Y_train)
y_test_pred = regressor_9.predict(X_test3)
test_results.append(("gradient boosting",np.sqrt(mean_squared_error(y_test_pred,Y_test))))

# optimise model
'''rs = {"n_estimators" : [100,300,500],
      "max_depth" : [2,4,6],
      "learning_rate" : [0.01,0.1,1]
            }
grid = GridSearchCV(estimator = GradientBoostingRegressor(), param_grid = rs,
                    cv =3,scoring = 'neg_mean_squared_error')
grid.fit(X_train3,Y_train)
grid.best_params_'''

# #try boosting
regressor_10 = GradientBoostingRegressor(n_estimators=500, max_depth = 6,
                                        learning_rate = 0.1)
regressor_10.fit(X_train3,Y_train)
y_test_pred = regressor_10.predict(X_test3)
test_results.append(("gradient boosting optimise",np.sqrt(mean_squared_error(y_test_pred,Y_test))))