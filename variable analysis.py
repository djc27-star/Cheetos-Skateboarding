# load libraries
from pandas import read_csv
from pandas import isna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error
import statsmodels.api as smf 

# read dataset
data = read_csv("housing.csv")

# check for na 
isna(data).sum() # 207 missing values in total_bedrooms
data = data.dropna()

col = data.columns

# go threw each column to check values are correct

# median house values
data['median_house_value'].dtypes # dtype('float64')
for i in data['median_house_value']:
    assert i > 0 # all values are greater than 0

# longitude
# range -125 - -114 via google maps
data['longitude'].dtypes # dtype('float64')
for i in data['longitude']:
    assert -125 <= i <= -114 # all values in range

# 'latitude'
# range 32 - 43
data['latitude'].dtypes # dtype('float64')
for i in data['latitude']:
    assert 32 <= i <= 43 # all values in range

# 'housing_median_age'
data['housing_median_age'].dtypes # dtype('float64')
for i in data['housing_median_age']:
    assert i > 0 & isinstance(i,int) # all values are greater than 0
    
# 'total_rooms'
data['total_rooms'].dtypes # dtype('float64')
for i in data['total_rooms']:
    assert i > 0 & isinstance(i,int) # all values are greater than 0
    
# 'total_bedrooms'
data['total_bedrooms'].dtypes # dtype('float64')
for i in data['total_bedrooms']:
    assert i > 0 & isinstance(i,int) # all values are greater than 0
    
# 'population'
data['population'].dtypes # dtype('float64')
for i in data['population']:
    assert i > 0 & isinstance(i,int)
    
# 'households'
data['households'].dtypes # dtype('float64')
for i in data['households']:
    assert i > 0 & isinstance(i,int)
    
# 'median_income'
data['median_income'].dtypes # dtype('float64')
for i in data['median_income']:
    assert i > 0 
    
# "ocean_proximity"
data["ocean_proximity"].dtypes #  dtype('O')
proximity_groups = data.groupby("ocean_proximity").size()


# remove island houses and create dummy variable
data = data[data["ocean_proximity"] != 'ISLAND']
labelencoder_x= LabelEncoder() 
data["ocean_proximity"]= labelencoder_x.fit_transform(data["ocean_proximity"]) 

# rooms per population may be better indicator than just rooms or population
data['rooms per person'] = data['total_rooms']/data['population']

# family sizes could be a strong indicator
data['family size'] = data['population']/data['households']



# explorative data analysis
corrMat = data.corr()
sns.heatmap(corrMat, annot=True)
plt.show()

# find correlation with median house values
value_corr = []
for i in range(len(data.columns)):
    value_corr.append((data.columns[i],corrMat.iloc[8,i]))

# identify highly correlated variables
var_corr = []
for i in range(len(data.columns)):
    for j in range(len(data.columns)):
                   if i == j:
                       pass
                   elif corrMat.iloc[i,j] >= 0.7 or corrMat.iloc[i,j] <= -0.7 :
                       var_corr.append((data.columns[i],data.columns[j]))
                       
# 'latitude' 'housing_median_age' 'total_rooms' 'median_income' 'rooms per person'
# correlated with 'median_house_value' and not other variables

data = data[["latitude",'housing_median_age','total_rooms','median_income',
             'median_house_value','rooms per person']]

sns.pairplot(data)
plt.show()

data['intercept'] = np.ones((20428,1))

# maybe signs of polynomial regression needed 
# try linear first and build from there 

x = data.drop("median_house_value", axis =1)
y = data["median_house_value"]

# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

# lists for results
train_results = []
test_results = []

# average model
mu = np.mean(y)
train_results.append(("average model",np.sqrt(mean_squared_error(np.ones((len(y_train),1))*mu,y_train))))
test_results.append(("average model",np.sqrt(mean_squared_error(np.ones((len(y_test),1))*mu,y_test))))
mu
# simple linear regression
regressor_1= LinearRegression()  
regressor_1.fit(x_train, y_train)
y_train_pred = regressor_1.predict(x_train)
y_test_pred = regressor_1.predict(x_test)
train_results.append(("1st model",np.sqrt(mean_squared_error(y_train_pred,y_train))))
test_results.append(("1st model",np.sqrt(mean_squared_error(y_test_pred,y_test))))

# remove insignificant variables
regressor_OLS=smf.OLS(endog = y_train, exog=x_train).fit()  
regressor_OLS.summary() # all variables are significant


