# Load libraries
from pandas import read_csv
from pandas import isna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from pandas import get_dummies
from sklearn.model_selection import train_test_split 

# Read dataset
data = read_csv("housing.csv")

# Check for na values 
isna(data).sum() # 207 missing values in total_bedrooms
# proportionally small number of missing values
# not much loss in removing them
data = data.dropna()

# Go threw each column to check values are correct
print(data.columns)

# 'median house values'
data['median_house_value'].dtypes # dtype('float64')
for i in data['median_house_value']:
    assert i > 0 # all values are greater than 0

# 'longitude'
# range -125 - -114 via google maps
data['longitude'].dtypes # dtype('float64')
for i in data['longitude']:
    assert -125 <= i <= -114 # all values in range

# 'latitude'
# range 32 - 43 via google maps
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
print(data.groupby("ocean_proximity").size())

# Few houses on island, wont be able to make accurate prediction
# remove island houses and create dummy variable
data = data[data["ocean_proximity"] != 'ISLAND']
proximity = data["ocean_proximity"]
labelencoder_x= LabelEncoder() 
data["ocean_proximity"]= labelencoder_x.fit_transform(data["ocean_proximity"]) 

# Explorative data analysis

sns.heatmap(data.corr(), annot=True)
plt.show()

# identify highly correlated variables
corrMat = data.corr()
correlated = []
for i in range(len(data.columns)):
    for j in range(len(data.columns)):
                   if i == j:
                       pass
                   elif corrMat.iloc[i,j] >= 0.7 or corrMat.iloc[i,j] <= -0.7 :
                       correlated.append((data.columns[i],corrMat.iloc[i,8], data.columns[j],corrMat.iloc[8,j]))
                       
# Remove longitude, total_bedrooms, population, households as highly correlated
# with other variables more correlated to house price

data = data[["latitude",'housing_median_age','total_rooms','median_income',
             'median_house_value','ocean_proximity']]

sns.pairplot(data)
plt.show()

# median_icome appears linear but unsure on rest
# Try linear first and then look into polynomials


# Prepare data for analysis
dummy = get_dummies(proximity) 
data['intercept'] = np.ones((20428,1))
data["INLAND"] = dummy["INLAND"]
data["NEAR BAY"] = dummy["NEAR BAY"]
data["NEAR OCEAN"] = dummy["NEAR OCEAN"]
del dummy
del data["ocean_proximity"]
del proximity

# Split into independant and dependant variables.
x = data.drop("median_house_value", axis =1)
y = data["median_house_value"]
del data

# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

