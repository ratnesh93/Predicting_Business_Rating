"""
Author: Ratnesh Chandak
versions:
    Python 3.7.4
    pandas==0.25.1
    scikit-learn==0.21.3

"""

def weightedAverage(a,b):
    weightedAvg=0.0
    l=len(a)
    
    #normalization
    aRange=5.0 #max-min
    bRange=2.0 #max-min
    
    a[:] = [x / aRange for x in a]
    b[:] = [x / bRange for x in b]
    
    for i in range(l):
        weightedAvg+=a[i]*b[i]
    return weightedAvg/l

def gettingSentiment(data):
    user_Weighted_Average=[]
    for i in range(len(data)):
        text=data.iloc[i].reviews_list[1:-1].split(")")
        userRatings=[]
        sentiments=[]
        for review in text:
            #print(review)
            User_Rating=re.search("/d(.)/d",review)
            #unwantedPattern=""
            if User_Rating!=None:
                userRatings.append(float(User_Rating.group()))
                unwantedPattern=User_Rating.group()+"|(Rated)|[^\w]"
            else:
                userRatings.append(1.0)
                unwantedPattern="(Rated)|[^\w]"
            
            #print(unwantedPattern)
            b=re.sub(unwantedPattern," ",review,flags=re.IGNORECASE)

            analysis=TextBlob(b)
            sentiments.append(analysis.sentiment.polarity)

        if(len(userRatings)>0):
            user_Weighted_Average.append(weightedAverage(userRatings,sentiments))

    return user_Weighted_Average

import pandas as pd
from textblob import TextBlob
import re
from time import time
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

dataFull=pd.read_excel('zomato_train.xlsx',index_col=False)
data=dataFull.copy()

t0 = time()

total_NaN_Rates=data['rates'].isna().sum()
print("total Nan in column rates which we will be taking as y is : ",total_NaN_Rates)
print("---------dropping NaN y rows---------------")

data['rates'] = pd.to_numeric(data['rates'],errors='coerce')
data.dropna(subset=['rates'], inplace=True)
y=data['rates']

data.drop(['rates'],axis=1,inplace=True)

print("-----------------splitting data into training and test data in 80:20 ratio-------------------")
from sklearn.model_selection import train_test_split
X_train_Full, X_test_Full, y_train, y_test = train_test_split(data,y, train_size=0.8, test_size=0.2,random_state=0)

#dropping features which might not be required for rating restuarant
#listed_in(type) and online_order has strong corelation, so dropping one
#listed_in(city) and location has strong corelation, so dropping one
droppingFeatures=['url','address','name','phone','dish_liked','cuisines','menu_item','listed_in(type)','listed_in(city)']
print("--------------dropping following features-------------")
print(droppingFeatures)

X_train = X_train_Full.copy()
X_test = X_test_Full.copy()

X_train.drop(droppingFeatures, axis=1, inplace=True)
X_test.drop(droppingFeatures, axis=1, inplace=True)


print("----------------------doing sentiment analysis-----------------")
print("----------------------this will take time about 6 mins-----------------")
#adding new column to dataframe
X_train['user_Weighted_Average']=pd.Series(gettingSentiment(X_train))
X_test['user_Weighted_Average']=pd.Series(gettingSentiment(X_test))

#removing reviews list as we have calculated sentiments from that data
X_train.drop(['reviews_list'],axis=1,inplace=True)
X_test.drop(['reviews_list'],axis=1,inplace=True)

#-----------------checking data type of each columns-------------------
#print(data.dtypes)
#"while analyzing data types column of approx_cost is object type, so converting it into float and handling error 
#by converting non-floatable object to NaN"

X_train['approx_cost(for two people)'] = pd.to_numeric(X_train['approx_cost(for two people)'],errors='coerce')
X_test['approx_cost(for two people)'] = pd.to_numeric(X_test['approx_cost(for two people)'],errors='coerce')


categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and X_train[cname].dtype == "object"]

print("------------printing columns whose unique vale is less than 10---------------------")
print(categorical_cols)

print("------------labelling categorical columns----------------")
lb_make = LabelEncoder()
for col in categorical_cols:
    X_train[col] = lb_make.fit_transform(X_train[col])
    X_test[col] = lb_make.fit_transform(X_test[col])

t1 = time()
print('data manupulation time: ',round(t1-t0,3),'s')

#print("------------data.describe()-------------")
#print(X_train.describe())
#print(X_test.describe())

X_train.drop(['location'],axis=1,inplace=True)
X_train.drop(['rest_type'],axis=1,inplace=True)
X_test.drop(['location'],axis=1,inplace=True)
X_test.drop(['rest_type'],axis=1,inplace=True)

print("------------------handling missing value by filling with mean------------------------------")

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train),columns=X_train.columns)
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test),columns=X_test.columns)

print("-------------------doing normalization of columns------------------------------------")
normalised_imputed_X_train=(imputed_X_train-imputed_X_train.min())/(imputed_X_train.max()-imputed_X_train.min())
normalised_imputed_X_test=(imputed_X_test-imputed_X_test.min())/(imputed_X_test.max()-imputed_X_test.min())


print("-------------------building model------------------------")

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

y_train=y_train.reset_index()['rates']
model.fit(normalised_imputed_X_train, y_train)
t2 = time()
print('building model time: ',round(t2-t1,3),'s')

print("-------------------predicting for test cases---------------")
predictions = pd.Series(model.predict(normalised_imputed_X_test))

t3=time()
print('predicting time: ',round(t3-t1,3),'s')

print("----------------printing mean abosolute error----------------------")

y_test=y_test.reset_index()['rates']
score = mean_absolute_error(y_test, predictions)
print('MAE:', score)

print('Total time: ',round(t3-t0,3),'s')

output = pd.DataFrame({'actual Rating': pd.Series(y_test), 'Predicted Rating': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")