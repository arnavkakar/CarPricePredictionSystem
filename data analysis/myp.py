

########## I have added the google colab link #################
# https://colab.research.google.com/drive/17BqyXG_bBxdXMvJaNw9Pcx0aoMdfy4kT?usp=sharing



#importing modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from subprocess import check_output


# Any results you write to the current directory are saved as output.

#Read dataset
df=pd.read_csv("car_ad.csv", encoding='latin-1') #reading the csv file

df = df.drop(df[df.price <= 0 ].index)

df.price[df.price ==0].count()

df.head(3)

#df = df.price[df.price > 0]
#Drop column where price is zero

df = df.drop(df[df.price == 0].index)

df.price[df.price == 0].count()

df.shape

sns.countplot(df['body'])

sns.boxplot(x='mileage',y='price',data=df)

sns.regplot(x='mileage',y='price',data=df)

sns.countplot(df['registration'])

sns.boxplot(x='registration',y='price',data=df)

sns.regplot(x='engV',y='price',data=df)

sns.countplot(df['engType'])

#hot encode
from sklearn import model_selection, preprocessing
for c in df.columns:
    if df[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))
        
df.head(3)

y_train = df["price"]
x_train = df.drop(["price"], axis=1)

data_train, data_test, label_train, label_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

x_train.head(3)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

df.price.skew()

label_test_log=np.log(label_test)

label_train_log=np.log(label_train)

dtrain_log = xgb.DMatrix(data_train, label_train_log)

label_train1=np.log(label_train)

label_train_log.skew()

dtrain = xgb.DMatrix(data_train, label_train)

#without log transform
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

#with log transform
cv_output_log = xgb.cv(xgb_params, dtrain_log, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

#with log transform
num_boost_rounds = len(cv_output)
model_log = xgb.train(dict(xgb_params, silent=0), dtrain_log, num_boost_round= 

num_boost_rounds)


#pickling the current model to use it in views.py
import pickle
filename = 'finalized_car_model.sav'
pickle.dump(model, open(filename, 'wb'))

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

#without log transformation
dtest=xgb.DMatrix(data_test)

data_test.head(2)

#without log
loaded_model = pickle.load(open(filename, 'rb'))
y_predict = loaded_model.predict(dtest)
out = pd.DataFrame({'Actual_price': label_test, 'predict_price': y_predict,'Diff' :(label_test-y_predict)})
out[['Actual_price','predict_price','Diff']].head(5)
# for index, row in out.iterrows():
#     print(row['predict_price'], row['Actual_price'])

#with log transformation
y_predict_log = model_log.predict(dtest)
y_predict_log=np.exp(y_predict_log)
out_log = pd.DataFrame({'Actual_price': label_test, 'predict_price': y_predict_log,'Diff' :(label_test-y_predict_log)})
out_log[['Actual_price','predict_price','Diff']].head(5)






