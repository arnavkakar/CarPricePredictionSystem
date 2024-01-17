from django.conf import settings
from django.shortcuts import render
import pandas as pd # data processing
import matplotlib.pyplot as plt 
import xgboost as xgb
import numpy as np
from subprocess import check_output

# Create your views here.
def index(request):
    if request:
        return render(request, 'index.html')


def about(request):
    if request:
        return render(request, 'about.html')


def getstarted(request):
    if request:
        return render(request, 'getstarted.html')


def result(request):
    
    if request.GET:
        #taking the inputs from the user
        car=(request.GET['car'])
        price=(int(request.GET['price']))
        body=(request.GET['body'])
        mileage=(int(request.GET['mileage']))
        engV=(float(request.GET['engV']))
        engType=(request.GET['engType'])
        registration=(request.GET['registration'])
        year=(int(request.GET['year']))
        model=(request.GET['model'])
        drive=(request.GET['drive'])
        data= {
            'car':[car],
            'price':[price],
            'body':[body],
            'mileage':[mileage],
            'engV':[engV],
            'engType':[engType],
            'registration':[registration],
            'year':[year],
            'model':[model],
            'drive':[drive]
        }
        df=pd.DataFrame(data)
        df = df.drop(df[df.price <= 0 ].index)
        df = df.drop(df[df.price == 0].index)

        from sklearn import model_selection, preprocessing
        for c in df.columns: 
            if df[c].dtype == 'object':
               lbl = preprocessing.LabelEncoder()
               lbl.fit(list(df[c].values))
               df[c] = lbl.transform(list(df[c].values))
        y_train = df["price"]
        x_train = df.drop(["price"], axis=1)  
        xgb_params = {
                       'eta': 0.05,
                       'max_depth': 5,
                       'subsample': 0.7,
                       'colsample_bytree': 0.7,
                       'objective': 'reg:linear',
                       'eval_metric': 'rmse',
                       'silent': 1
                    }  


        #using the pickled model we trained in data analysis folder
        import pickle
        filename = 'finalized_car_model.sav'

        #without log transformation
        dtest=xgb.DMatrix(x_train)


        #without log
        loaded_model = pickle.load(open(filename, 'rb'))
        y_predict = loaded_model.predict(dtest)
        out = pd.DataFrame({'Actual_price': y_train, 'predicted_price': y_predict,'Diff' :(y_train-y_predict)})
        ans = y_predict
        print(ans)

        return render(request, 'getstarted.html',{'ans':ans})

