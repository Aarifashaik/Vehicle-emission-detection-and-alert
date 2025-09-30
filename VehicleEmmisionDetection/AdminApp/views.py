from django.shortcuts import render
import pymysql
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

# Create your views here.
def index(request):
    return render(request,'AdminApp/index.html')
def login(request):
    return render(request,'AdminApp/Admin.html')
def LogAction(request):
    username=request.POST.get('username')
    password=request.POST.get('password')
    if username=='Admin' and password=='Admin':      
        return render(request,'AdminApp/AdminHome.html')
    else:
        context={'data':'Login Failed ....!!'}
        return render(request,'AdminApp/Admin.html',context)
def home(request):
    return render(request,'AdminApp/AdminHome.html')
global df
def LoadData(request):
    global df
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df=pd.read_csv(BASE_DIR+"\\dataset\\CO2 Emissions_Canada.csv")
    #data.fillna(0, inplace=True)
    context={'data':"Dataset Loaded\n"}
    
    return render(request,'AdminApp/AdminHome.html',context)
global X
global y
global X_train,X_test,y_train,y_test
def split(request):
    global X_train,X_test,y_train,y_test
    global df
    #df1=pd.DataFrame({col: df[col].astype('category').cat.codes for col in df}, index=df.index)
    df['Make']=df['Make'].map({'ACURA':1,'ALFA ROMEO':2,'ASTON MARTIN':3,'AUDI':4,'BENTLEY':5,'BMW':6,'BUICK':7,'CADILLAC':8,'CHEVROLET':9,'CHRYSLER':10,'DODGE':11,'FIAT':12,'FORD':13,'GMC':14,'HONDA':15,'HYUNDAI':16,'INFINITI':17,'JAGUAR':18,'JEEP':19,'KIA':20,'LAMBORGHINI':21,'LAND ROVER':22,'LEXUS':23,'LINCOLN':24,'MASERATI':25,'MAZDA':26,'MERCEDES-BENZ':27,'MINI':28,'MITSUBISHI':29,'NISSAN':30,'PORSCHE':31,'RAM':32,'ROLLS-ROYCE':33,'SCION':34,'SMART':35,'SRT':36,'SUBARU':37,'TOYOTA':38,'VOLKSWAGEN':39,'VOLVO':40,'GENESIS':41,'BUGATTI':42})
    df['Fuel Type']=df['Fuel Type'].map({'Z':1,'D':2,'X':3,'E':4,'N':5})
    df['Transmission']=df['Transmission'].map({'AS5':1,'M6':2,'AV7':3,'AS6':4,'AM6':5,'A6':6,'AM7':7,'AV8':8,'AS8':9,'A7':10,'A8':11,'M7':12,'A4':13,'M5':14,'AV':15,'A5':16,'AS7':17,'A9':18,'AS9':19,'AV6':20,'AS4':21, 'AM5':22, 'AM8':23, 'AM9':24, 'AS10':25, 'A10':26, 'AV10':27})
    df=df.drop(columns=['Model','Vehicle Class'])
    X=df[['Make','Engine Size(L)','Cylinders','Transmission','Fuel Type','Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)','Fuel Consumption Comb (L/100 km)','Fuel Consumption Comb (mpg)']]
    y = df['CO2 Emissions(g/km)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    table="<table  border='1' style='margin-top:100px;'><tr><th>Total Dataset Records</th><th>80% records as training data</th><th>20% records as test data</th></tr>"
    table+="<tr><td>"+str(len(df))+"</td><td>"+str(len(X_train))+"</td><td>"+str(len(y_test))+"</td></tr>"
    table+="</table>"
    context={"data":table,"data2":"DataSet Preprocessed and Splitted Data"}
    return render(request,'AdminApp/AdminHome.html',context)
global LRacc
global LRModel
def runLinearRegression(request):
    global LRacc
    global LRModel
    LRModel = LinearRegression()  
    LRModel.fit(X_train, y_train)
    LRacc=LRModel.score(X_train, y_train)*100
    context={"data":"Linear Regression Accurary: "+str(LRacc)}
    return render(request,'AdminApp/AdminHome.html',context)

    
global RRacc
global Rmodel
def runRandomRegression(request):
    global RRacc
    global Rmodel
    Rmodel = RandomForestRegressor()  
    Rmodel.fit(X_train, y_train)
    RRacc=Rmodel.score(X_train, y_train)*100
    context={"data":"RandomForest Accurary: "+str(RRacc)}
    return render(request,'AdminApp/AdminHome.html',context)
global knnacc
global knnmodel
def runKNeighborsRegressor(request):
    global knnacc
    global knnmodel
    knnmodel = KNeighborsRegressor()  
    knnmodel.fit(X_train, y_train)
    knnacc=knnmodel.score(X_train, y_train)*100
    context={"data":"KNNeighbors Accurary: "+str(knnacc)}
    return render(request,'AdminApp/AdminHome.html',context)

global XGacc
global XGmodel
def runXGBoost(request):
    global XGacc
    global XGmodel
    XGmodel = xg.XGBRegressor()  
    XGmodel.fit(X_train, y_train)
    XGacc=XGmodel.score(X_train, y_train)*100
    context={"data":"XGBoost Accurary: "+str(XGacc)}
    return render(request,'AdminApp/AdminHome.html',context)
  
def runComparision(request):   
    global LRacc,RRacc,knnacc,XGacc
    bars = ['Linear Regression','RandomforestRegression','KNN Regresion','XGBoost']
    height = [LRacc,RRacc,knnacc,XGacc]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    return render(request,'AdminApp/AdminHome.html')

def predict(request):
    return render(request,'AdminApp/Prediction.html')

def PredAction(request):
    global model
    global Rmodel
    g=request.POST.get('Make')
    a=request.POST.get('engsize')
    hyp=request.POST.get('cylinders')
    heart=request.POST.get('trans')
    em=request.POST.get('fuel')
    wt=request.POST.get('fuelconsumption')
    Rt=request.POST.get('fchwy')
    agl=request.POST.get('fuelconcom')
    bmi=request.POST.get('fccomb')
  
    pred=Rmodel.predict([[g,a,hyp,heart,em,wt,Rt,agl,bmi]])
    r=int(pred)
    if r>130:
        context={'value':r,'data':'Your Vehicle CO2 Emissions(g/km) is more than 130 g/km. \nso better to change vehicle\nTo make Air Quality.'}
        return render(request,'AdminApp/PredictedData.html',context)
    else:
        context={'value':r,'data':'Your Vehicle CO2 Emissions(g/km) is less than 130 g/km. \nIt is good vehicle.'}
        return render(request,'AdminApp/PredictedData.html',context)
        
        
           
    
        
        
    
    



    




    

