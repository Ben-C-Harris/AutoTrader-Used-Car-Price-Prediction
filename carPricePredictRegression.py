# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:29:12 2019

@author: u05ha
"""
import warnings
#import seaborn
#from pylab import *
#import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
#import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

LOAD_DATAFILE = "dataFullDatasetAutoTraderPickle_FULLDATARETRIEVAL20181210_Master.pkl"
RUN_ALL_DATA = True
MAKE = "Volkswagen"
MODEL = "Golf"
PLOT = False
PLOT_R2_THRESHOLD = 50
MODELS = ["Lasso", "Forest", "LinReg"]
MODEL_TYPE = 1 # Lasso, Forest, LinReg
CROSS_FOLD_NUM = 10 # 10
MINIMUM_NUMBER_OF_MODELS_COMPLETE = 5000000 # 5000
MINIMUM_NUMBER_OF_CARS_FOR_ANALYSIS = 50
DISABLE_SKLEARN_WARNING = True

def cutStringCols(df):
    cols_to_remove = []
    
    for col in df.columns:
        try:
            _ = df[col].astype(float)
        except ValueError:
#            print('Couldn\'t covert %s to float' % col)
            cols_to_remove.append(col)
            pass

    # keep only the columns in df that do not contain string
    df = df[[col for col in df.columns if col not in cols_to_remove]]
    return df


def printOutLogger(y_test, pred, reg):
    print("\n — — — — — — — — — — — — — — — — — — — — — — — ")
    print("Mean Absolute Error is: ", mean_absolute_error(y_test, pred))
    print(" — — — — — — — — — — — — — — — — — — — — — — — ")
    print("Mean Squared Error is : ", mean_squared_error(y_test, pred))
    print(" — — — — — — — — — — — — — — — — — — — — — — — ")
    print("The R2 square value is: ", r2_score(y_test, pred) * 100)    
    print(" — — — — — — — — — — — — — — — — — — — — — — — ")
    return r2_score(y_test, pred) * 100
    
    
def plotVarVsPrice(train, test, xTest, pred, variable):
    plt.figure(1)
    variableString = str(variable)
    xTestVariable = xTest[variableString]
    sns.scatterplot(x=variableString, y="Price", data=train, label = "Train")
    sns.scatterplot(x=variableString, y="Price", data=test, label = "Test")
    sns.scatterplot(x=xTestVariable, y=pred, label = "Predicted")
    plt.title(str(variable) + " vs Price")
    plt.xlabel(variable)
    plt.ylabel("Car Price (£)")

    
def plotRegression(yTest, pred):
    plt.figure(2)
    sns.regplot(yTest, pred, color = "teal")
    plt.title("Model prediction accuracy in test data")
    plt.xlabel("Real Price (£)")    
    plt.ylabel("Predicted Price (£)")
    
    
def tenFoldCrossVal(df):
    kf = KFold(n_splits=5)
    kf.get_n_splits(df)    
    
    
def chooseMakeModel(make, model):
    dfMakeModel = pd.read_pickle(LOAD_DATAFILE)
    dfMakeModel["Price"] = dfMakeModel.Price.astype(float)
    
    dfMake = dfMakeModel[dfMakeModel['Make'] == str(make)].copy(deep = True)    
    dfModel = dfMake[dfMake['Model'] == model].copy(deep = True)
    dfModel = dfModel.reset_index()
    return dfModel


def cleanDF(df):
    dfX = df.drop("Price", axis = 1).copy(deep = True)
    dfX = cutStringCols(dfX)
    dfY = df["Price"]   
    return [dfX, dfY]


def buildPredictModel(train, test, xTrain, yTrain, xTest, yTest):
    if DISABLE_SKLEARN_WARNING:
        disableSKLearnWarning()
        
    if MODELS[MODEL_TYPE] == "Lasso":
        print("Constructing Lasso Regression Model...")
        # Creating the Lasso Regression Model
        reg = linear_model.Lasso(alpha=0.1)
        # Fitting and predicting the trained values to the Lassor Regression Model
        reg.fit(xTrain, yTrain)
        pred = reg.predict(xTest)
        r2 = printOutLogger(yTest, pred, reg)
    elif MODELS[MODEL_TYPE] == "Forest":
        print("Constructing Random Forest Regression Model...")
        model = RandomForestRegressor()
        model.fit(xTrain, yTrain)
        pred = model.predict(xTest)    
        r2 = printOutLogger(yTest, pred, model)
    elif MODELS[MODEL_TYPE] == "LinReg":
        print("Constructing Linear Regression Model...")    
        model = LinearRegression()
        model.fit(xTrain, yTrain)
        pred = model.predict(xTest)
        r2 = printOutLogger(yTest, pred, model)    
        
    if PLOT:
#        plotVarVsPrice(train, test, xTest, pred,"Miles")
#        plotRegression(yTest, pred)
        finalPlot(train, test, xTest, yTest, pred, "Miles", r2)
    
    return r2


def printR2Vals(r2Vals):
    print("\n — — — — — — — — — — — — — — — — — — — — — — — ")
    print("Average R2 value over " + str(CROSS_FOLD_NUM) + " KFolds is: " + str(sum(r2Vals) / len(r2Vals)))
    print(" — — — — — — — — — — — — — — — — — — — — — — — \n")


def printFinalR2(r2Vals, carNums):
    r2NullValRemoved = list(filter(lambda a: a != 0, r2Vals))
    print("\n — — — — — — — — — — — — — — — — — — — — — — — ")
    print("Total Average R2 value over " + str(int(len(r2Vals)/CROSS_FOLD_NUM)) + " Specific Models of car containing " + str(carNums) + " seperate vehicles in total with " + str(CROSS_FOLD_NUM) + " KFolds each is: " + str(sum(r2NullValRemoved) / len(r2NullValRemoved)))
    print(" — — — — — — — — — — — — — — — — — — — — — — — \n")


def R2Check(r2Val):
    magR2 = abs(r2Val)
    if( (magR2 >= 0) and (magR2 <= 100) ):
        return magR2
    else:
        return 0


def disableSKLearnWarning():
    warnings.filterwarnings("ignore")


def findEachMakeModel():
    dfMakeModel = pd.read_pickle(LOAD_DATAFILE)
    dfPairs= dfMakeModel.groupby(['Make','Model']).size().reset_index().rename(columns={0:'Count'})
    return dfPairs
    

def executeCarRegression(MAKE, MODEL):
    dfCars = chooseMakeModel(MAKE, MODEL)
    
    r2Vals = []
    kf = KFold(n_splits=CROSS_FOLD_NUM, shuffle = True)
    for trainIndex, testIndex in kf.split(dfCars):
        train  = dfCars[dfCars.index.isin(trainIndex)]
        test  = dfCars[dfCars.index.isin(testIndex)]
        
        # Split train and test into X and Y data
        xTrain, yTrain = cleanDF(train)
        xTest, yTest = cleanDF(test)
        
        # Deliver model and predictions
        r2 = buildPredictModel(train, test, xTrain, yTrain, xTest, yTest)

        r2Checked = R2Check(r2)

        r2Vals.append(r2Checked)
    
    printR2Vals(r2Vals)
    return r2Vals


def howManyCars(completedMakeModel, dfMakeModelPairs):
    totalCarsAnalysed = 0
    for carType in completedMakeModel:
        rule1 = dfMakeModelPairs.Make == carType[0]
        rule2 = dfMakeModelPairs.Model == carType[1]
        num = dfMakeModelPairs[rule1]
        num = num[rule2].Count
        totalCarsAnalysed += int(num)
    return totalCarsAnalysed


def finalPlot(train, test, xTest, yTest, pred, variable, r2):
    if r2 > PLOT_R2_THRESHOLD:
        fig, ((ax1),(ax2)) = plt.subplots(nrows=2, ncols = 1, figsize = (10,10))
    
        variableString = str(variable)
        xTestVariable = xTest[variableString]
        
        sns.scatterplot(x=variableString, y="Price", data=train, label = "Train", ax=ax1)
        sns.scatterplot(x=variableString, y="Price", data=test, label = "Test", ax=ax1)
        sns.scatterplot(x=xTestVariable, y=pred, label = "Predicted", ax=ax1)
        ax1.set_title(str(variable) + " vs Price")
        ax1.set_xlabel(variable)
        ax1.set_ylabel("Car Price (£)")
    
        sns.regplot(yTest, pred, color = "teal", ax=ax2)
        plt.title("Model prediction accuracy in test data")
        plt.xlabel("Real Price (£)")    
        plt.ylabel("Predicted Price (£)")
        
        plt.subplots_adjust(hspace = 0.3)
        
        titleString = MAKE + " " + MODEL + " Regression Analysis ($R^2$=" + '{0:.2f}'.format(r2) + ")"
        fig.suptitle(titleString, fontsize=14)
        
        plotName = 'plots/' + str(MAKE) + "_" + str(MODEL) + "_Plot.png"
        plt.savefig(plotName)
        
        plt.close()



# =============================================================================
# 
# =============================================================================

if RUN_ALL_DATA:
    dfMakeModelPairs = findEachMakeModel()  

    completedMakeModel = []
    incompleteMakeMode = []
    totalR2values = []
    for index, row in dfMakeModelPairs.iterrows():
        if len(completedMakeModel) >= MINIMUM_NUMBER_OF_MODELS_COMPLETE:
            break
        
        MAKE = row.Make
        MODEL = row.Model
        count = row.Count
        
        # Only perform regression on model of car if there is data of at least X many cars
        # Not enough data means poor regression...
        if count < MINIMUM_NUMBER_OF_CARS_FOR_ANALYSIS:
            incompleteMakeMode.append([MAKE, MODEL])
            print("Cannot Complete Regression on: {0}, {1}, as it only has {2} cars.".format(MAKE, MODEL, count))
            continue
        else:
            completedMakeModel.append([MAKE, MODEL])
            print("Completing Regression on: {0}, {1}, using {2} cars.".format(MAKE, MODEL, count))
        
        makeModelR2 = executeCarRegression(MAKE, MODEL)
                
        totalR2values.extend(makeModelR2)
        
        carNums = howManyCars(completedMakeModel, dfMakeModelPairs)
                
        printFinalR2(totalR2values, carNums)       
    
else:
    executeCarRegression(MAKE, MODEL)
    












