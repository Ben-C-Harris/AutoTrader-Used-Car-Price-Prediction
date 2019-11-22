import os
import sys
import warnings
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# Choose dataset to load - assumes common file structure to GitHub
LOAD_DATAFILE = "fullDataSet.pkl"

# Would you like plots, if you do, choose the minimum R2 required for plotting
PLOT = False
PLOT_R2_THRESHOLD = 95

# Choose which model you want to run - Forest is best
MODELS = ["Lasso", "Forest", "LinReg"]
MODEL_TYPE = 1 # Lasso, Forest, LinReg

# K Fold Validation
CROSS_FOLD_NUM = 10

# Choose if you only want to run on the first X number of cars
MINIMUM_NUMBER_OF_MODELS_COMPLETE = 100000
# Choose minimum number of cars in the dataset for a given make and model to create regression
MINIMUM_NUMBER_OF_CARS_FOR_ANALYSIS = 10

#SKLearn gives a future deprecation warning - disable for cleaner logging if required
DISABLE_SKLEARN_WARNING = True

# Output is large, so write to text file if required
PRINT_TO_CONSOLE = False


'''
Class for car price prediction using regression machine learning. Three model 
implimentations avaliable: Lasso, Forest, and LinReg. Forest performs best.
'''
class CarPriceRegressionPredictor():
    
    def __init__(self, LOAD_DATAFILE): # init selfVars
        self.MAKE = "MAKE"
        self.MODEL = "MODEL"       
    
    
    '''
    Regression requires columns of type float. This method removes columns 
    of type !float.
    '''
    def cutStringCols(self, df):
        cols_to_remove = []
        for col in df.columns:
            try:
                _ = df[col].astype(float)
            except ValueError:
                cols_to_remove.append(col)
                pass
    
        # keep only the columns in df that do not contain string
        df = df[[col for col in df.columns if col not in cols_to_remove]]
        return df
    
    
    '''
    Console logger for values after each model fit/predict
    '''
    def printOutLogger(self, y_test, pred, reg):
        print("\n — — — — — — — — — — — — — — — — — — — — — — — ")
        print("Mean Absolute Error is: ", mean_absolute_error(y_test, pred))
        print(" — — — — — — — — — — — — — — — — — — — — — — — ")
        print("Mean Squared Error is : ", mean_squared_error(y_test, pred))
        print(" — — — — — — — — — — — — — — — — — — — — — — — ")
        print("The R2 square value is: ", r2_score(y_test, pred) * 100)    
        print(" — — — — — — — — — — — — — — — — — — — — — — — ")
        return r2_score(y_test, pred) * 100
        
    
    '''
    Load pkl file, and deliver df sliced to only have designated car make and model
    '''
    def chooseMakeModel(self, make, model):
        my_path = os.path.abspath(os.path.dirname(__file__)) # Find your working path
        path = os.path.join(my_path, LOAD_DATAFILE) # join your dir to pkl file name
    
        dfMakeModel = pd.read_pickle(path) # Read pkl data into dataframe
        dfMakeModel["Price"] = dfMakeModel.Price.astype(float)
        
        # slice df from all cars to only those of type Make and Model
        dfMake = dfMakeModel[dfMakeModel['Make'] == str(make)].copy(deep = True)    
        dfModel = dfMake[dfMake['Model'] == model].copy(deep = True)
        dfModel = dfModel.reset_index()
        return dfModel
    
    
    '''
    Split df into clear x and y for regression
    '''
    def cleanDF(self, df):
        dfX = df.drop("Price", axis = 1).copy(deep = True) # Remove price, as price is output of model
        dfX = self.cutStringCols(dfX) # Remove !float columns
        dfY = df["Price"]
        return [dfX, dfY]
    
    
    '''
    Build regression model and predict on test data based upon user/self set variable MODEL_TYPE
    '''
    def buildPredictModel(self, train, test, xTrain, yTrain, xTest, yTest):
        if DISABLE_SKLEARN_WARNING: # Disable sklearn warning
            self.disableSKLearnWarning()
            
        # Choose which model you're building
        if MODELS[MODEL_TYPE] == "Lasso":
            print("Constructing Multiple Lasso Regression Model...")
            # Creating the Lasso Regression Model
            reg = linear_model.Lasso(alpha=0.1)
            # Fitting and predicting the trained values to the Lasso Regression Model
            reg.fit(xTrain, yTrain)
            pred = reg.predict(xTest)
            r2 = self.printOutLogger(yTest, pred, reg)
        elif MODELS[MODEL_TYPE] == "Forest":
            print("Constructing Multiple Random Forest Regression Model...")
            model = RandomForestRegressor()
            model.fit(xTrain, yTrain)
            pred = model.predict(xTest)    
            r2 = self.printOutLogger(yTest, pred, model)
        elif MODELS[MODEL_TYPE] == "LinReg":
            print("Constructing Multiple Linear Regression Model...")    
            model = LinearRegression()
            model.fit(xTrain, yTrain)
            pred = model.predict(xTest)
            r2 = self.printOutLogger(yTest, pred, model)    
            
        if PLOT: # Plot of fit precision for price, and also miles vs price for train, test, predict
            self.finalPlot(train, test, xTest, yTest, pred, "Miles", r2)
        
        return r2
    
    
    '''
    Console log R2 as each fit/predict occurs throughout KFolds cycle
    '''
    def printR2Vals(self, r2Vals):
        print("\n — — — — — — — — — — — — — — — — — — — — — — — ")
        print("Average R2 value over " + str(CROSS_FOLD_NUM) + " KFolds is: " + str(sum(r2Vals) / len(r2Vals)))
        print(" — — — — — — — — — — — — — — — — — — — — — — — \n")
    
    
    '''
    Console log R2 total running average taking into account full KFolds
    '''
    def printFinalR2(self, r2Vals, carNums):
        r2NullValRemoved = list(filter(lambda a: a != 0, r2Vals)) # Remove zero values
        print("\n — — — — — — — — — — — — — — — — — — — — — — — ")
        print("Total Average R2 value over " + str(int(len(r2Vals)/CROSS_FOLD_NUM)) + " Specific Models of car containing " + str(carNums) + " seperate vehicles in total with " + str(CROSS_FOLD_NUM) + " KFolds each is: " + str(sum(r2NullValRemoved) / len(r2NullValRemoved)))
        print(" — — — — — — — — — — — — — — — — — — — — — — — \n")
    
    
    '''
    Check R2 is not out of band
    '''
    def R2Check(self, r2Val):
        magR2 = abs(r2Val)
        if( (magR2 >= 0) and (magR2 <= 100) ):
            return magR2
        else:
            return 0
    
    
    '''
    Disable sklearn future deprecated warning - implimented to keep console log clean.
    If cloning from GitHub - make sure this is not now an issue...
    '''
    def disableSKLearnWarning(self):
        warnings.filterwarnings("ignore")
    
    
    '''
    Find each unique pairing of Make and Model, and how often they occur.
    '''
    def findEachMakeModel(self):
        dfMakeModel = pd.read_pickle(LOAD_DATAFILE)
        dfPairs= dfMakeModel.groupby(['Make','Model']).size().reset_index().rename(columns={0:'Count'})
        return dfPairs
        
    
    '''
    Complete the actual fit/predict regression on specified car makes and models
    '''
    def executeCarRegression(self, MAKE, MODEL):
        dfCars = self.chooseMakeModel(MAKE, MODEL)
        r2Vals = []
        kf = KFold(n_splits=CROSS_FOLD_NUM, shuffle = True) # Perform KFolds test/train
        for trainIndex, testIndex in kf.split(dfCars):
            train  = dfCars[dfCars.index.isin(trainIndex)]
            test  = dfCars[dfCars.index.isin(testIndex)]
            
            # Split train and test into suitable X and Y data
            xTrain, yTrain = self.cleanDF(train)
            xTest, yTest = self.cleanDF(test)
            
            # Deliver model and predictions
            r2 = self.buildPredictModel(train, test, xTrain, yTrain, xTest, yTest)

            # check fit performance through R2    
            r2Checked = self.R2Check(r2)    
            r2Vals.append(r2Checked)
        
        self.printR2Vals(r2Vals)
        return r2Vals
    
    
    '''
    How many cars in total have been analysed
    '''
    def howManyCars(self, completedMakeModel, dfMakeModelPairs):
        totalCarsAnalysed = 0
        for carType in completedMakeModel:
            # Assign df slice conditions
            rule1 = dfMakeModelPairs.Make == carType[0]
            rule2 = dfMakeModelPairs.Model == carType[1]
            
            # Slice df based upon above rules and count their occurrences
            num = dfMakeModelPairs[rule1]
            num = num[rule2].Count
            totalCarsAnalysed += int(num)
            
        return totalCarsAnalysed
    
    
    '''
    Final subplots in figure showing predicted vs real price of car and
    train, test, and predicted values for miles vs price of car
    '''
    def finalPlot(self, train, test, xTest, yTest, pred, variable, r2):
        if r2 > PLOT_R2_THRESHOLD: # Only create plot if R2 is above threshold dictated by user
            fig, ((ax1),(ax2)) = plt.subplots(nrows=2, ncols = 1, figsize = (10,10)) # Set subplots and size
        
            variableString = str(variable)
            xTestVariable = xTest[variableString]
            
            # Create Scatter plot
            sns.scatterplot(x=variableString, y="Price", data=train, label = "Train", ax=ax1)
            sns.scatterplot(x=variableString, y="Price", data=test, label = "Test", ax=ax1)
            sns.scatterplot(x=xTestVariable, y=pred, label = "Predicted", ax=ax1)
            ax1.set_title(str(variable) + " vs Price")
            ax1.set_xlabel(variable)
            ax1.set_ylabel("Car Price (£)")
        
            # Create Regression plot
            sns.regplot(yTest, pred, color = "teal", ax=ax2)
            plt.title("Model prediction accuracy in test data")
            plt.xlabel("Real Price (£)")    
            plt.ylabel("Predicted Price (£)")
            
            # Set titles and format spaces
            plt.subplots_adjust(hspace = 0.3)
            titleString = self.MAKE + " " + self.MODEL + " Regression Analysis ($R^2$=" + '{0:.2f}'.format(r2) + ")"
            fig.suptitle(titleString, fontsize=14)

            # Save to file
            plotName = 'plots/' + str(self.MAKE) + "_" + str(self.MODEL) + "_Plot.png"
            plt.savefig(plotName)
            
            plt.close()
            
            
    '''
    Plot histogram for distribution of R2 values in Regression models
    '''
    def printR2Distribution(self, totalR2values, modelNum):
        plt.hist(totalR2values, normed=True, bins=MINIMUM_NUMBER_OF_CARS_FOR_ANALYSIS)
        plt.xlabel('$R^2$ Value');    
        plt.ylabel('Probability');    
        plt.title("Distribution of $R^2$ over " + str(modelNum) + " models with " + str(CROSS_FOLD_NUM) + " fold validation")    
        plt.savefig("histR2.png")
        plt.close()
   
    
# ============================================================================= 
# =============================================================================


'''
Complete regression ML on avaliable cars within input dataset based upon 
user/self defined variables at top of .py file.

Method will initiate logging and figures as requested by the user incl. KFold 
validation.
'''
def carRegressionPredictor():
    if not PRINT_TO_CONSOLE:
        sys.stdout = open('loggerOut.txt', 'w') # Log to .txt instead of console
    
    predictor = CarPriceRegressionPredictor(LOAD_DATAFILE) # instantiate class object
    
    # Find unique Makes and Models and how often they occur
    dfMakeModelPairs = predictor.findEachMakeModel() 
    
    completedMakeModel = [] # List for each car that has sufficient number of cars in dataSet
    incompleteMakeMode = [] # List for each car that has insufficient number of cars in dataSet
    totalR2values = [] # List of calculated R2 values
    for index, row in dfMakeModelPairs.iterrows(): # Iterate through all makes and models
        if len(completedMakeModel) >= MINIMUM_NUMBER_OF_MODELS_COMPLETE:
            break # Break if you've completed enough
        
        # Define variables
        MAKE = row.Make
        MODEL = row.Model
        count = row.Count
        
        # Only perform regression on model of car if there is data of at least X many cars
        if count < MINIMUM_NUMBER_OF_CARS_FOR_ANALYSIS:
            incompleteMakeMode.append([MAKE, MODEL])
            print("Cannot Complete Regression on: {0}, {1}, as it only has {2} cars.".format(MAKE, MODEL, count))
            continue
        else:
            completedMakeModel.append([MAKE, MODEL])
            print("Completing Regression on: {0}, {1}, using {2} cars.".format(MAKE, MODEL, count))
        
        makeModelR2 = predictor.executeCarRegression(MAKE, MODEL) # Complete fit and predict
                
        totalR2values.extend(makeModelR2) # Append R2 for later use
        
        # Calculate how many cars assesed thus far
        carNums = predictor.howManyCars(completedMakeModel, dfMakeModelPairs)
        
        # Plot for each make and model pair
        predictor.printFinalR2(totalR2values, carNums)       
        
    # Plot histogram of R2 distribution
    predictor.printR2Distribution(totalR2values, len(completedMakeModel))    


if __name__ == "__main__":
    carRegressionPredictor()
    
    
    
    
    
    