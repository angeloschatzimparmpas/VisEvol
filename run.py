from flask import Flask, render_template, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS, cross_origin

import json
import copy
import warnings
import re
import random
import math  
import pandas as pd  
import numpy as np
import multiprocessing

from joblib import Memory

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from imblearn.metrics import geometric_mean_score
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import umap

# this block of code is for the connection between the server, the database, and the client (plus routing)

# access MongoDB 
app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb://localhost:27017/mydb"
mongo = PyMongo(app)

cors = CORS(app, resources={r"/data/*": {"origins": "*"}})

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/Reset', methods=["GET", "POST"])
def reset():

    global Results
    Results = []
    global ResultsCM
    ResultsCM = []

    global DataRawLength
    global DataResultsRaw
    global previousState
    previousState = []

    global filterActionFinal
    filterActionFinal = ''

    global keySpecInternal
    keySpecInternal = 1

    global dataSpacePointsIDs
    dataSpacePointsIDs = []

    global previousStateActive
    previousStateActive = []

    global RANDOM_SEED
    RANDOM_SEED = 42

    global KNNModelsCount
    global LRModelsCount

    global factors
    factors = [1,1,1,1,0,0,0,0]

    global crossValidation
    crossValidation = 5

    global randomSearchVar
    randomSearchVar = 100

    global keyData
    keyData = 0

    KNNModelsCount = 0
    LRModelsCount = KNNModelsCount+randomSearchVar
    MLPModelsCount = LRModelsCount+randomSearchVar
    RFModelsCount = MLPModelsCount+randomSearchVar
    GradBModelsCount = RFModelsCount+randomSearchVar

    global XData
    XData = []
    global yData
    yData = []

    global EnsembleActive
    EnsembleActive = []

    global addKNN
    addKNN = 0

    global addLR
    addLR = addKNN+randomSearchVar

    global addMLP
    addMLP = addLR+randomSearchVar

    global addRF
    addRF = addMLP+randomSearchVar

    global addGradB
    addGradB = addRF+randomSearchVar

    global countAllModels
    countAllModels = 0

    global XDataStored
    XDataStored = []
    global yDataStored
    yDataStored = []
    
    global detailsParams
    detailsParams = []

    global algorithmList
    algorithmList = []

    global ClassifierIDsList
    ClassifierIDsList = ''

    # Initializing models

    global resultsList
    resultsList = []

    global RetrieveModelsList
    RetrieveModelsList = []

    global allParametersPerformancePerModel
    allParametersPerformancePerModel = []

    global allParametersPerfCrossMutr
    allParametersPerfCrossMutr = []

    global HistoryPreservation
    HistoryPreservation = []

    global all_classifiers
    all_classifiers = []
    
    # models
    global KNNModels
    KNNModels = []
    global RFModels
    RFModels = []

    global scoring
    scoring = {'accuracy': 'accuracy', 'precision_micro': 'precision_micro', 'precision_macro': 'precision_macro', 'precision_weighted': 'precision_weighted', 'recall_micro': 'recall_micro', 'recall_macro': 'recall_macro', 'recall_weighted': 'recall_weighted', 'roc_auc_ovo_weighted': 'roc_auc_ovo_weighted'}

    global loopFeatures
    loopFeatures = 2

    global results
    results = []

    global resultsMetrics
    resultsMetrics = []

    global parametersSelData
    parametersSelData = []

    global target_names
    target_names = []

    global target_namesLoc
    target_namesLoc = []
    return 'The reset was done!'

# retrieve data from client and select the correct data set
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRequest', methods=["GET", "POST"])
def retrieveFileName():
    global DataRawLength
    global DataResultsRaw
    global DataResultsRawTest
    global DataRawLengthTest

    fileName = request.get_data().decode('utf8').replace("'", '"')
    data = json.loads(fileName)  

    global keySpecInternal
    keySpecInternal = 1

    global filterActionFinal
    filterActionFinal = ''

    global dataSpacePointsIDs
    dataSpacePointsIDs = []

    global RANDOM_SEED
    RANDOM_SEED = 42

    global keyData
    keyData = 0

    global KNNModelsCount
    global LRModelsCount
    global MLPModelsCount
    global RFModelsCount
    global GradBModelsCount

    global factors
    factors = data['Factors']

    global crossValidation
    crossValidation = int(data['CrossValidation'])

    global randomSearchVar
    randomSearchVar = int(data['RandomSearch'])

    KNNModelsCount = 0
    LRModelsCount = KNNModelsCount+randomSearchVar
    MLPModelsCount = LRModelsCount+randomSearchVar
    RFModelsCount = MLPModelsCount+randomSearchVar
    GradBModelsCount = RFModelsCount+randomSearchVar

    global XData
    XData = []

    global previousState
    previousState = []

    global previousStateActive
    previousStateActive = []

    global yData
    yData = []

    global XDataStored
    XDataStored = []

    global yDataStored
    yDataStored = []

    global filterDataFinal
    filterDataFinal = 'mean'

    global ClassifierIDsList
    ClassifierIDsList = ''

    global algorithmList
    algorithmList = []

    global detailsParams
    detailsParams = []
    
    global EnsembleActive
    EnsembleActive = []

    global addKNN
    addKNN = 0

    global addLR
    addLR = addKNN+randomSearchVar

    global addMLP
    addMLP = addLR+randomSearchVar

    global addRF
    addRF = addMLP+randomSearchVar

    global addGradB
    addGradB = addRF+randomSearchVar

    # Initializing models

    global RetrieveModelsList
    RetrieveModelsList = []

    global resultsList
    resultsList = []

    global allParametersPerformancePerModel
    allParametersPerformancePerModel = []

    global allParametersPerfCrossMutr
    allParametersPerfCrossMutr = []

    global HistoryPreservation
    HistoryPreservation = []

    global all_classifiers
    all_classifiers = []

    global scoring
    scoring = {'accuracy': 'accuracy', 'precision_weighted': 'precision_weighted', 'recall_weighted': 'recall_weighted', 'f1_weighted': 'f1_weighted', 'roc_auc_ovo_weighted': 'roc_auc_ovo_weighted'}

    global loopFeatures
    loopFeatures = 2

    # models
    global KNNModels
    global MLPModels
    global LRModels
    global RFModels
    global GradBModels

    KNNModels = []
    MLPModels = []
    LRModels = []
    RFModels = []
    GradBModels = []

    global results
    results = []

    global resultsMetrics
    resultsMetrics = []

    global parametersSelData
    parametersSelData = []

    global StanceTest
    StanceTest = False

    global target_names
    
    target_names = []

    global target_namesLoc
    
    target_namesLoc = []

    DataRawLength = -1
    DataRawLengthTest = -1

    if data['fileName'] == 'HeartC':
        CollectionDB = mongo.db.HeartC.find()
    elif data['fileName'] == 'StanceC':
        StanceTest = True
        CollectionDB = mongo.db.StanceC.find()
        CollectionDBTest = mongo.db.StanceCTest.find()
    elif data['fileName'] == 'DiabetesC':
        CollectionDB = mongo.db.DiabetesC.find()
    else:
        CollectionDB = mongo.db.IrisC.find()
    DataResultsRaw = []
    for index, item in enumerate(CollectionDB):
        item['_id'] = str(item['_id'])
        item['InstanceID'] = index
        DataResultsRaw.append(item)
    DataRawLength = len(DataResultsRaw)

    DataResultsRawTest = []
    if (StanceTest):
        for index, item in enumerate(CollectionDBTest):
            item['_id'] = str(item['_id'])
            item['InstanceID'] = index
            DataResultsRawTest.append(item)
        DataRawLengthTest = len(DataResultsRawTest)

    dataSetSelection()
    return 'Everything is okay'

# Retrieve data set from client
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/SendtoSeverDataSet', methods=["GET", "POST"])
def sendToServerData():

    uploadedData = request.get_data().decode('utf8').replace("'", '"')
    uploadedDataParsed = json.loads(uploadedData)
    DataResultsRaw = uploadedDataParsed['uploadedData']

    DataResults = copy.deepcopy(DataResultsRaw)

    for dictionary in DataResultsRaw:
        for key in dictionary.keys():
            if (key.find('*') != -1):
                target = key
                continue
        continue
    DataResultsRaw.sort(key=lambda x: x[target], reverse=True)
    DataResults.sort(key=lambda x: x[target], reverse=True)

    for dictionary in DataResults:
        del dictionary[target]

    global AllTargets
    global target_names
    global target_namesLoc
    AllTargets = [o[target] for o in DataResultsRaw]
    AllTargetsFloatValues = []

    previous = None
    Class = 0
    for i, value in enumerate(AllTargets):
        if (i == 0):
            previous = value
            target_names.append(value)
        if (value == previous):
            AllTargetsFloatValues.append(Class)
        else:
            Class = Class + 1
            target_names.append(value)
            AllTargetsFloatValues.append(Class)
            previous = value

    ArrayDataResults = pd.DataFrame.from_dict(DataResults)

    global XData, yData, RANDOM_SEED
    XData, yData = ArrayDataResults, AllTargetsFloatValues

    global XDataStored, yDataStored
    XDataStored = XData.copy()
    yDataStored = yData.copy()
    
    return 'Processed uploaded data set'

def dataSetSelection():
    global XDataTest, yDataTest
    XDataTest = pd.DataFrame()
    global StanceTest
    global AllTargets
    global target_names
    target_namesLoc = []
    if (StanceTest):
        DataResultsTest = copy.deepcopy(DataResultsRawTest)

        for dictionary in DataResultsRawTest:
            for key in dictionary.keys():
                if (key.find('*') != -1):
                    target = key
                    continue
            continue

        DataResultsRawTest.sort(key=lambda x: x[target], reverse=True)
        DataResultsTest.sort(key=lambda x: x[target], reverse=True)

        for dictionary in DataResultsTest:
            del dictionary['_id']
            del dictionary['InstanceID']
            del dictionary[target]

        AllTargetsTest = [o[target] for o in DataResultsRawTest]
        AllTargetsFloatValuesTest = []

        previous = None
        Class = 0
        for i, value in enumerate(AllTargetsTest):
            if (i == 0):
                previous = value
                target_namesLoc.append(value)
            if (value == previous):
                AllTargetsFloatValuesTest.append(Class)
            else:
                Class = Class + 1
                target_namesLoc.append(value)
                AllTargetsFloatValuesTest.append(Class)
                previous = value

        ArrayDataResultsTest = pd.DataFrame.from_dict(DataResultsTest)

        XDataTest, yDataTest = ArrayDataResultsTest, AllTargetsFloatValuesTest

    DataResults = copy.deepcopy(DataResultsRaw)

    for dictionary in DataResultsRaw:
        for key in dictionary.keys():
            if (key.find('*') != -1):
                target = key
                continue
        continue

    DataResultsRaw.sort(key=lambda x: x[target], reverse=True)
    DataResults.sort(key=lambda x: x[target], reverse=True)

    for dictionary in DataResults:
        del dictionary['_id']
        del dictionary['InstanceID']
        del dictionary[target]

    AllTargets = [o[target] for o in DataResultsRaw]
    AllTargetsFloatValues = []

    previous = None
    Class = 0
    for i, value in enumerate(AllTargets):
        if (i == 0):
            previous = value
            target_names.append(value)
        if (value == previous):
            AllTargetsFloatValues.append(Class)
        else:
            Class = Class + 1
            target_names.append(value)
            AllTargetsFloatValues.append(Class)
            previous = value

    ArrayDataResults = pd.DataFrame.from_dict(DataResults)

    global XData, yData, RANDOM_SEED
    XData, yData = ArrayDataResults, AllTargetsFloatValues

    global XDataStored, yDataStored
    XDataStored = XData.copy()
    yDataStored = yData.copy()

    warnings.simplefilter('ignore')
    return 'Everything is okay'

# Retrieve data from client 
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/factors', methods=["GET", "POST"])
def RetrieveFactors():
    global factors
    global allParametersPerformancePerModel
    Factors = request.get_data().decode('utf8').replace("'", '"')
    FactorsInt = json.loads(Factors)
    factors = FactorsInt['Factors']

    return 'Everything Okay'

# Initialize every model for each algorithm
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRequestSelParameters', methods=["GET", "POST"])
def retrieveModel():

    # get the models from the frontend
    RetrievedModel = request.get_data().decode('utf8').replace("'", '"')
    RetrievedModel = json.loads(RetrievedModel)

    global algorithms
    algorithms = RetrievedModel['Algorithms']

    global XData
    global yData
    global LRModelsCount
    global countAllModels

    # loop through the algorithms
    global allParametersPerformancePerModel
    global HistoryPreservation

    for eachAlgor in algorithms:
        print(eachAlgor)
        if (eachAlgor) == 'KNN':
            clf = KNeighborsClassifier()
            params = {'n_neighbors': list(range(1, 100)), 'metric': ['chebyshev', 'manhattan', 'euclidean', 'minkowski'], 'algorithm': ['brute', 'kd_tree', 'ball_tree'], 'weights': ['uniform', 'distance']}
            AlgorithmsIDsEnd = countAllModels
        elif (eachAlgor) == 'LR':
            clf = LogisticRegression(random_state=RANDOM_SEED)
            params = {'C': list(np.arange(1,100,1)), 'max_iter': list(np.arange(50,500,50)), 'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'], 'penalty': ['l2', 'none']}
            countAllModels = countAllModels + randomSearchVar
            AlgorithmsIDsEnd = countAllModels
        elif (eachAlgor) == 'MLP':
            start = 60
            stop = 120
            step = 1
            random.seed(RANDOM_SEED)
            ranges = [(n, random.randint(1,3)) for n in range(start, stop, step)]
            clf = MLPClassifier(random_state=RANDOM_SEED)
            params = {'hidden_layer_sizes': ranges,'alpha': list(np.arange(0.00001,0.001,0.0002)), 'tol': list(np.arange(0.00001,0.001,0.0004)), 'max_iter': list(np.arange(100,200,100)), 'activation': ['relu', 'identity', 'logistic', 'tanh'], 'solver' : ['adam', 'sgd']}
            countAllModels = countAllModels + randomSearchVar
            AlgorithmsIDsEnd = countAllModels
        elif (eachAlgor) == 'RF':
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': list(range(20, 100)), 'criterion': ['gini', 'entropy']}
            countAllModels = countAllModels + randomSearchVar
            AlgorithmsIDsEnd = countAllModels
        else: 
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': list(range(20, 100)), 'learning_rate': list(np.arange(0.01,0.23,0.11)), 'criterion': ['friedman_mse', 'mse', 'mae']}
            countAllModels = countAllModels + randomSearchVar
            AlgorithmsIDsEnd = countAllModels
        allParametersPerformancePerModel = randomSearch(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd)
    HistoryPreservation = allParametersPerformancePerModel.copy()
    # call the function that sends the results to the frontend

    return 'Everything Okay'

location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def randomSearch(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd):

    search = RandomizedSearchCV(    
        estimator=clf, param_distributions=params, n_iter=100,
        cv=crossValidation, refit='accuracy', scoring=scoring,
        verbose=0, n_jobs=-1)

    # fit and extract the probabilities
    search.fit(XData, yData)

    # process the results
    cv_results = []
    cv_results.append(search.cv_results_)
    df_cv_results = pd.DataFrame.from_dict(cv_results)

    # number of models stored
    number_of_models = len(df_cv_results.iloc[0][0])

    # initialize results per row
    df_cv_results_per_row = []

    # loop through number of models
    modelsIDs = []
    for i in range(number_of_models):
        number = AlgorithmsIDsEnd+i
        modelsIDs.append(eachAlgor+str(number))
         # initialize results per item
        df_cv_results_per_item = []
        for column in df_cv_results.iloc[0]:
            df_cv_results_per_item.append(column[i])
        df_cv_results_per_row.append(df_cv_results_per_item)

    # store the results into a pandas dataframe
    df_cv_results_classifiers = pd.DataFrame(data = df_cv_results_per_row, columns= df_cv_results.columns)

    # copy and filter in order to get only the metrics
    metrics = df_cv_results_classifiers.copy()
    metrics = metrics.filter(['mean_test_accuracy','mean_test_precision_weighted','mean_test_recall_weighted','mean_test_f1_weighted','mean_test_roc_auc_ovo_weighted']) 

    # concat parameters and performance
    parametersPerformancePerModel = pd.DataFrame(df_cv_results_classifiers['params'])
    parametersLocal = parametersPerformancePerModel['params'].copy()

    Models = []
    for index, items in enumerate(parametersLocal):
        Models.append(index)
    parametersLocalNew = [ parametersLocal[your_key] for your_key in Models ]

    perModelProb = []
    
    resultsWeighted = []
    resultsCorrCoef = []
    resultsLogLoss = []
    resultsLogLossFinal = []

    # influence calculation for all the instances
    inputs = range(len(XData))
    num_cores = multiprocessing.cpu_count()
    
    for eachModelParameters in parametersLocalNew:
        clf.set_params(**eachModelParameters)
        clf.fit(XData, yData) 
        yPredict = clf.predict(XData)
        yPredict = np.nan_to_num(yPredict)
        yPredictProb = cross_val_predict(clf, XData, yData, cv=crossValidation, method='predict_proba')
        yPredictProb = np.nan_to_num(yPredictProb)
        perModelProb.append(yPredictProb.tolist())

        resultsWeighted.append(geometric_mean_score(yData, yPredict, average='weighted'))
        resultsCorrCoef.append(matthews_corrcoef(yData, yPredict))
        resultsLogLoss.append(log_loss(yData, yPredictProb, normalize=True))

    maxLog = max(resultsLogLoss)
    minLog = min(resultsLogLoss)
    for each in resultsLogLoss:
        resultsLogLossFinal.append((each-minLog)/(maxLog-minLog))

    metrics.insert(5,'geometric_mean_score_weighted',resultsWeighted)
    metrics.insert(6,'matthews_corrcoef',resultsCorrCoef)
    metrics.insert(7,'log_loss',resultsLogLossFinal)

    perModelProbPandas = pd.DataFrame(perModelProb)

    results.append(modelsIDs)
    results.append(parametersPerformancePerModel)
    results.append(metrics)
    results.append(perModelProbPandas)

    return results

def PreprocessingIDs():
    dicKNN = allParametersPerformancePerModel[0]
    dicLR = allParametersPerformancePerModel[4]
    dicMLP = allParametersPerformancePerModel[8]
    dicRF = allParametersPerformancePerModel[12]
    dicGradB = allParametersPerformancePerModel[16]

    df_concatIDs = dicKNN + dicLR + dicMLP + dicRF + dicGradB

    return df_concatIDs

def PreprocessingMetrics():
    dicKNN = allParametersPerformancePerModel[2]
    dicLR = allParametersPerformancePerModel[6]
    dicMLP = allParametersPerformancePerModel[10]
    dicRF = allParametersPerformancePerModel[14]
    dicGradB = allParametersPerformancePerModel[18]

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfLR = pd.DataFrame.from_dict(dicLR)
    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfRF = pd.DataFrame.from_dict(dicRF)
    dfGradB = pd.DataFrame.from_dict(dicGradB)

    df_concatMetrics = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])
    df_concatMetrics = df_concatMetrics.reset_index(drop=True)
    return df_concatMetrics

def PreprocessingPred():
    dicKNN = allParametersPerformancePerModel[3]
    dicLR = allParametersPerformancePerModel[7]
    dicMLP = allParametersPerformancePerModel[11]
    dicRF = allParametersPerformancePerModel[15]
    dicGradB = allParametersPerformancePerModel[19]

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfLR = pd.DataFrame.from_dict(dicLR)
    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfRF = pd.DataFrame.from_dict(dicRF)
    dfGradB = pd.DataFrame.from_dict(dicGradB)

    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])
    df_concatProbs.reset_index(drop=True, inplace=True)

    predictionsKNN = []
    for column, content in dfKNN.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsKNN.append(el)

    predictionsLR = []
    for column, content in dfLR.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsLR.append(el)

    predictionsMLP = []
    for column, content in dfMLP.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsMLP.append(el)

    predictionsRF = []
    for column, content in dfRF.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsRF.append(el)

    predictionsGradB = []
    for column, content in dfGradB.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsGradB.append(el)
    
    predictions = []
    for column, content in df_concatProbs.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictions.append(el)

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

def PreprocessingPredEnsemble():

    global EnsembleActive

    numberIDKNN = []
    numberIDLR = []
    numberIDMLP = []
    numberIDRF = []
    numberIDGradB = []

    for el in EnsembleActive:
        match = re.match(r"([a-z]+)([0-9]+)", el, re.I)
        if match:
            items = match.groups()
            if (items[0] == 'KNN'):
                numberIDKNN.append(int(items[1]))
            elif (items[0] == 'LR'):
                numberIDLR.append(int(items[1]))
            elif (items[0] == 'MLP'):
                numberIDMLP.append(int(items[1]))
            elif (items[0] == 'RF'):
                numberIDRF.append(int(items[1]))
            else:
                numberIDGradB.append(int(items[1]))

    dicKNN = allParametersPerformancePerModel[3]
    dicLR = allParametersPerformancePerModel[7]
    dicMLP = allParametersPerformancePerModel[11]
    dicRF = allParametersPerformancePerModel[15]
    dicGradB = allParametersPerformancePerModel[19]

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfLR = pd.DataFrame.from_dict(dicLR)
    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfRF = pd.DataFrame.from_dict(dicRF)
    dfGradB = pd.DataFrame.from_dict(dicGradB)

    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])
    df_concatProbs = df_concatProbs.reset_index(drop=True)

    dfKNN = df_concatProbs.loc[numberIDKNN]
    dfLR = df_concatProbs.loc[numberIDLR]
    dfMLP = df_concatProbs.loc[numberIDMLP]
    dfRF = df_concatProbs.loc[numberIDRF]
    dfGradB = df_concatProbs.loc[numberIDGradB]

    df_concatProbs = pd.DataFrame()
    df_concatProbs = df_concatProbs.iloc[0:0]
    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])

    predictionsKNN = []
    for column, content in dfKNN.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsKNN.append(el)

    predictionsLR = []
    for column, content in dfLR.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsLR.append(el)

    predictionsMLP = []
    for column, content in dfMLP.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsMLP.append(el)

    predictionsRF = []
    for column, content in dfRF.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsRF.append(el)

    predictionsGradB = []
    for column, content in dfGradB.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsGradB.append(el)

    predictions = []
    for column, content in df_concatProbs.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictions.append(el)

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

def PreprocessingParam():
    dicKNN = allParametersPerformancePerModel[1]
    dicLR = allParametersPerformancePerModel[5]
    dicMLP = allParametersPerformancePerModel[9]
    dicRF = allParametersPerformancePerModel[13]
    dicGradB = allParametersPerformancePerModel[17]

    dicKNN = dicKNN['params']
    dicLR = dicLR['params']
    dicMLP = dicMLP['params']
    dicRF = dicRF['params']
    dicGradB = dicGradB['params']
    
    dicKNN = {int(k):v for k,v in dicKNN.items()}
    dicLR = {int(k):v for k,v in dicLR.items()}
    dicMLP = {int(k):v for k,v in dicMLP.items()}
    dicRF = {int(k):v for k,v in dicRF.items()}
    dicGradB = {int(k):v for k,v in dicGradB.items()}

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfLR = pd.DataFrame.from_dict(dicLR)
    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfRF = pd.DataFrame.from_dict(dicRF)
    dfGradB = pd.DataFrame.from_dict(dicGradB)

    dfKNN = dfKNN.T
    dfLR = dfLR.T
    dfMLP = dfMLP.T
    dfRF = dfRF.T
    dfGradB = dfGradB.T

    df_params = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])
    df_params = df_params.reset_index(drop=True)
    return df_params

def PreprocessingParamSep():
    dicKNN = allParametersPerformancePerModel[1]
    dicLR = allParametersPerformancePerModel[5]
    dicMLP = allParametersPerformancePerModel[9]
    dicRF = allParametersPerformancePerModel[13]
    dicGradB = allParametersPerformancePerModel[17]

    dicKNN = dicKNN['params']
    dicLR = dicLR['params']
    dicMLP = dicMLP['params']
    dicRF = dicRF['params']
    dicGradB = dicGradB['params']
    
    dicKNN = {int(k):v for k,v in dicKNN.items()}
    dicLR = {int(k):v for k,v in dicLR.items()}
    dicMLP = {int(k):v for k,v in dicMLP.items()}
    dicRF = {int(k):v for k,v in dicRF.items()}
    dicGradB = {int(k):v for k,v in dicGradB.items()}

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfLR = pd.DataFrame.from_dict(dicLR)
    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfRF = pd.DataFrame.from_dict(dicRF)
    dfGradB = pd.DataFrame.from_dict(dicGradB)

    dfKNN = dfKNN.T
    dfLR = dfLR.T
    dfMLP = dfMLP.T
    dfRF = dfRF.T
    dfGradB = dfGradB.T

    return [dfKNN, dfLR, dfMLP, dfRF, dfGradB]

# remove that maybe!
def preProcsumPerMetric(factors):
    sumPerClassifier = []
    loopThroughMetrics = PreprocessingMetrics()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    loopThroughMetrics.loc[:, 'log_loss'] = 1 - loopThroughMetrics.loc[:, 'log_loss']
    for row in loopThroughMetrics.iterrows():
        rowSum = 0
        name, values = row
        for loop, elements in enumerate(values):
            rowSum = elements*factors[loop] + rowSum
        if sum(factors) is 0:
            sumPerClassifier = 0
        else:
            sumPerClassifier.append(rowSum/sum(factors) * 100)
    return sumPerClassifier

def preProcMetricsAllAndSel():
    loopThroughMetrics = PreprocessingMetrics()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    global factors
    metricsPerModelColl = []
    metricsPerModelColl.append(loopThroughMetrics['mean_test_accuracy'])
    metricsPerModelColl.append(loopThroughMetrics['geometric_mean_score_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_precision_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_recall_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_f1_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['matthews_corrcoef'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_roc_auc_ovo_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['log_loss'])

    f=lambda a: (abs(a)+a)/2
    for index, metric in enumerate(metricsPerModelColl):
        if (index == 5):
            metricsPerModelColl[index] = ((f(metric))*factors[index]) * 100
        elif (index == 7):
            metricsPerModelColl[index] = ((1 - metric)*factors[index] ) * 100
        else:  
            metricsPerModelColl[index] = (metric*factors[index]) * 100
        metricsPerModelColl[index] = metricsPerModelColl[index].to_json()
    return metricsPerModelColl

def FunMDS (data):
    mds = MDS(n_components=2, random_state=RANDOM_SEED)
    XTransformed = mds.fit_transform(data).T
    XTransformed = XTransformed.tolist()
    return XTransformed

def FunTsne (data):
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED).fit_transform(data)
    tsne.shape
    return tsne

def FunUMAP (data):
    trans = umap.UMAP(n_neighbors=15, random_state=RANDOM_SEED).fit(data)
    Xpos = trans.embedding_[:, 0].tolist()
    Ypos = trans.embedding_[:, 1].tolist()
    return [Xpos,Ypos]

# Sending the overview classifiers' results to be visualized as a scatterplot
@app.route('/data/PlotClassifiers', methods=["GET", "POST"])
def SendToPlot():
    while (len(DataResultsRaw) != DataRawLength):
        pass
    InitializeEnsemble()
    response = {    
        'OverviewResults': Results
    }
    return jsonify(response)

def InitializeEnsemble(): 
    XModels = PreprocessingMetrics()
    global ModelSpaceMDS
    global ModelSpaceTSNE
    global allParametersPerformancePerModel
    global EnsembleActive

    XModels = XModels.fillna(0)

    ModelSpaceMDS = FunMDS(XModels)
    ModelSpaceTSNE = FunTsne(XModels)
    ModelSpaceTSNE = ModelSpaceTSNE.tolist()
    ModelSpaceUMAP = FunUMAP(XModels)

    if (len(EnsembleActive) == 0):
        PredictionProbSel = PreprocessingPred()
    else:
        PredictionProbSel = PreprocessingPredEnsemble()

    returnResults(ModelSpaceMDS,ModelSpaceTSNE,ModelSpaceUMAP,PredictionProbSel)

def returnResults(ModelSpaceMDS,ModelSpaceTSNE,ModelSpaceUMAP,PredictionProbSel):

    global Results
    global AllTargets
    Results = []

    parametersGen = PreprocessingParam()
    metricsPerModel = preProcMetricsAllAndSel()
    sumPerClassifier = preProcsumPerMetric(factors)
    ModelsIDs = PreprocessingIDs()
    

    parametersGenPD = parametersGen.to_json(orient='records')
    XDataJSONEntireSet = XData.to_json(orient='records')
    XDataColumns = XData.columns.tolist()

    Results.append(json.dumps(ModelsIDs))
    Results.append(json.dumps(sumPerClassifier))
    Results.append(json.dumps(parametersGenPD))
    Results.append(json.dumps(metricsPerModel))
    Results.append(json.dumps(XDataJSONEntireSet))
    Results.append(json.dumps(XDataColumns))
    Results.append(json.dumps(yData))
    Results.append(json.dumps(target_names))
    Results.append(json.dumps(AllTargets))
    Results.append(json.dumps(ModelSpaceMDS))
    Results.append(json.dumps(ModelSpaceTSNE))
    Results.append(json.dumps(ModelSpaceUMAP))
    Results.append(json.dumps(PredictionProbSel))

    return Results

# Initialize crossover and mutation processes
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/CrossoverMutation', methods=["GET", "POST"])
def CrossoverMutateFun():

    # get the models from the frontend
    RemainingIds = request.get_data().decode('utf8').replace("'", '"')
    RemainingIds = json.loads(RemainingIds)

    RemainingIds = RemainingIds['RemainingPoints']
    
    global EnsembleActive

    EnsembleActive = request.get_data().decode('utf8').replace("'", '"')
    EnsembleActive = json.loads(EnsembleActive)

    EnsembleActive = EnsembleActive['StoreEnsemble']
    random.seed(RANDOM_SEED)
    
    global XData
    global yData
    global addKNN
    global addLR
    global addMLP
    global addRF
    global addGradB
    global countAllModels

    # loop through the algorithms
    global allParametersPerfCrossMutr
    global HistoryPreservation
    global allParametersPerformancePerModel

    KNNIDs = list(filter(lambda k: 'KNN' in k, RemainingIds))
    LRIDs = list(filter(lambda k: 'LR' in k, RemainingIds))
    MLPIDs = list(filter(lambda k: 'MLP' in k, RemainingIds))
    RFIDs = list(filter(lambda k: 'RF' in k, RemainingIds))
    GradBIDs = list(filter(lambda k: 'GradB' in k, RemainingIds))

    countKNN = 0
    countLR = 0
    countMLP = 0
    countRF = 0
    countGradB = 0
    setMaxLoopValue = 5
    paramAllAlgs = PreprocessingParam()

    KNNIntIndex = []
    LRIntIndex = []
    MLPIntIndex = []
    RFIntIndex = []
    GradBIntIndex = []
    
    localCrossMutr = []
    allParametersPerfCrossMutrKNNC = []
    while countKNN < setMaxLoopValue:
        for dr in KNNIDs:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
        KNNPickPair = random.sample(KNNIntIndex,2)
        pairDF = paramAllAlgs.iloc[KNNPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['algorithm'] == crossoverDF['algorithm'].iloc[0]) & (paramAllAlgs['metric'] == crossoverDF['metric'].iloc[0]) & (paramAllAlgs['n_neighbors'] == crossoverDF['n_neighbors'].iloc[0]) & (paramAllAlgs['weights'] == crossoverDF['weights'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = KNeighborsClassifier()
            params = {'n_neighbors': [crossoverDF['n_neighbors'].iloc[0]], 'metric': [crossoverDF['metric'].iloc[0]], 'algorithm': [crossoverDF['algorithm'].iloc[0]], 'weights': [crossoverDF['weights'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countKNN
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNN', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrKNNC.append(localCrossMutr[0])
    allParametersPerfCrossMutrKNNC.append(localCrossMutr[1])
    allParametersPerfCrossMutrKNNC.append(localCrossMutr[2])
    allParametersPerfCrossMutrKNNC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrKNNC

    countKNN = 0
    KNNIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrKNNM = []

    while countKNN < setMaxLoopValue:
        for dr in KNNIDs:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
        KNNPickPair = random.sample(KNNIntIndex,1)

        pairDF = paramAllAlgs.iloc[KNNPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            if (column == 'n_neighbors'):
                randomNumber = random.randint(101, math.floor(((len(yData)/crossValidation)*(crossValidation-1)))-1)
                listData.append(randomNumber)
                crossoverDF[column] = listData
            else:
                valuePerColumn = pairDF[column].iloc[0]
                listData.append(valuePerColumn)
                crossoverDF[column] = listData
        if (((paramAllAlgs['algorithm'] == crossoverDF['algorithm'].iloc[0]) & (paramAllAlgs['metric'] == crossoverDF['metric'].iloc[0]) & (paramAllAlgs['n_neighbors'] == crossoverDF['n_neighbors'].iloc[0]) & (paramAllAlgs['weights'] == crossoverDF['weights'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = KNeighborsClassifier()
            params = {'n_neighbors': [crossoverDF['n_neighbors'].iloc[0]], 'metric': [crossoverDF['metric'].iloc[0]], 'algorithm': [crossoverDF['algorithm'].iloc[0]], 'weights': [crossoverDF['weights'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countKNN
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNN', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrKNNM.append(localCrossMutr[0])
    allParametersPerfCrossMutrKNNM.append(localCrossMutr[1])
    allParametersPerfCrossMutrKNNM.append(localCrossMutr[2])
    allParametersPerfCrossMutrKNNM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrKNNM

    localCrossMutr.clear()
    allParametersPerfCrossMutrLRC = []

    while countLR < setMaxLoopValue:
        for dr in LRIDs:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
        LRPickPair = random.sample(LRIntIndex,2)

        pairDF = paramAllAlgs.iloc[LRPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['C'] == crossoverDF['C'].iloc[0]) & (paramAllAlgs['max_iter'] == crossoverDF['max_iter'].iloc[0]) & (paramAllAlgs['solver'] == crossoverDF['solver'].iloc[0]) & (paramAllAlgs['penalty'] == crossoverDF['penalty'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = LogisticRegression(random_state=RANDOM_SEED)
            params = {'C': [crossoverDF['C'].iloc[0]], 'max_iter': [crossoverDF['max_iter'].iloc[0]], 'solver': [crossoverDF['solver'].iloc[0]], 'penalty': [crossoverDF['penalty'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countLR
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LR', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrLRC.append(localCrossMutr[0])
    allParametersPerfCrossMutrLRC.append(localCrossMutr[1])
    allParametersPerfCrossMutrLRC.append(localCrossMutr[2])
    allParametersPerfCrossMutrLRC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrLRC

    countLR = 0
    LRIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrLRM = []

    while countLR < setMaxLoopValue:
        for dr in LRIDs:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
        LRPickPair = random.sample(LRIntIndex,1)

        pairDF = paramAllAlgs.iloc[LRPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            if (column == 'C'):
                randomNumber = random.randint(101, 1000)
                listData.append(randomNumber)
                crossoverDF[column] = listData
            else:
                valuePerColumn = pairDF[column].iloc[0]
                listData.append(valuePerColumn)
                crossoverDF[column] = listData
        if (((paramAllAlgs['C'] == crossoverDF['C'].iloc[0]) & (paramAllAlgs['max_iter'] == crossoverDF['max_iter'].iloc[0]) & (paramAllAlgs['solver'] == crossoverDF['solver'].iloc[0]) & (paramAllAlgs['penalty'] == crossoverDF['penalty'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = LogisticRegression(random_state=RANDOM_SEED)
            params = {'C': [crossoverDF['C'].iloc[0]], 'max_iter': [crossoverDF['max_iter'].iloc[0]], 'solver': [crossoverDF['solver'].iloc[0]], 'penalty': [crossoverDF['penalty'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countLR
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LR', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrLRM.append(localCrossMutr[0])
    allParametersPerfCrossMutrLRM.append(localCrossMutr[1])
    allParametersPerfCrossMutrLRM.append(localCrossMutr[2])
    allParametersPerfCrossMutrLRM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrLRM
    
    localCrossMutr.clear()
    allParametersPerfCrossMutrMLPC = []

    while countMLP < setMaxLoopValue:
        for dr in MLPIDs:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
        MLPPickPair = random.sample(MLPIntIndex,2)

        pairDF = paramAllAlgs.iloc[MLPPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['hidden_layer_sizes'] == crossoverDF['hidden_layer_sizes'].iloc[0]) & (paramAllAlgs['alpha'] == crossoverDF['alpha'].iloc[0]) & (paramAllAlgs['tol'] == crossoverDF['tol'].iloc[0]) & (paramAllAlgs['max_iter'] == crossoverDF['max_iter'].iloc[0]) & (paramAllAlgs['activation'] == crossoverDF['activation'].iloc[0]) & (paramAllAlgs['solver'] == crossoverDF['solver'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = MLPClassifier(random_state=RANDOM_SEED)
            params = {'hidden_layer_sizes': [crossoverDF['hidden_layer_sizes'].iloc[0]], 'alpha': [crossoverDF['alpha'].iloc[0]], 'tol': [crossoverDF['tol'].iloc[0]], 'max_iter': [crossoverDF['max_iter'].iloc[0]], 'activation': [crossoverDF['activation'].iloc[0]], 'solver': [crossoverDF['solver'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countMLP
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLP', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrMLPC.append(localCrossMutr[0])
    allParametersPerfCrossMutrMLPC.append(localCrossMutr[1])
    allParametersPerfCrossMutrMLPC.append(localCrossMutr[2])
    allParametersPerfCrossMutrMLPC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrMLPC

    countMLP = 0
    MLPIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrMLPM = []

    while countMLP < setMaxLoopValue:
        for dr in MLPIDs:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
        MLPPickPair = random.sample(MLPIntIndex,1)

        pairDF = paramAllAlgs.iloc[MLPPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            if (column == 'hidden_layer_sizes'):
                randomNumber = (random.randint(10,60), random.randint(4,10))
                listData.append(randomNumber)
                crossoverDF[column] = listData
            else:
                valuePerColumn = pairDF[column].iloc[0]
                listData.append(valuePerColumn)
                crossoverDF[column] = listData
        if (((paramAllAlgs['hidden_layer_sizes'] == crossoverDF['hidden_layer_sizes'].iloc[0]) & (paramAllAlgs['alpha'] == crossoverDF['alpha'].iloc[0]) & (paramAllAlgs['tol'] == crossoverDF['tol'].iloc[0]) & (paramAllAlgs['max_iter'] == crossoverDF['max_iter'].iloc[0]) & (paramAllAlgs['activation'] == crossoverDF['activation'].iloc[0]) & (paramAllAlgs['solver'] == crossoverDF['solver'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = MLPClassifier(random_state=RANDOM_SEED)
            params = {'hidden_layer_sizes': [crossoverDF['hidden_layer_sizes'].iloc[0]], 'alpha': [crossoverDF['alpha'].iloc[0]], 'tol': [crossoverDF['tol'].iloc[0]], 'max_iter': [crossoverDF['max_iter'].iloc[0]], 'activation': [crossoverDF['activation'].iloc[0]], 'solver': [crossoverDF['solver'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countMLP
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLP', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrMLPM.append(localCrossMutr[0])
    allParametersPerfCrossMutrMLPM.append(localCrossMutr[1])
    allParametersPerfCrossMutrMLPM.append(localCrossMutr[2])
    allParametersPerfCrossMutrMLPM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrMLPM

    localCrossMutr.clear()
    allParametersPerfCrossMutrRFC = []

    while countRF < setMaxLoopValue:
        for dr in RFIDs:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))
        RFPickPair = random.sample(RFIntIndex,2)

        pairDF = paramAllAlgs.iloc[RFPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RF', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrRFC.append(localCrossMutr[0])
    allParametersPerfCrossMutrRFC.append(localCrossMutr[1])
    allParametersPerfCrossMutrRFC.append(localCrossMutr[2])
    allParametersPerfCrossMutrRFC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrRFC

    countRF = 0
    RFIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrRFM = []

    while countRF < setMaxLoopValue:
        for dr in RFIDs:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))
        RFPickPair = random.sample(RFIntIndex,1)

        pairDF = paramAllAlgs.iloc[RFPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            if (column == 'n_estimators'):
                randomNumber = random.randint(100, 200)
                listData.append(randomNumber)
                crossoverDF[column] = listData
            else:
                valuePerColumn = pairDF[column].iloc[0]
                listData.append(valuePerColumn)
                crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RF', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrRFM.append(localCrossMutr[0])
    allParametersPerfCrossMutrRFM.append(localCrossMutr[1])
    allParametersPerfCrossMutrRFM.append(localCrossMutr[2])
    allParametersPerfCrossMutrRFM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrRFM

    localCrossMutr.clear()
    allParametersPerfCrossMutrGradBC = []

    while countGradB < setMaxLoopValue:
        for dr in GradBIDs:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))
        GradBPickPair = random.sample(GradBIntIndex,2)

        pairDF = paramAllAlgs.iloc[GradBPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradB', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrGradBC.append(localCrossMutr[0])
    allParametersPerfCrossMutrGradBC.append(localCrossMutr[1])
    allParametersPerfCrossMutrGradBC.append(localCrossMutr[2])
    allParametersPerfCrossMutrGradBC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrGradBC

    countGradB = 0
    GradBIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrGradBM = []
            
    while countGradB < setMaxLoopValue:
        for dr in GradBIDs:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))
        GradPickPair = random.sample(GradBIntIndex,1)

        pairDF = paramAllAlgs.iloc[GradBPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            if (column == 'n_estimators'):
                randomNumber = random.randint(100, 200)
                listData.append(randomNumber)
                crossoverDF[column] = listData
            else:
                valuePerColumn = pairDF[column].iloc[0]
                listData.append(valuePerColumn)
                crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RF', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue

    for loop in range(setMaxLoopValue - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrGradBM.append(localCrossMutr[0])
    allParametersPerfCrossMutrGradBM.append(localCrossMutr[1])
    allParametersPerfCrossMutrGradBM.append(localCrossMutr[2])
    allParametersPerfCrossMutrGradBM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrGradBM

    localCrossMutr.clear()

    allParametersPerfCrossMutr = allParametersPerfCrossMutrKNNC + allParametersPerfCrossMutrKNNM + allParametersPerfCrossMutrLRC + allParametersPerfCrossMutrLRM + allParametersPerfCrossMutrMLPC + allParametersPerfCrossMutrMLPM + allParametersPerfCrossMutrRFC + allParametersPerfCrossMutrRFM + allParametersPerfCrossMutrGradBC + allParametersPerfCrossMutrGradBM

    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrKNNC[0] + allParametersPerfCrossMutrKNNM[0]

    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrKNNC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrKNNM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrKNNC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrKNNM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNC[3]], ignore_index=True)
    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNM[3]], ignore_index=True)
    

    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrLRC[0] + allParametersPerfCrossMutrLRM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRC[3]], ignore_index=True)
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRM[3]], ignore_index=True)

    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrMLPC[0] + allParametersPerfCrossMutrMLPM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrMLPC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrMLPM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrMLPC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrMLPM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPC[3]], ignore_index=True)
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPM[3]], ignore_index=True)

    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrRFC[0] + allParametersPerfCrossMutrRFM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrRFC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrRFM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrRFC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrRFM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFC[3]], ignore_index=True)
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFM[3]], ignore_index=True)

    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrGradBC[0] + allParametersPerfCrossMutrGradBM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrGradBC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrGradBM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrGradBC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrGradBM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBC[3]], ignore_index=True)
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBM[3]], ignore_index=True)

    addKNN = addLR

    addLR = addLR + setMaxLoopValue*2

    addMLP = addLR + setMaxLoopValue*2

    addRF = addMLP + setMaxLoopValue*2

    addGradB = addRF + setMaxLoopValue*2

    return 'Everything Okay'

def crossoverMutation(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd):
    search = GridSearchCV(    
    estimator=clf, param_grid=params, cv=crossValidation, refit='accuracy', 
    scoring=scoring, verbose=0, n_jobs=-1)

    # fit and extract the probabilities
    search.fit(XData, yData)

    # process the results
    cv_results = []
    cv_results.append(search.cv_results_)
    df_cv_results = pd.DataFrame.from_dict(cv_results)

    # number of models stored
    number_of_models = len(df_cv_results.iloc[0][0])

    # initialize results per row
    df_cv_results_per_row = []

    # loop through number of models
    modelsIDs = []
    for i in range(number_of_models):
        number = AlgorithmsIDsEnd+i
        modelsIDs.append(eachAlgor+str(number))
         # initialize results per item
        df_cv_results_per_item = []
        for column in df_cv_results.iloc[0]:
            df_cv_results_per_item.append(column[i])
        df_cv_results_per_row.append(df_cv_results_per_item)

    # store the results into a pandas dataframe
    df_cv_results_classifiers = pd.DataFrame(data = df_cv_results_per_row, columns= df_cv_results.columns)

    # copy and filter in order to get only the metrics
    metrics = df_cv_results_classifiers.copy()
    metrics = metrics.filter(['mean_test_accuracy','mean_test_precision_weighted','mean_test_recall_weighted','mean_test_f1_weighted','mean_test_roc_auc_ovo_weighted']) 

    # concat parameters and performance
    parametersPerformancePerModel = pd.DataFrame(df_cv_results_classifiers['params'])
    parametersLocal = parametersPerformancePerModel['params'].copy()

    Models = []
    for index, items in enumerate(parametersLocal):
        Models.append(index)

    parametersLocalNew = [ parametersLocal[your_key] for your_key in Models ]

    perModelProb = []
    
    resultsWeighted = []
    resultsCorrCoef = []
    resultsLogLoss = []
    resultsLogLossFinal = []

    # influence calculation for all the instances
    inputs = range(len(XData))
    num_cores = multiprocessing.cpu_count()
    
    for eachModelParameters in parametersLocalNew:
        clf.set_params(**eachModelParameters)
        clf.fit(XData, yData) 
        yPredict = clf.predict(XData)
        yPredict = np.nan_to_num(yPredict)
        yPredictProb = cross_val_predict(clf, XData, yData, cv=crossValidation, method='predict_proba')
        yPredictProb = np.nan_to_num(yPredictProb)
        perModelProb.append(yPredictProb.tolist())

        resultsWeighted.append(geometric_mean_score(yData, yPredict, average='weighted'))
        resultsCorrCoef.append(matthews_corrcoef(yData, yPredict))
        resultsLogLoss.append(log_loss(yData, yPredictProb, normalize=True))

    maxLog = max(resultsLogLoss)
    minLog = min(resultsLogLoss)
    for each in resultsLogLoss:
        resultsLogLossFinal.append((each-minLog)/(maxLog-minLog))

    metrics.insert(5,'geometric_mean_score_weighted',resultsWeighted)
    metrics.insert(6,'matthews_corrcoef',resultsCorrCoef)
    metrics.insert(7,'log_loss',resultsLogLossFinal)

    perModelProbPandas = pd.DataFrame(perModelProb)

    results.append(modelsIDs)
    results.append(parametersPerformancePerModel)
    results.append(metrics)
    results.append(perModelProbPandas)

    return results

def PreprocessingIDsCM():
    dicKNNC = allParametersPerfCrossMutr[0]
    dicKNNM = allParametersPerfCrossMutr[4]
    dicLRC = allParametersPerfCrossMutr[8]
    dicLRM = allParametersPerfCrossMutr[12]
    dicMLPC = allParametersPerfCrossMutr[16]
    dicMLPM = allParametersPerfCrossMutr[20]
    dicRFC = allParametersPerfCrossMutr[24]
    dicRFM = allParametersPerfCrossMutr[28]
    dicGradBC = allParametersPerfCrossMutr[32]
    dicGradBM = allParametersPerfCrossMutr[36]
    

    df_concatIDs = dicKNNC + dicKNNM + dicLRC + dicLRM + dicMLPC + dicMLPM + dicRFC + dicRFM + dicGradBC + dicGradBM
    return df_concatIDs

def PreprocessingMetricsCM():
    dicKNNC = allParametersPerfCrossMutr[2]
    dicKNNM = allParametersPerfCrossMutr[6]
    dicLRC = allParametersPerfCrossMutr[10]
    dicLRM = allParametersPerfCrossMutr[14]
    dicMLPC = allParametersPerfCrossMutr[18]
    dicMLPM = allParametersPerfCrossMutr[22]
    dicRFC = allParametersPerfCrossMutr[26]
    dicRFM = allParametersPerfCrossMutr[30]
    dicGradBC = allParametersPerfCrossMutr[34]
    dicGradBM = allParametersPerfCrossMutr[38]

    dfKNNC = pd.DataFrame.from_dict(dicKNNC)
    dfKNNM = pd.DataFrame.from_dict(dicKNNM)
    dfLRC = pd.DataFrame.from_dict(dicLRC)
    dfLRM = pd.DataFrame.from_dict(dicLRM)
    dfMLPC = pd.DataFrame.from_dict(dicMLPC)
    dfMLPM = pd.DataFrame.from_dict(dicMLPM)
    dfRFC = pd.DataFrame.from_dict(dicRFC)
    dfRFM = pd.DataFrame.from_dict(dicRFM)
    dfGradBC = pd.DataFrame.from_dict(dicGradBC)
    dfGradBM = pd.DataFrame.from_dict(dicGradBM)

    df_concatMetrics = pd.concat([dfKNNC, dfKNNM, dfLRC, dfLRM, dfMLPC, dfMLPM, dfRFC, dfRFM, dfGradBC, dfGradBM])
    df_concatMetrics = df_concatMetrics.reset_index(drop=True)
    return df_concatMetrics

def PreprocessingPredCM():
    dicKNNC = allParametersPerfCrossMutr[3]
    dicKNNM = allParametersPerfCrossMutr[7]
    dicLRC = allParametersPerfCrossMutr[11]
    dicLRM = allParametersPerfCrossMutr[15]
    dicMLPC = allParametersPerfCrossMutr[19]
    dicMLPM = allParametersPerfCrossMutr[23]
    dicRFC = allParametersPerfCrossMutr[27]
    dicRFM = allParametersPerfCrossMutr[31]
    dicGradBC = allParametersPerfCrossMutr[35]
    dicGradBM = allParametersPerfCrossMutr[39]

    dfKNNC = pd.DataFrame.from_dict(dicKNNC)
    dfKNNM = pd.DataFrame.from_dict(dicKNNM)
    dfLRC = pd.DataFrame.from_dict(dicLRC)
    dfLRM = pd.DataFrame.from_dict(dicLRM)
    dfMLPC = pd.DataFrame.from_dict(dicMLPC)
    dfMLPM = pd.DataFrame.from_dict(dicMLPM)
    dfRFC = pd.DataFrame.from_dict(dicRFC)
    dfRFM = pd.DataFrame.from_dict(dicRFM)
    dfGradBC = pd.DataFrame.from_dict(dicGradBC)
    dfGradBM = pd.DataFrame.from_dict(dicGradBM)

    dfKNN = pd.concat([dfKNNC, dfKNNM])

    dfLR = pd.concat([dfLRC, dfLRM])

    dfMLP = pd.concat([dfMLPC, dfMLPM])

    dfRF = pd.concat([dfRFC, dfRFM])

    dfGradB = pd.concat([dfGradBC, dfGradBM])

    df_concatProbs = pd.concat([dfKNNC, dfKNNM, dfLRC, dfLRM, dfMLPC, dfMLPM, dfRFC, dfRFM, dfGradBC, dfGradBM])

    predictionsKNN = []
    for column, content in dfKNN.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsKNN.append(el)

    predictionsLR = []
    for column, content in dfLR.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsLR.append(el)

    predictionsMLP = []
    for column, content in dfMLP.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsMLP.append(el)

    predictionsRF = []
    for column, content in dfRF.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsRF.append(el)

    predictionsGradB = []
    for column, content in dfGradB.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsGradB.append(el)
    
    predictions = []
    for column, content in df_concatProbs.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictions.append(el)

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

def PreprocessingParamCM():
    dicKNNC = allParametersPerfCrossMutr[1]
    dicKNNM = allParametersPerfCrossMutr[5]
    dicLRC = allParametersPerfCrossMutr[9]
    dicLRM = allParametersPerfCrossMutr[13]
    dicMLPC = allParametersPerfCrossMutr[17]
    dicMLPM = allParametersPerfCrossMutr[21]
    dicRFC = allParametersPerfCrossMutr[25]
    dicRFM = allParametersPerfCrossMutr[29]
    dicGradBC = allParametersPerfCrossMutr[33]
    dicGradBM = allParametersPerfCrossMutr[37]

    dicKNNC = dicKNNC['params']
    dicKNNM = dicKNNM['params']
    dicLRC = dicLRC['params']
    dicLRM = dicLRM['params']
    dicMLPC = dicMLPC['params']
    dicMLPM = dicMLPM['params']
    dicRFC = dicRFC['params']
    dicRFM = dicRFM['params']
    dicGradBC = dicGradBC['params']
    dicGradBM = dicGradBM['params']

    
    dicKNNC = {int(k):v for k,v in dicKNNC.items()}
    dicKNNM = {int(k):v for k,v in dicKNNM.items()}
    dicLRC = {int(k):v for k,v in dicLRC.items()}
    dicLRM = {int(k):v for k,v in dicLRM.items()}
    dicMLPC = {int(k):v for k,v in dicMLPC.items()}
    dicMLPM = {int(k):v for k,v in dicMLPM.items()}
    dicRFC = {int(k):v for k,v in dicRFC.items()}
    dicRFM = {int(k):v for k,v in dicRFM.items()}
    dicGradBC = {int(k):v for k,v in dicGradBC.items()}
    dicGradBM = {int(k):v for k,v in dicGradBM.items()}

    dfKNNC = pd.DataFrame.from_dict(dicKNNC)
    dfKNNM = pd.DataFrame.from_dict(dicKNNM)
    dfLRC = pd.DataFrame.from_dict(dicLRC)
    dfLRM = pd.DataFrame.from_dict(dicLRM)
    dfMLPC = pd.DataFrame.from_dict(dicMLPC)
    dfMLPM = pd.DataFrame.from_dict(dicMLPM)
    dfRFC = pd.DataFrame.from_dict(dicRFC)
    dfRFM = pd.DataFrame.from_dict(dicRFM)
    dfGradBC = pd.DataFrame.from_dict(dicGradBC)
    dfGradBM = pd.DataFrame.from_dict(dicGradBM)

    dfKNNC = dfKNNC.T
    dfKNNM = dfKNNM.T
    dfLRC = dfLRC.T
    dfLRM = dfLRM.T
    dfMLPC = dfMLPC.T
    dfMLPM = dfMLPM.T
    dfRFC = dfRFC.T
    dfRFM = dfRFM.T
    dfGradBC = dfGradBC.T
    dfGradBM = dfGradBM.T

    df_params = pd.concat([dfKNNC, dfKNNM, dfLRC, dfLRM, dfMLPC, dfMLPM, dfRFC, dfRFM, dfGradBC, dfGradBM])
    df_params = df_params.reset_index(drop=True)
    return df_params

def PreprocessingParamSepCM():
    dicKNNC = allParametersPerfCrossMutr[1]
    dicKNNM = allParametersPerfCrossMutr[5]
    dicLRC = allParametersPerfCrossMutr[9]
    dicLRM = allParametersPerfCrossMutr[13]
    dicMLPC = allParametersPerfCrossMutr[17]
    dicMLPM = allParametersPerfCrossMutr[21]
    dicRFC = allParametersPerfCrossMutr[25]
    dicRFM = allParametersPerfCrossMutr[29]
    dicGradBC = allParametersPerfCrossMutr[33]
    dicGradBM = allParametersPerfCrossMutr[37]

    dicKNNC = dicKNNC['params']
    dicKNNM = dicKNNM['params']
    dicLRC = dicLRC['params']
    dicLRM = dicLRM['params']
    dicMLPC = dicMLPC['params']
    dicMLPM = dicMLPM['params']
    dicRFC = dicRFC['params']
    dicRFM = dicRFM['params']
    dicGradBC = dicGradBC['params']
    dicGradBM = dicGradBM['params']

    dicKNNC = {int(k):v for k,v in dicKNNC.items()}
    dicKNNM = {int(k):v for k,v in dicKNNM.items()}
    dicLRC = {int(k):v for k,v in dicLRC.items()}
    dicLRM = {int(k):v for k,v in dicLRM.items()}
    dicMLPC = {int(k):v for k,v in dicMLPC.items()}
    dicMLPM = {int(k):v for k,v in dicMLPM.items()}
    dicRFC = {int(k):v for k,v in dicRFC.items()}
    dicRFM = {int(k):v for k,v in dicRFM.items()}
    dicGradBC = {int(k):v for k,v in dicGradBC.items()}
    dicGradBM = {int(k):v for k,v in dicGradBM.items()}

    dfKNNC = pd.DataFrame.from_dict(dicKNNC)
    dfKNNM = pd.DataFrame.from_dict(dicKNNM)
    dfLRC = pd.DataFrame.from_dict(dicLRC)
    dfLRM = pd.DataFrame.from_dict(dicLRM)
    dfMLPC = pd.DataFrame.from_dict(dicMLPC)
    dfMLPM = pd.DataFrame.from_dict(dicMLPM)
    dfRFC = pd.DataFrame.from_dict(dicRFC)
    dfRFM = pd.DataFrame.from_dict(dicRFM)
    dfGradBC = pd.DataFrame.from_dict(dicGradBC)
    dfGradBM = pd.DataFrame.from_dict(dicGradBM)

    dfKNNC = dfKNNC.T
    dfKNNM = dfKNNM.T
    dfLRC = dfLRC.T
    dfLRM = dfLRM.T
    dfMLPC = dfMLPC.T
    dfMLPM = dfMLPM.T
    dfRFC = dfRFC.T
    dfRFM = dfRFM.T
    dfGradBC = dfGradBC.T
    dfGradBM = dfGradBM.T

    return [dfKNNC, dfKNNM, dfLRC, dfLRM, dfMLPC, dfMLPM, dfRFC, dfRFM, dfGradBC, dfGradBM]

# remove that maybe!
def preProcsumPerMetricCM(factors):
    sumPerClassifier = []
    loopThroughMetrics = PreprocessingMetricsCM()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    loopThroughMetrics.loc[:, 'log_loss'] = 1 - loopThroughMetrics.loc[:, 'log_loss']
    for row in loopThroughMetrics.iterrows():
        rowSum = 0
        name, values = row
        for loop, elements in enumerate(values):
            rowSum = elements*factors[loop] + rowSum
        if sum(factors) is 0:
            sumPerClassifier = 0
        else:
            sumPerClassifier.append(rowSum/sum(factors) * 100)
    return sumPerClassifier

def preProcMetricsAllAndSelCM():
    loopThroughMetrics = PreprocessingMetricsCM()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    global factors
    metricsPerModelColl = []
    metricsPerModelColl.append(loopThroughMetrics['mean_test_accuracy'])
    metricsPerModelColl.append(loopThroughMetrics['geometric_mean_score_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_precision_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_recall_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_f1_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['matthews_corrcoef'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_roc_auc_ovo_weighted'])
    metricsPerModelColl.append(loopThroughMetrics['log_loss'])

    f=lambda a: (abs(a)+a)/2
    for index, metric in enumerate(metricsPerModelColl):
        if (index == 5):
            metricsPerModelColl[index] = ((f(metric))*factors[index]) * 100
        elif (index == 7):
            metricsPerModelColl[index] = ((1 - metric)*factors[index] ) * 100
        else:  
            metricsPerModelColl[index] = (metric*factors[index]) * 100
        metricsPerModelColl[index] = metricsPerModelColl[index].to_json()
    return metricsPerModelColl

# Sending the overview classifiers' results to be visualized as a scatterplot
@app.route('/data/PlotCrossMutate', methods=["GET", "POST"])
def SendToPlotCM():
    while (len(DataResultsRaw) != DataRawLength):
        pass
    PreProcessingInitial()
    response = {    
        'OverviewResultsCM': ResultsCM
    }
    return jsonify(response)

def PreProcessingInitial(): 
    XModels = PreprocessingMetricsCM()
    global allParametersPerfCrossMutr

    XModels = XModels.fillna(0)

    ModelSpaceMDSCM = FunMDS(XModels)
    ModelSpaceTSNECM = FunTsne(XModels)
    ModelSpaceTSNECM = ModelSpaceTSNECM.tolist()
    ModelSpaceUMAPCM = FunUMAP(XModels)

    PredictionProbSel = PreprocessingPredCM()

    CrossMutateResults(ModelSpaceMDSCM,ModelSpaceTSNECM,ModelSpaceUMAPCM,PredictionProbSel)

def CrossMutateResults(ModelSpaceMDSCM,ModelSpaceTSNECM,ModelSpaceUMAPCM,PredictionProbSel):

    global ResultsCM
    global AllTargets
    ResultsCM = []

    parametersGen = PreprocessingParamCM()
    metricsPerModel = preProcMetricsAllAndSelCM()
    sumPerClassifier = preProcsumPerMetricCM(factors)
    ModelsIDs = PreprocessingIDsCM()
    

    parametersGenPD = parametersGen.to_json(orient='records')
    XDataJSONEntireSet = XData.to_json(orient='records')
    XDataColumns = XData.columns.tolist()

    ResultsCM.append(json.dumps(ModelsIDs))
    ResultsCM.append(json.dumps(sumPerClassifier))
    ResultsCM.append(json.dumps(parametersGenPD))
    ResultsCM.append(json.dumps(metricsPerModel))
    ResultsCM.append(json.dumps(XDataJSONEntireSet))
    ResultsCM.append(json.dumps(XDataColumns))
    ResultsCM.append(json.dumps(yData))
    ResultsCM.append(json.dumps(target_names))
    ResultsCM.append(json.dumps(AllTargets))
    ResultsCM.append(json.dumps(ModelSpaceMDSCM))
    ResultsCM.append(json.dumps(ModelSpaceTSNECM))
    ResultsCM.append(json.dumps(ModelSpaceUMAPCM))
    ResultsCM.append(json.dumps(PredictionProbSel))

    return ResultsCM

def PreprocessingPredSel(SelectedIDs):

    global addKNN
    global addLR
    global addMLP
    global addRF
    global addGradB

    numberIDKNN = []
    numberIDLR = []
    numberIDMLP = []
    numberIDRF = []
    numberIDGradB = []

    for el in SelectedIDs:
        match = re.match(r"([a-z]+)([0-9]+)", el, re.I)
        if match:
            items = match.groups()
            if (items[0] == 'KNN'):
                numberIDKNN.append(int(items[1]) - addKNN)
            elif (items[0] == 'LR'):
                numberIDLR.append(int(items[1]) - addLR)
            elif (items[0] == 'MLP'):
                numberIDMLP.append(int(items[1]) - addMLP)
            elif (items[0] == 'RF'):
                numberIDRF.append(int(items[1]) - addRF)
            else:
                numberIDGradB.append(int(items[1]) - addGradB)

    dicKNN = allParametersPerformancePerModel[3]
    dicLR = allParametersPerformancePerModel[7]
    dicMLP = allParametersPerformancePerModel[11]
    dicRF = allParametersPerformancePerModel[15]
    dicGradB = allParametersPerformancePerModel[19]

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfKNN = dfKNN.loc[numberIDKNN]

    dfLR = pd.DataFrame.from_dict(dicLR)
    dfLR = dfLR.loc[numberIDLR]

    dfLR.index += addKNN

    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfMLP = dfMLP.loc[numberIDMLP]

    dfMLP.index += addKNN + addLR

    dfRF = pd.DataFrame.from_dict(dicRF)
    dfRF = dfRF.loc[numberIDRF]

    dfRF.index += addKNN + addLR + addMLP

    dfGradB = pd.DataFrame.from_dict(dicGradB)
    dfGradB = dfGradB.loc[numberIDGradB]

    dfGradB.index += addKNN + addLR + addMLP + addRF

    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])

    predictionsKNN = []
    for column, content in dfKNN.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsKNN.append(el)

    predictionsLR = []
    for column, content in dfLR.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsLR.append(el)

    predictionsMLP = []
    for column, content in dfMLP.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsMLP.append(el)

    predictionsRF = []
    for column, content in dfRF.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsRF.append(el)

    predictionsGradB = []
    for column, content in dfGradB.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsGradB.append(el)

    predictions = []
    for column, content in df_concatProbs.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictions.append(el)

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/SendtoSeverSelIDs', methods=["GET", "POST"])
def RetrieveSelIDsPredict():
    global ResultsSelPred
    ResultsSelPred = []
    RetrieveIDsSelection = request.get_data().decode('utf8').replace("'", '"')
    RetrieveIDsSelection = json.loads(RetrieveIDsSelection)
    RetrieveIDsSelection = RetrieveIDsSelection['predictSelectionIDs']
    
    ResultsSelPred = PreprocessingPredSel(RetrieveIDsSelection)

    return 'Everything Okay'

@app.route('/data/RetrievePredictions', methods=["GET", "POST"])
def SendPredictSel():
    global ResultsSelPred
    response = {    
        'PredictSel': ResultsSelPred
    }
    return jsonify(response)

def PreprocessingPredSelEnsem(SelectedIDsEnsem):

    numberIDKNN = []
    numberIDLR = []
    numberIDMLP = []
    numberIDRF = []
    numberIDGradB = []

    for el in SelectedIDsEnsem:
        match = re.match(r"([a-z]+)([0-9]+)", el, re.I)
        if match:
            items = match.groups()
            if (items[0] == 'KNN'):
                numberIDKNN.append(int(items[1]))
            elif (items[0] == 'LR'):
                numberIDLR.append(int(items[1]))
            elif (items[0] == 'MLP'):
                numberIDLR.append(int(items[1]))
            elif (items[0] == 'RF'):
                numberIDLR.append(int(items[1]))
            else:
                numberIDLR.append(int(items[1]))

    dicKNN = allParametersPerformancePerModel[3]
    dicLR = allParametersPerformancePerModel[7]
    dicMLP = allParametersPerformancePerModel[11]
    dicRF = allParametersPerformancePerModel[15]
    dicGradB = allParametersPerformancePerModel[19]

    dfKNN = pd.DataFrame.from_dict(dicKNN)
    dfLR = pd.DataFrame.from_dict(dicLR)
    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfRF = pd.DataFrame.from_dict(dicRF)
    dfGradB = pd.DataFrame.from_dict(dicGradB)

    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])
    df_concatProbs = df_concatProbs.reset_index(drop=True)

    dfKNN = df_concatProbs.loc[numberIDKNN]
    dfLR = df_concatProbs.loc[numberIDLR]
    dfMLP = df_concatProbs.loc[numberIDMLP]
    dfRF = df_concatProbs.loc[numberIDRF]
    dfGradB = df_concatProbs.loc[numberIDGradB]

    df_concatProbs = pd.DataFrame()
    df_concatProbs = df_concatProbs.iloc[0:0]
    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])

    predictionsKNN = []
    for column, content in dfKNN.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsKNN.append(el)

    predictionsLR = []
    for column, content in dfLR.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsLR.append(el)

    predictionsMLP = []
    for column, content in dfMLP.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsMLP.append(el)

    predictionsRF = []
    for column, content in dfRF.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsRF.append(el)

    predictionsGradB = []
    for column, content in dfGradB.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictionsGradB.append(el)

    predictions = []
    for column, content in df_concatProbs.items():
        el = [sum(x)/len(x) for x in zip(*content)]
        predictions.append(el)

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/SendtoSeverSelIDsEnsem', methods=["GET", "POST"])
def RetrieveSelIDsPredictEnsem():
    global ResultsSelPredEnsem
    ResultsSelPredEnsem = []
    RetrieveIDsSelectionEnsem = request.get_data().decode('utf8').replace("'", '"')
    RetrieveIDsSelectionEnsem = json.loads(RetrieveIDsSelectionEnsem)
    RetrieveIDsSelectionEnsem = RetrieveIDsSelectionEnsem['predictSelectionIDsCM']
    
    ResultsSelPredEnsem = PreprocessingPredSelEnsem(RetrieveIDsSelectionEnsem)

    return 'Everything Okay'

@app.route('/data/RetrievePredictionsEnsem', methods=["GET", "POST"])
def SendPredictSelEnsem():
    global ResultsSelPredEnsem
    response = {    
        'PredictSelEnsem': ResultsSelPredEnsem
    }
    return jsonify(response)