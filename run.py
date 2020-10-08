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

from joblib import Parallel, delayed, Memory

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth
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

    global yDataSorted
    yDataSorted = []

    global PerClassResultsClass0
    PerClassResultsClass0 = []
    global PerClassResultsClass1
    PerClassResultsClass1 = []

    global Results
    Results = []
    global ResultsCM
    ResultsCM = []
    global ResultsCMSecond
    ResultsCMSecond = []

    global DataRawLength
    global DataResultsRaw
    global previousState
    previousState = []

    global filterActionFinal
    filterActionFinal = ''

    global dataSpacePointsIDs
    dataSpacePointsIDs = []

    global RANDOM_SEED
    RANDOM_SEED = 42

    global KNNModelsCount
    global LRModelsCount
    global MLPModelsCount
    global RFModelsCount
    global GradBModelsCount

    global factors
    factors = [1,1,1,1,0,0,0,0]

    global crossValidation
    crossValidation = 5

    global randomSearchVar
    randomSearchVar = 100

    global stage1addKNN
    global stage1addLR
    global stage1addMLP
    global stage1addRF
    global stage1addGradB
    global stageTotalReached

    stage1addKNN = 0
    stage1addLR = 0
    stage1addMLP = 0
    stage1addRF = 0
    stage1addGradB = 0
    stageTotalReached = randomSearchVar*5

    global keyData
    keyData = 0

    KNNModelsCount = 0
    LRModelsCount = KNNModelsCount+randomSearchVar
    MLPModelsCount = LRModelsCount+randomSearchVar
    RFModelsCount = MLPModelsCount+randomSearchVar
    GradBModelsCount = RFModelsCount+randomSearchVar

    global storeClass0
    storeClass0 = 0

    global storeClass1
    storeClass1 = 0

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
    scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro', 'f1_macro': 'f1_macro', 'roc_auc_ovo': 'roc_auc_ovo'}

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

    global names_labels
    names_labels = []

    global keySend
    keySend=0

    return 'The reset was done!'

# retrieve data from client and select the correct data set
@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRequest', methods=["GET", "POST"])
def retrieveFileName():
    global DataRawLength
    global DataResultsRaw
    global DataResultsRawTest
    global DataRawLengthTest

    global yDataSorted
    yDataSorted = []

    fileName = request.get_data().decode('utf8').replace("'", '"')
    data = json.loads(fileName)  

    global filterActionFinal
    filterActionFinal = ''

    global dataSpacePointsIDs
    dataSpacePointsIDs = []

    global RANDOM_SEED
    RANDOM_SEED = 42

    global keyData
    keyData = 0

    global factors
    factors = data['Factors']

    global crossValidation
    crossValidation = int(data['CrossValidation'])

    global randomSearchVar
    randomSearchVar = int(data['RandomSearch'])

    global stage1addKNN
    global stage1addLR
    global stage1addMLP
    global stage1addRF
    global stage1addGradB
    global stageTotalReached

    stage1addKNN = 0
    stage1addLR = 0
    stage1addMLP = 0
    stage1addRF = 0
    stage1addGradB = 0
    stageTotalReached = randomSearchVar*5

    global storeClass0
    storeClass0 = 0

    global storeClass1
    storeClass1 = 0

    global XData
    XData = []

    global previousState
    previousState = []

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
    scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro', 'f1_macro': 'f1_macro', 'roc_auc_ovo': 'roc_auc_ovo'}

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

    global names_labels
    names_labels = []

    global keySend
    keySend=0

    DataRawLength = -1
    DataRawLengthTest = -1

    if data['fileName'] == 'HeartC':
        CollectionDB = mongo.db.HeartC.find()
        names_labels.append('Healthy')
        names_labels.append('Diseased')
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

    global storeClass0
    global storeClass1

    for item in yData:
        if (item == 0):
            storeClass0 = storeClass0 + 1
        else:
            storeClass1 = storeClass1 + 1
    
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
    
    global storeClass0
    global storeClass1

    for item in yData:
        if (item == 0):
            storeClass0 = storeClass0 + 1
        else:
            storeClass1 = storeClass1 + 1

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
    global countAllModels

    # loop through the algorithms
    global allParametersPerformancePerModel
    global HistoryPreservation

    for eachAlgor in algorithms:
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
            params = {'n_estimators': list(range(20, 100)), 'max_depth': list(range(2, 20)), 'criterion': ['gini', 'entropy']}
            countAllModels = countAllModels + randomSearchVar
            AlgorithmsIDsEnd = countAllModels
        else: 
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': list(range(20, 100)), 'loss': ['deviance', 'exponential'], 'learning_rate': list(np.arange(0.01,0.56,0.11)), 'subsample': list(np.arange(0.1,1,0.1)), 'criterion': ['friedman_mse', 'mse', 'mae']}
            countAllModels = countAllModels + randomSearchVar
            AlgorithmsIDsEnd = countAllModels
            countAllModels = countAllModels + randomSearchVar
        allParametersPerformancePerModel = randomSearch(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd)
    HistoryPreservation = allParametersPerformancePerModel.copy()
    # call the function that sends the results to the frontend

    return 'Everything Okay'

location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def randomSearch(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd):
    print(clf)
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
    metrics = metrics.filter(['mean_test_accuracy','mean_test_precision_macro','mean_test_recall_macro','mean_test_f1_macro','mean_test_roc_auc_ovo']) 
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

        resultsWeighted.append(geometric_mean_score(yData, yPredict, average='macro'))
        resultsCorrCoef.append(matthews_corrcoef(yData, yPredict))
        resultsLogLoss.append(log_loss(yData, yPredictProb, normalize=True))

    maxLog = max(resultsLogLoss)
    minLog = min(resultsLogLoss)
    for each in resultsLogLoss:
        resultsLogLossFinal.append((each-minLog)/(maxLog-minLog))

    metrics.insert(5,'geometric_mean_score_macro',resultsWeighted)
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
    global allParametersPerformancePerModel
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

def PreprocessingMetricsEnsem():
    global allParametersPerformancePerModelEnsem
    dicKNN = allParametersPerformancePerModelEnsem[2]
    dicLR = allParametersPerformancePerModelEnsem[6]
    dicMLP = allParametersPerformancePerModelEnsem[10]
    dicRF = allParametersPerformancePerModelEnsem[14]
    dicGradB = allParametersPerformancePerModelEnsem[18]

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
    df_concatProbs.reset_index(drop=True)

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
    
    global storeClass0
    global storeClass1
    global yDataSorted

    firstElKNN = []
    firstElLR = []
    firstElMLP = []
    firstElRF = []
    firstElGradB = []
    firstElPredAv = []
    lastElKNN = []
    lastElLR = []
    lastElMLP = []
    lastElRF = []
    lastElGradB = []
    lastElPredAv = []
    yDataSortedFirst = []
    yDataSortedLast = []

    for index, item in enumerate(yData):
        if (item == 0):
            if (len(predictionsKNN[index]) != 0):
                firstElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                firstElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                firstElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                firstElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                firstElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                firstElPredAv.append(predictions[index][item]*100)
            yDataSortedFirst.append(item)
        else:
            if (len(predictionsKNN[index]) != 0):
                lastElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                lastElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                lastElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                lastElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                lastElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                lastElPredAv.append(predictions[index][item]*100)
            yDataSortedLast.append(item)

    if (storeClass0 > 169 & storeClass1 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)
        lastElKNN = computeClusters(lastElKNN)
        firstElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass0 < 169 & storeClass1 > 169):
                
        lastElKNN = computeClusters(lastElKNN)
        lastElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass1 < 169 & storeClass0 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)

    else:
        pass

    predictionsKNN = firstElKNN + lastElKNN
    predictionsLR = firstElLR + lastElLR        
    predictionsMLP = firstElMLP + lastElMLP
    predictionsRF = firstElRF + lastElRF
    predictionsGradB = firstElGradB + lastElGradB
    predictions = firstElPredAv + lastElPredAv
    yDataSorted = yDataSortedFirst + yDataSortedLast

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

def computeClusters(dataLocal):
    if (dataLocal.length != 0):
        X = np.array(list(zip(dataLocal,np.zeros(len(dataLocal)))), dtype=np.int)
        bandwidth = estimate_bandwidth(X, quantile=0.015, random_state=RANDOM_SEED, n_jobs=-1)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        gatherPoints = []
        for k in range(n_clusters_):
            my_members = labels == k
            gatherPoints.append(sum(X[my_members, 0])/len(X[my_members, 0]))
    else:
        gatherPoints = []
    return gatherPoints

def EnsembleIDs():
    global EnsembleActive
    global numberIDKNNGlob
    global numberIDLRGlob
    global numberIDMLPGlob
    global numberIDRFGlob
    global numberIDGradBGlob
    numberIDKNNGlob = []
    numberIDLRGlob = []
    numberIDMLPGlob = []
    numberIDRFGlob = []
    numberIDGradBGlob = []

    for el in EnsembleActive:
        match = re.match(r"([a-z]+)([0-9]+)", el, re.I)
        if match:
            items = match.groups()
            if ((items[0] == "KNN") | (items[0] == "KNNC") | (items[0] == "KNNM") | (items[0] == "KNNCC") | (items[0] == "KNNCM") | (items[0] == "KNNMC") | (items[0] == "KNNMM")):
                numberIDKNNGlob.append(int(items[1]))
            elif ((items[0] == "LR") | (items[0] == "LRC") | (items[0] == "LRM") | (items[0] == "LRCC") | (items[0] == "LRCM") | (items[0] == "LRMC") | (items[0] == "LRMM")):
                numberIDLRGlob.append(int(items[1]))
            elif ((items[0] == "MLP") | (items[0] == "MLPC") | (items[0] == "MLPM") | (items[0] == "MLPCC") | (items[0] == "MLPCM") | (items[0] == "MLPMC") | (items[0] == "MLPMM")):
                numberIDMLPGlob.append(int(items[1]))
            elif ((items[0] == "RF") | (items[0] == "RFC") | (items[0] == "RFM") | (items[0] == "RFCC") | (items[0] == "RFCM") | (items[0] == "RFMC") | (items[0] == "RFMM")):
                numberIDRFGlob.append(int(items[1]))
            else:
                numberIDGradBGlob.append(int(items[1]))

    EnsembleIdsAll = numberIDKNNGlob + numberIDLRGlob + numberIDMLPGlob + numberIDRFGlob + numberIDGradBGlob
    return EnsembleIdsAll

def PreprocessingPredEnsemble():

    global EnsembleActive
    global allParametersPerformancePerModelEnsem
    numberIDKNN = []
    numberIDLR = []
    numberIDMLP = []
    numberIDRF = []
    numberIDGradB = []

    for el in EnsembleActive:
        match = re.match(r"([a-z]+)([0-9]+)", el, re.I)
        if match:
            items = match.groups()

            if ((items[0] == "KNN") | (items[0] == "KNNC") | (items[0] == "KNNM") | (items[0] == "KNNCC") | (items[0] == "KNNCM") | (items[0] == "KNNMC") | (items[0] == "KNNMM")):
                numberIDKNN.append(int(items[1]))
            elif ((items[0] == "LR") | (items[0] == "LRC") | (items[0] == "LRM") | (items[0] == "LRCC") | (items[0] == "LRCM") | (items[0] == "LRMC") | (items[0] == "LRMM")):
                numberIDLR.append(int(items[1]))
            elif ((items[0] == "MLP") | (items[0] == "MLPC") | (items[0] == "MLPM") | (items[0] == "MLPCC") | (items[0] == "MLPCM") | (items[0] == "MLPMC") | (items[0] == "MLPMM")):
                numberIDMLP.append(int(items[1]))
            elif ((items[0] == "RF") | (items[0] == "RFC") | (items[0] == "RFM") | (items[0] == "RFCC") | (items[0] == "RFCM") | (items[0] == "RFMC") | (items[0] == "RFMM")):
                numberIDRF.append(int(items[1]))
            else:
                numberIDGradB.append(int(items[1]))
 
    dicKNN = allParametersPerformancePerModelEnsem[3]
    dicLR = allParametersPerformancePerModelEnsem[7]
    dicMLP = allParametersPerformancePerModelEnsem[11]
    dicRF = allParametersPerformancePerModelEnsem[15]
    dicGradB = allParametersPerformancePerModelEnsem[19]

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

    global storeClass0
    global storeClass1
    global yDataSorted

    firstElKNN = []
    firstElLR = []
    firstElMLP = []
    firstElRF = []
    firstElGradB = []
    firstElPredAv = []
    lastElKNN = []
    lastElLR = []
    lastElMLP = []
    lastElRF = []
    lastElGradB = []
    lastElPredAv = []
    yDataSortedFirst = []
    yDataSortedLast = []

    for index, item in enumerate(yData):
        if (item == 0):
            if (len(predictionsKNN[index]) != 0):
                firstElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                firstElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                firstElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                firstElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                firstElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                firstElPredAv.append(predictions[index][item]*100)
            yDataSortedFirst.append(item)
        else:
            if (len(predictionsKNN[index]) != 0):
                lastElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                lastElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                lastElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                lastElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                lastElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                lastElPredAv.append(predictions[index][item]*100)
            yDataSortedLast.append(item)

    if (storeClass0 > 169 & storeClass1 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)
        lastElKNN = computeClusters(lastElKNN)
        firstElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass0 < 169 & storeClass1 > 169):
                
        lastElKNN = computeClusters(lastElKNN)
        lastElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass1 < 169 & storeClass0 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)

    else:
        pass

    predictionsKNN = firstElKNN + lastElKNN
    predictionsLR = firstElLR + lastElLR        
    predictionsMLP = firstElMLP + lastElMLP
    predictionsRF = firstElRF + lastElRF
    predictionsGradB = firstElGradB + lastElGradB
    predictions = firstElPredAv + lastElPredAv
    yDataSorted = yDataSortedFirst + yDataSortedLast

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

def PreprocessingParamEnsem():
    dicKNN = allParametersPerformancePerModelEnsem[1]
    dicLR = allParametersPerformancePerModelEnsem[5]
    dicMLP = allParametersPerformancePerModelEnsem[9]
    dicRF = allParametersPerformancePerModelEnsem[13]
    dicGradB = allParametersPerformancePerModelEnsem[17]

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
        if sum(factors) == 0:
            sumPerClassifier = 0
        else:
            sumPerClassifier.append(rowSum/sum(factors) * 100)
    return sumPerClassifier

def preProcsumPerMetricEnsem(factors):
    sumPerClassifier = []
    loopThroughMetrics = PreprocessingMetricsEnsem()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    loopThroughMetrics.loc[:, 'log_loss'] = 1 - loopThroughMetrics.loc[:, 'log_loss']
    for row in loopThroughMetrics.iterrows():
        rowSum = 0
        name, values = row
        for loop, elements in enumerate(values):
            rowSum = elements*factors[loop] + rowSum
        if sum(factors) == 0:
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
    metricsPerModelColl.append(loopThroughMetrics['geometric_mean_score_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_precision_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_recall_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_f1_macro'])
    metricsPerModelColl.append(loopThroughMetrics['matthews_corrcoef'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_roc_auc_ovo'])
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

def preProcMetricsAllAndSelEnsem():
    loopThroughMetrics = PreprocessingMetricsEnsem()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    global factors
    metricsPerModelColl = []
    metricsPerModelColl.append(loopThroughMetrics['mean_test_accuracy'])
    metricsPerModelColl.append(loopThroughMetrics['geometric_mean_score_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_precision_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_recall_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_f1_macro'])
    metricsPerModelColl.append(loopThroughMetrics['matthews_corrcoef'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_roc_auc_ovo'])
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

    global ModelSpaceMDS
    global ModelSpaceTSNE
    global allParametersPerformancePerModel
    global EnsembleActive
    global ModelsIDs
    global keySend
    global metricsPerModel
    global factors

    if (len(EnsembleActive) == 0):
        XModels = PreprocessingMetrics()
        parametersGen = PreprocessingParam()
        PredictionProbSel = PreprocessingPred()
        ModelsIDs = PreprocessingIDs()
        sumPerClassifier = preProcsumPerMetric(factors)
        metricsPerModel = preProcMetricsAllAndSel()
    else:
        XModels = PreprocessingMetricsEnsem()
        parametersGen = PreprocessingParamEnsem()
        PredictionProbSel = PreprocessingPredEnsemble()
        ModelsIDs = EnsembleActive
        modelsIdsCuts = EnsembleIDs()
        sumPerClassifier = preProcsumPerMetricEnsem(factors)
        metricsPerModel = preProcMetricsAllAndSelEnsem()
        EnsembleModel(modelsIdsCuts, keySend)
        keySend=1

    XModels = XModels.fillna(0)
    dropMetrics = []
    for index, element in enumerate(factors):
        if (element == 0):
            dropMetrics.append(index)
    
    XModels.drop(XModels.columns[dropMetrics], axis=1, inplace=True)

    ModelSpaceMDS = FunMDS(XModels)
    ModelSpaceTSNE = FunTsne(XModels)
    ModelSpaceTSNE = ModelSpaceTSNE.tolist()
    ModelSpaceUMAP = FunUMAP(XModels)

    returnResults(ModelSpaceMDS,ModelSpaceTSNE,ModelSpaceUMAP,parametersGen,sumPerClassifier,PredictionProbSel)

def EnsembleModel (Models, keyRetrieved):

    global XDataTest, yDataTest
    global scores
    global previousState
    global crossValidation
    global keyData
    scores = []

    global all_classifiersSelection  
    all_classifiersSelection = []

    global all_classifiers

    global XData
    global yData
    global sclf

    global randomSearchVar
    greater = randomSearchVar*5

    global stage1addKNN
    global stage1addLR
    global stage1addMLP
    global stage1addRF
    global stage1addGradB
    global stageTotalReached

    global numberIDKNNGlob
    global numberIDLRGlob
    global numberIDMLPGlob
    global numberIDRFGlob
    global numberIDGradBGlob
    all_classifiers = []
    columnsInit = []
    columnsInit = [XData.columns.get_loc(c) for c in XData.columns if c in XData]

    temp = allParametersPerformancePerModel[1]

    temp = temp['params']
    temp = {int(k):v for k,v in temp.items()}
    tempDic = {    
        'params': temp
    }
    dfParamKNN = pd.DataFrame.from_dict(tempDic)
    dfParamKNNFilt = dfParamKNN.iloc[:,0]
    for eachelem in numberIDKNNGlob:
        if (eachelem >= stageTotalReached):
            arg = dfParamKNNFilt[eachelem-addKNN]
        elif (eachelem >= greater):
            arg = dfParamKNNFilt[eachelem-stage1addKNN]
        else:
            arg = dfParamKNNFilt[eachelem-KNNModelsCount]
        all_classifiers.append(make_pipeline(ColumnSelector(cols=columnsInit), KNeighborsClassifier().set_params(**arg)))

    temp = allParametersPerformancePerModel[5]
    temp = temp['params']
    temp = {int(k):v for k,v in temp.items()}
    tempDic = {    
        'params': temp
    }
    print(numberIDLRGlob)
    dfParamLR = pd.DataFrame.from_dict(tempDic)
    dfParamLRFilt = dfParamLR.iloc[:,0]
    print(dfParamLRFilt)
    print(addLR)
    print(stage1addLR)
    for eachelem in numberIDLRGlob:
        if (eachelem >= stageTotalReached):
            print('mpike1')
            print(eachelem-addLR)
            arg = dfParamLRFilt[eachelem-addLR]
        elif (eachelem >= greater):
            print('mpike2')
            arg = dfParamLRFilt[eachelem-stage1addLR]
        else:
            arg = dfParamLRFilt[eachelem-LRModelsCount]
            print('mpike3')
        print(arg)
        all_classifiers.append(make_pipeline(ColumnSelector(cols=columnsInit), LogisticRegression(random_state=RANDOM_SEED).set_params(**arg)))

    temp = allParametersPerformancePerModel[9]
    temp = temp['params']
    temp = {int(k):v for k,v in temp.items()}
    tempDic = {    
        'params': temp
    }
    dfParamMLP = pd.DataFrame.from_dict(tempDic)
    dfParamMLPFilt = dfParamMLP.iloc[:,0]
    for eachelem in numberIDMLPGlob:
        if (eachelem >= stageTotalReached):
            arg = dfParamMLPFilt[eachelem-addMLP]
        elif (eachelem >= greater):
            arg = dfParamMLPFilt[eachelem-stage1addMLP]
        else:
            arg = dfParamMLPFilt[eachelem-MLPModelsCount]
        all_classifiers.append(make_pipeline(ColumnSelector(cols=columnsInit), MLPClassifier(random_state=RANDOM_SEED).set_params(**arg)))

    temp = allParametersPerformancePerModel[13]
    temp = temp['params']
    temp = {int(k):v for k,v in temp.items()}
    tempDic = {    
        'params': temp
    } 
    dfParamRF = pd.DataFrame.from_dict(tempDic)
    dfParamRFFilt = dfParamRF.iloc[:,0]

    for eachelem in numberIDRFGlob:
        if (eachelem >= stageTotalReached):
            arg = dfParamRFFilt[eachelem-addRF]
        elif (eachelem >= greater):
            arg = dfParamRFFilt[eachelem-stage1addRF]
        else:
            arg = dfParamRFFilt[eachelem-RFModelsCount]
        all_classifiers.append(make_pipeline(ColumnSelector(cols=columnsInit), RandomForestClassifier(random_state=RANDOM_SEED).set_params(**arg)))

    temp = allParametersPerformancePerModel[17]
    temp = temp['params']
    temp = {int(k):v for k,v in temp.items()}
    tempDic = {    
        'params': temp
    }
    dfParamGradB = pd.DataFrame.from_dict(tempDic)
    dfParamGradBFilt = dfParamGradB.iloc[:,0]
    for eachelem in numberIDGradBGlob:
        if (eachelem >= stageTotalReached):
            arg = dfParamGradBFilt[eachelem-addGradB]
        elif (eachelem >= greater):
            arg = dfParamGradBFilt[eachelem-stage1addGradB]
        else:
            arg = dfParamGradBFilt[eachelem-GradBModelsCount]
        all_classifiers.append(make_pipeline(ColumnSelector(cols=columnsInit), GradientBoostingClassifier(random_state=RANDOM_SEED).set_params(**arg)))

    global sclf 
    sclf = 0
    print(all_classifiers)
    sclf = EnsembleVoteClassifier(clfs=all_classifiers,
                        voting='soft')

    global PerClassResultsClass0
    PerClassResultsClass0 = []
    global PerClassResultsClass1
    PerClassResultsClass1 = []

    nested_score = model_selection.cross_val_score(sclf, X=XData, y=yData, cv=crossValidation, scoring=make_scorer(classification_report_with_accuracy_score))
    PerClassResultsClass0Con = pd.concat(PerClassResultsClass0, axis=1, sort=False)
    PerClassResultsClass1Con = pd.concat(PerClassResultsClass1, axis=1, sort=False)
    averageClass0 = PerClassResultsClass0Con.mean(axis=1)
    averageClass1 = PerClassResultsClass1Con.mean(axis=1)
    y_pred = cross_val_predict(sclf, XData, yData, cv=crossValidation)
    conf_mat = confusion_matrix(yData, y_pred)
    cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    cm.diagonal()

    if (keyRetrieved == 0):
        scores.append(cm[0][0])
        scores.append(cm[1][1])
        scores.append(cm[0][0])
        scores.append(cm[1][1])
        scores.append(averageClass0.precision)
        scores.append(averageClass1.precision)
        scores.append(averageClass0.precision)
        scores.append(averageClass1.precision)
        scores.append(averageClass0.recall)
        scores.append(averageClass1.recall)
        scores.append(averageClass0.recall)
        scores.append(averageClass1.recall)
        scores.append(averageClass0['f1-score'])
        scores.append(averageClass1['f1-score'])
        scores.append(averageClass0['f1-score'])
        scores.append(averageClass1['f1-score'])
        previousState.append(scores[0])
        previousState.append(scores[1])
        previousState.append(scores[4])
        previousState.append(scores[5])
        previousState.append(scores[8])
        previousState.append(scores[9])
        previousState.append(scores[12])
        previousState.append(scores[13])
    else:
        scores.append(cm[0][0])
        scores.append(cm[1][1])
        if (cm[0][0] > previousState[0]):
            scores.append(cm[0][0])
            previousState[0] = cm[0][0]
        else:
            scores.append(previousState[0])
        if (cm[1][1] > previousState[1]):
            scores.append(cm[1][1])
            previousState[1] = cm[1][1]
        else:
            scores.append(previousState[1])
        scores.append(averageClass0.precision)
        scores.append(averageClass1.precision)
        if (averageClass0.precision > previousState[2]):
            scores.append(averageClass0.precision)
            previousState[2] = averageClass0.precision
        else:
            scores.append(previousState[2])
        if (averageClass1.precision > previousState[3]):
            scores.append(averageClass1.precision)
            previousState[3] = averageClass1.precision
        else:
            scores.append(previousState[3])
        scores.append(averageClass0.recall)
        scores.append(averageClass1.recall)
        if (averageClass0.recall > previousState[4]):
            scores.append(averageClass0.recall)
            previousState[4] = averageClass0.recall
        else:
            scores.append(previousState[4])
        if (averageClass1.recall > previousState[5]):
            scores.append(averageClass1.recall)
            previousState[5] = averageClass1.recall
        else:
            scores.append(previousState[5])
        scores.append(averageClass0['f1-score'])
        scores.append(averageClass1['f1-score'])
        if (averageClass0['f1-score'] > previousState[6]):
            scores.append(averageClass0['f1-score'])
            previousState[6] = averageClass0['f1-score']
        else:
            scores.append(previousState[6])
        if (averageClass1['f1-score'] > previousState[7]):
            scores.append(averageClass1['f1-score'])
            previousState[7] = averageClass1['f1-score']
        else:
            scores.append(previousState[7])

    return 'Okay'

# Sending the final results to be visualized as a line plot
@app.route('/data/SendFinalResultsBacktoVisualize', methods=["GET", "POST"])
def SendToPlotFinalResults():
    global scores
    response = {    
        'FinalResults': scores
    }
    return jsonify(response)

def classification_report_with_accuracy_score(y_true, y_pred):
    global PerClassResultsClass0
    global PerClassResultsClass1
    PerClassResultsLocal = pd.DataFrame.from_dict(classification_report(y_true, y_pred, output_dict=True))
    Filter_PerClassResultsLocal0 = PerClassResultsLocal['0']
    Filter_PerClassResultsLocal0 = Filter_PerClassResultsLocal0[:-1]
    Filter_PerClassResultsLocal1 = PerClassResultsLocal['1']
    Filter_PerClassResultsLocal1 = Filter_PerClassResultsLocal1[:-1]
    PerClassResultsClass0.append(Filter_PerClassResultsLocal0)
    PerClassResultsClass1.append(Filter_PerClassResultsLocal1)
    return accuracy_score(y_true, y_pred) # return accuracy score

def returnResults(ModelSpaceMDS,ModelSpaceTSNE,ModelSpaceUMAP,parametersGen,sumPerClassifier,PredictionProbSel):

    global Results
    global AllTargets
    global names_labels
    global EnsembleActive
    global ModelsIDs
    global metricsPerModel
    global yDataSorted
    global storeClass0
    global storeClass1

    if(storeClass0 > 169 | storeClass1 > 169):
        mode = 1
    else:
        mode = 0

    Results = []

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
    Results.append(json.dumps(names_labels))
    Results.append(json.dumps(yDataSorted))
    Results.append(json.dumps(mode))

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
    global CurStage

    EnsembleActive = request.get_data().decode('utf8').replace("'", '"')
    EnsembleActive = json.loads(EnsembleActive)

    EnsembleActive = EnsembleActive['StoreEnsemble']

    setMaxLoopValue = request.get_data().decode('utf8').replace("'", '"')
    setMaxLoopValue = json.loads(setMaxLoopValue)

    setMaxLoopValue = setMaxLoopValue['loopNumber']

    CurStage = request.get_data().decode('utf8').replace("'", '"')
    CurStage = json.loads(CurStage)

    CurStage = CurStage['Stage']

    if (CurStage == 1):
        InitializeFirstStageCM(RemainingIds, setMaxLoopValue)
    elif (CurStage == 2):
        InitializeSecondStageCM(RemainingIds, setMaxLoopValue)
    else:
        RemoveSelected(RemainingIds)
    return 'Okay'

def RemoveSelected(RemainingIds):
    global allParametersPerfCrossMutr

    for loop in range(20): 
        indexes = []
        for i, val in enumerate(allParametersPerfCrossMutr[loop*4]): 
            if (val not in RemainingIds):
                indexes.append(i)
        for index in sorted(indexes, reverse=True):
            del allParametersPerfCrossMutr[loop*4][index]
        allParametersPerfCrossMutr[loop*4+1].drop(allParametersPerfCrossMutr[loop*4+1].index[indexes], inplace=True)
        allParametersPerfCrossMutr[loop*4+2].drop(allParametersPerfCrossMutr[loop*4+2].index[indexes], inplace=True)
        allParametersPerfCrossMutr[loop*4+3].drop(allParametersPerfCrossMutr[loop*4+3].index[indexes], inplace=True)

    return 'Okay'

def InitializeSecondStageCM (RemainingIds, setMaxLoopValue):
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

    global randomSearchVar
    greater = randomSearchVar*5

    KNNIDsC = list(filter(lambda k: 'KNNC' in k, RemainingIds))
    LRIDsC = list(filter(lambda k: 'LRC' in k, RemainingIds))
    MLPIDsC = list(filter(lambda k: 'MLPC' in k, RemainingIds))
    RFIDsC = list(filter(lambda k: 'RFC' in k, RemainingIds))
    GradBIDsC = list(filter(lambda k: 'GradBC' in k, RemainingIds))
    KNNIDsM = list(filter(lambda k: 'KNNM' in k, RemainingIds))
    LRIDsM = list(filter(lambda k: 'LRM' in k, RemainingIds))
    MLPIDsM = list(filter(lambda k: 'MLPM' in k, RemainingIds))
    RFIDsM = list(filter(lambda k: 'RFM' in k, RemainingIds))
    GradBIDsM = list(filter(lambda k: 'GradBM' in k, RemainingIds))

    countKNN = 0
    countLR = 0
    countMLP = 0
    countRF = 0
    countGradB = 0
    paramAllAlgs = PreprocessingParam()

    KNNIntIndex = []
    LRIntIndex = []
    MLPIntIndex = []
    RFIntIndex = []
    GradBIntIndex = []
    
    localCrossMutr = []
    allParametersPerfCrossMutrKNNCC = []
    for dr in KNNIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
        else:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countKNN < setMaxLoopValue[40]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNNCC', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[40]

    for loop in range(setMaxLoopValue[40] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrKNNCC.append(localCrossMutr[0])
    allParametersPerfCrossMutrKNNCC.append(localCrossMutr[1])
    allParametersPerfCrossMutrKNNCC.append(localCrossMutr[2])
    allParametersPerfCrossMutrKNNCC.append(localCrossMutr[3])
    
    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrKNNCC

    countKNN = 0
    KNNIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrKNNCM = []
    for dr in KNNIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
        else:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countKNN < setMaxLoopValue[34]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNNCM', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[34]

    for loop in range(setMaxLoopValue[34] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrKNNCM.append(localCrossMutr[0])
    allParametersPerfCrossMutrKNNCM.append(localCrossMutr[1])
    allParametersPerfCrossMutrKNNCM.append(localCrossMutr[2])
    allParametersPerfCrossMutrKNNCM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrKNNCM

    countKNN = 0
    KNNIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrKNNMC = []
    for dr in KNNIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
        else:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countKNN < setMaxLoopValue[28]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNNMC', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[28]

    for loop in range(setMaxLoopValue[28] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrKNNMC.append(localCrossMutr[0])
    allParametersPerfCrossMutrKNNMC.append(localCrossMutr[1])
    allParametersPerfCrossMutrKNNMC.append(localCrossMutr[2])
    allParametersPerfCrossMutrKNNMC.append(localCrossMutr[3])
    
    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrKNNMC

    countKNN = 0
    KNNIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrKNNMM = []
    for dr in KNNIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
        else:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countKNN < setMaxLoopValue[22]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNNMM', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[22]

    for loop in range(setMaxLoopValue[22] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrKNNMM.append(localCrossMutr[0])
    allParametersPerfCrossMutrKNNMM.append(localCrossMutr[1])
    allParametersPerfCrossMutrKNNMM.append(localCrossMutr[2])
    allParametersPerfCrossMutrKNNMM.append(localCrossMutr[3])
    
    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrKNNMM

    localCrossMutr.clear()
    allParametersPerfCrossMutrLRCC = []
    for dr in LRIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
        else:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countLR < setMaxLoopValue[39]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LRCC', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[39]

    for loop in range(setMaxLoopValue[39] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrLRCC.append(localCrossMutr[0])
    allParametersPerfCrossMutrLRCC.append(localCrossMutr[1])
    allParametersPerfCrossMutrLRCC.append(localCrossMutr[2])
    allParametersPerfCrossMutrLRCC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrLRCC

    countLR = 0
    LRIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrLRCM = []
    for dr in LRIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
        else:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countLR < setMaxLoopValue[33]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LRCM', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[33]

    for loop in range(setMaxLoopValue[33] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrLRCM.append(localCrossMutr[0])
    allParametersPerfCrossMutrLRCM.append(localCrossMutr[1])
    allParametersPerfCrossMutrLRCM.append(localCrossMutr[2])
    allParametersPerfCrossMutrLRCM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrLRCM
    
    countLR = 0
    LRIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrLRMC = []
    for dr in LRIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
        else:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countLR < setMaxLoopValue[27]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LRMC', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[27]

    for loop in range(setMaxLoopValue[27] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrLRMC.append(localCrossMutr[0])
    allParametersPerfCrossMutrLRMC.append(localCrossMutr[1])
    allParametersPerfCrossMutrLRMC.append(localCrossMutr[2])
    allParametersPerfCrossMutrLRMC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrLRMC

    countLR = 0
    LRIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrLRMM = []
    for dr in LRIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
        else:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countLR < setMaxLoopValue[21]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LRMM', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[21]

    for loop in range(setMaxLoopValue[21] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrLRMM.append(localCrossMutr[0])
    allParametersPerfCrossMutrLRMM.append(localCrossMutr[1])
    allParametersPerfCrossMutrLRMM.append(localCrossMutr[2])
    allParametersPerfCrossMutrLRMM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrLRMM

    localCrossMutr.clear()
    allParametersPerfCrossMutrMLPCC = []
    for dr in MLPIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
        else:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countMLP < setMaxLoopValue[38]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLPCC', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[38]

    for loop in range(setMaxLoopValue[38] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrMLPCC.append(localCrossMutr[0])
    allParametersPerfCrossMutrMLPCC.append(localCrossMutr[1])
    allParametersPerfCrossMutrMLPCC.append(localCrossMutr[2])
    allParametersPerfCrossMutrMLPCC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrMLPCC

    countMLP = 0
    MLPIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrMLPCM = []
    for dr in MLPIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
        else:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countMLP < setMaxLoopValue[32]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLPCM', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[32]

    for loop in range(setMaxLoopValue[32] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrMLPCM.append(localCrossMutr[0])
    allParametersPerfCrossMutrMLPCM.append(localCrossMutr[1])
    allParametersPerfCrossMutrMLPCM.append(localCrossMutr[2])
    allParametersPerfCrossMutrMLPCM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrMLPCM

    countMLP = 0
    MLPIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrMLPMC = []
    for dr in MLPIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
        else:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countMLP < setMaxLoopValue[26]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLPMC', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[26]

    for loop in range(setMaxLoopValue[26] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrMLPMC.append(localCrossMutr[0])
    allParametersPerfCrossMutrMLPMC.append(localCrossMutr[1])
    allParametersPerfCrossMutrMLPMC.append(localCrossMutr[2])
    allParametersPerfCrossMutrMLPMC.append(localCrossMutr[3])
    
    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrMLPMC

    countMLP = 0
    MLPIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrMLPMM = []
    for dr in MLPIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
        else:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countMLP < setMaxLoopValue[20]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLPMM', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[20]

    for loop in range(setMaxLoopValue[20] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrMLPMM.append(localCrossMutr[0])
    allParametersPerfCrossMutrMLPMM.append(localCrossMutr[1])
    allParametersPerfCrossMutrMLPMM.append(localCrossMutr[2])
    allParametersPerfCrossMutrMLPMM.append(localCrossMutr[3])
    
    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrMLPMM

    localCrossMutr.clear()
    allParametersPerfCrossMutrRFCC = []
    for dr in RFIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
        else:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countRF < setMaxLoopValue[37]:

        RFPickPair = random.sample(RFIntIndex,2)

        pairDF = paramAllAlgs.iloc[RFPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['max_depth'] == crossoverDF['max_depth'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'max_depth': [crossoverDF['max_depth'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RFCC', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[37]

    for loop in range(setMaxLoopValue[37] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrRFCC.append(localCrossMutr[0])
    allParametersPerfCrossMutrRFCC.append(localCrossMutr[1])
    allParametersPerfCrossMutrRFCC.append(localCrossMutr[2])
    allParametersPerfCrossMutrRFCC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrRFCC

    countRF = 0
    RFIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrRFCM = []
    for dr in RFIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
        else:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))

    while countRF < setMaxLoopValue[31]:

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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['max_depth'] == crossoverDF['max_depth'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'max_depth': [crossoverDF['max_depth'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RFCM', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[31]

    for loop in range(setMaxLoopValue[31] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrRFCM.append(localCrossMutr[0])
    allParametersPerfCrossMutrRFCM.append(localCrossMutr[1])
    allParametersPerfCrossMutrRFCM.append(localCrossMutr[2])
    allParametersPerfCrossMutrRFCM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrRFCM

    countRF = 0
    RFIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrRFMC = []
    while countRF < setMaxLoopValue[25]:
        for dr in RFIDsM:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
            else:
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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['max_depth'] == crossoverDF['max_depth'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'max_depth': [crossoverDF['max_depth'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RFMC', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[25]

    for loop in range(setMaxLoopValue[25] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrRFMC.append(localCrossMutr[0])
    allParametersPerfCrossMutrRFMC.append(localCrossMutr[1])
    allParametersPerfCrossMutrRFMC.append(localCrossMutr[2])
    allParametersPerfCrossMutrRFMC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrRFMC

    countRF = 0
    RFIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrRFMM = []
    for dr in RFIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
        else:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countRF < setMaxLoopValue[19]:

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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['max_depth'] == crossoverDF['max_depth'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'max_depth': [crossoverDF['max_depth'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RFMM', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[19]

    for loop in range(setMaxLoopValue[19] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrRFMM.append(localCrossMutr[0])
    allParametersPerfCrossMutrRFMM.append(localCrossMutr[1])
    allParametersPerfCrossMutrRFMM.append(localCrossMutr[2])
    allParametersPerfCrossMutrRFMM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrRFMM

    localCrossMutr.clear()
    allParametersPerfCrossMutrGradBCC = []

    for dr in GradBIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
        else:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))

    while countGradB < setMaxLoopValue[36]:

        GradBPickPair = random.sample(GradBIntIndex,2)
        
        pairDF = paramAllAlgs.iloc[GradBPickPair]

        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['loss'] == crossoverDF['loss'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['subsample'] == crossoverDF['subsample'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'loss': [crossoverDF['loss'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'subsample': [crossoverDF['subsample'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradBCC', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[36]

    for loop in range(setMaxLoopValue[36] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrGradBCC.append(localCrossMutr[0])
    allParametersPerfCrossMutrGradBCC.append(localCrossMutr[1])
    allParametersPerfCrossMutrGradBCC.append(localCrossMutr[2])
    allParametersPerfCrossMutrGradBCC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrGradBCC

    countGradB = 0
    GradBIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrGradBCM = []
    for dr in GradBIDsC:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            GradBIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
        else:
            GradBIndex.append(int(re.findall('\d+', dr)[0]))

    while countGradB < setMaxLoopValue[30]:

        GradBPickPair = random.sample(GradBIndex,1)
        
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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['loss'] == crossoverDF['loss'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['subsample'] == crossoverDF['subsample'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'loss': [crossoverDF['loss'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'subsample': [crossoverDF['subsample'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradBCM', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[30]

    for loop in range(setMaxLoopValue[30] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)

    allParametersPerfCrossMutrGradBCM.append(localCrossMutr[0])
    allParametersPerfCrossMutrGradBCM.append(localCrossMutr[1])
    allParametersPerfCrossMutrGradBCM.append(localCrossMutr[2])
    allParametersPerfCrossMutrGradBCM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrGradBCM

    countGradB = 0
    GradBIntIndex = []
    localCrossMutr.clear()
    allParametersPerfCrossMutrGradBMC = []
    for dr in GradBIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
        else:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))

    while countGradB < setMaxLoopValue[24]:

        GradBPickPair = random.sample(GradBIntIndex,2)

        pairDF = paramAllAlgs.iloc[GradBPickPair]

        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['loss'] == crossoverDF['loss'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['subsample'] == crossoverDF['subsample'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'loss': [crossoverDF['loss'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'subsample': [crossoverDF['subsample'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradBMC', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[24]

    for loop in range(setMaxLoopValue[24] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrGradBMC.append(localCrossMutr[0])
    allParametersPerfCrossMutrGradBMC.append(localCrossMutr[1])
    allParametersPerfCrossMutrGradBMC.append(localCrossMutr[2])
    allParametersPerfCrossMutrGradBMC.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrGradBMC

    countGradB = 0
    GradBIntIndex = [] 
    localCrossMutr.clear()
    allParametersPerfCrossMutrGradBMM = []
    for dr in GradBIDsM:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
        else:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countGradB < setMaxLoopValue[18]:

        GradBPickPair = random.sample(GradBIntIndex,1)

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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['loss'] == crossoverDF['loss'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['subsample'] == crossoverDF['subsample'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'loss': [crossoverDF['loss'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'subsample': [crossoverDF['subsample'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradBMM', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[18]

    for loop in range(setMaxLoopValue[18] - 1):
        localCrossMutr[0] = localCrossMutr[0] + localCrossMutr[(loop+1)*4]
        localCrossMutr[1] = pd.concat([localCrossMutr[1], localCrossMutr[(loop+1)*4+1]], ignore_index=True)
        localCrossMutr[2] = pd.concat([localCrossMutr[2], localCrossMutr[(loop+1)*4+2]], ignore_index=True)
        localCrossMutr[3] = pd.concat([localCrossMutr[3], localCrossMutr[(loop+1)*4+3]], ignore_index=True)
    
    allParametersPerfCrossMutrGradBMM.append(localCrossMutr[0])
    allParametersPerfCrossMutrGradBMM.append(localCrossMutr[1])
    allParametersPerfCrossMutrGradBMM.append(localCrossMutr[2])
    allParametersPerfCrossMutrGradBMM.append(localCrossMutr[3])

    HistoryPreservation = HistoryPreservation + allParametersPerfCrossMutrGradBMM

    localCrossMutr.clear()

    global allParametersPerformancePerModelEnsem

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrKNNCC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrKNNCM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrKNNCC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrKNNCM[2]], ignore_index=True)

    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrKNNCC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrKNNCM[3]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrLRCC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrLRCM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrLRCC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrLRCM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrLRCC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrLRCM[3]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrMLPCC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrMLPCM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrMLPCC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrMLPCM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrMLPCC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrMLPCM[3]], ignore_index=True)

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrRFCC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrRFCM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrRFCC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrRFCM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrRFCC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrRFCM[3]], ignore_index=True)

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrGradBCC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrGradBCM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrGradBCC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrGradBCM[2]], ignore_index=True)

    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrGradBCC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrGradBCM[3]], ignore_index=True)
    
    # MUTATION
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrKNNMC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrKNNMM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrKNNMC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrKNNMM[2]], ignore_index=True)

    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrKNNMC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrKNNMM[3]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrLRMC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrLRMM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrLRMC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrLRMM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrLRMC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrLRMM[3]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrMLPMC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrMLPMM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrMLPMC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrMLPMM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrMLPMC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrMLPMM[3]], ignore_index=True)

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrRFMC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrRFMM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrRFMC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrRFMM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrRFMC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrRFMM[3]], ignore_index=True)

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrGradBMC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrGradBMM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrGradBMC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrGradBMM[2]], ignore_index=True)

    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrGradBMC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrGradBMM[3]], ignore_index=True)

    allParametersPerfCrossMutr = allParametersPerfCrossMutrKNNCC + allParametersPerfCrossMutrKNNCM + allParametersPerfCrossMutrLRCC + allParametersPerfCrossMutrLRCM + allParametersPerfCrossMutrMLPCC + allParametersPerfCrossMutrMLPCM + allParametersPerfCrossMutrRFCC + allParametersPerfCrossMutrRFCM + allParametersPerfCrossMutrGradBCC + allParametersPerfCrossMutrGradBCM + allParametersPerfCrossMutrKNNMC + allParametersPerfCrossMutrKNNMM + allParametersPerfCrossMutrLRMC + allParametersPerfCrossMutrLRMM + allParametersPerfCrossMutrMLPMC + allParametersPerfCrossMutrMLPMM + allParametersPerfCrossMutrRFMC + allParametersPerfCrossMutrRFMM + allParametersPerfCrossMutrGradBMC + allParametersPerfCrossMutrGradBMM
    allParametersPerformancePerModel[0] = allParametersPerformancePerModel[0] + allParametersPerfCrossMutrKNNCC[0] + allParametersPerfCrossMutrKNNCM[0]

    allParametersPerformancePerModel[1] = pd.concat([allParametersPerformancePerModel[1], allParametersPerfCrossMutrKNNCC[1]], ignore_index=True)
    allParametersPerformancePerModel[1] = pd.concat([allParametersPerformancePerModel[1], allParametersPerfCrossMutrKNNCM[1]], ignore_index=True)
    allParametersPerformancePerModel[2] = pd.concat([allParametersPerformancePerModel[2], allParametersPerfCrossMutrKNNCC[2]], ignore_index=True)
    allParametersPerformancePerModel[2] = pd.concat([allParametersPerformancePerModel[2], allParametersPerfCrossMutrKNNCM[2]], ignore_index=True)

    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNCC[3]], ignore_index=True)
    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNCM[3]], ignore_index=True)
    
    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrLRCC[0] + allParametersPerfCrossMutrLRCM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRCC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRCM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRCC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRCM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRCC[3]], ignore_index=True)
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRCM[3]], ignore_index=True)

    allParametersPerformancePerModel[8] = allParametersPerformancePerModel[8] + allParametersPerfCrossMutrMLPCC[0] + allParametersPerfCrossMutrMLPCM[0]
    
    allParametersPerformancePerModel[9] = pd.concat([allParametersPerformancePerModel[9], allParametersPerfCrossMutrMLPCC[1]], ignore_index=True)
    allParametersPerformancePerModel[9] = pd.concat([allParametersPerformancePerModel[9], allParametersPerfCrossMutrMLPCM[1]], ignore_index=True)
    allParametersPerformancePerModel[10] = pd.concat([allParametersPerformancePerModel[10], allParametersPerfCrossMutrMLPCC[2]], ignore_index=True)
    allParametersPerformancePerModel[10] = pd.concat([allParametersPerformancePerModel[10], allParametersPerfCrossMutrMLPCM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPCC[3]], ignore_index=True)
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPCM[3]], ignore_index=True)

    allParametersPerformancePerModel[12] = allParametersPerformancePerModel[12] + allParametersPerfCrossMutrRFCC[0] + allParametersPerfCrossMutrRFCM[0]
    
    allParametersPerformancePerModel[13] = pd.concat([allParametersPerformancePerModel[13], allParametersPerfCrossMutrRFCC[1]], ignore_index=True)
    allParametersPerformancePerModel[13] = pd.concat([allParametersPerformancePerModel[13], allParametersPerfCrossMutrRFCM[1]], ignore_index=True)
    allParametersPerformancePerModel[14] = pd.concat([allParametersPerformancePerModel[14], allParametersPerfCrossMutrRFCC[2]], ignore_index=True)
    allParametersPerformancePerModel[14] = pd.concat([allParametersPerformancePerModel[14], allParametersPerfCrossMutrRFCM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFCC[3]], ignore_index=True)
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFCM[3]], ignore_index=True)

    allParametersPerformancePerModel[16] = allParametersPerformancePerModel[16] + allParametersPerfCrossMutrGradBCC[0] + allParametersPerfCrossMutrGradBCM[0]
    
    allParametersPerformancePerModel[17] = pd.concat([allParametersPerformancePerModel[17], allParametersPerfCrossMutrGradBCC[1]], ignore_index=True)
    allParametersPerformancePerModel[17] = pd.concat([allParametersPerformancePerModel[17], allParametersPerfCrossMutrGradBCM[1]], ignore_index=True)
    allParametersPerformancePerModel[18] = pd.concat([allParametersPerformancePerModel[18], allParametersPerfCrossMutrGradBCC[2]], ignore_index=True)
    allParametersPerformancePerModel[18] = pd.concat([allParametersPerformancePerModel[18], allParametersPerfCrossMutrGradBCM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBCC[3]], ignore_index=True)
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBCM[3]], ignore_index=True)

    allParametersPerformancePerModel[0] = allParametersPerformancePerModel[0] + allParametersPerfCrossMutrKNNMC[0] + allParametersPerfCrossMutrKNNMM[0]

    allParametersPerformancePerModel[1] = pd.concat([allParametersPerformancePerModel[1], allParametersPerfCrossMutrKNNMC[1]], ignore_index=True)
    allParametersPerformancePerModel[1] = pd.concat([allParametersPerformancePerModel[1], allParametersPerfCrossMutrKNNMM[1]], ignore_index=True)
    allParametersPerformancePerModel[2] = pd.concat([allParametersPerformancePerModel[2], allParametersPerfCrossMutrKNNMC[2]], ignore_index=True)
    allParametersPerformancePerModel[2] = pd.concat([allParametersPerformancePerModel[2], allParametersPerfCrossMutrKNNMM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNMC[3]], ignore_index=True)
    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNMM[3]], ignore_index=True)
    
    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrLRMC[0] + allParametersPerfCrossMutrLRMM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRMC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRMM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRMC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRMM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRMC[3]], ignore_index=True)
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRMM[3]], ignore_index=True)

    allParametersPerformancePerModel[8] = allParametersPerformancePerModel[8] + allParametersPerfCrossMutrMLPMC[0] + allParametersPerfCrossMutrMLPMM[0]
    
    allParametersPerformancePerModel[9] = pd.concat([allParametersPerformancePerModel[9], allParametersPerfCrossMutrMLPMC[1]], ignore_index=True)
    allParametersPerformancePerModel[9] = pd.concat([allParametersPerformancePerModel[9], allParametersPerfCrossMutrMLPMM[1]], ignore_index=True)
    allParametersPerformancePerModel[10] = pd.concat([allParametersPerformancePerModel[10], allParametersPerfCrossMutrMLPMC[2]], ignore_index=True)
    allParametersPerformancePerModel[10] = pd.concat([allParametersPerformancePerModel[10], allParametersPerfCrossMutrMLPMM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPMC[3]], ignore_index=True)
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPMM[3]], ignore_index=True)

    allParametersPerformancePerModel[12] = allParametersPerformancePerModel[12] + allParametersPerfCrossMutrRFMC[0] + allParametersPerfCrossMutrRFMM[0]
    
    allParametersPerformancePerModel[13] = pd.concat([allParametersPerformancePerModel[13], allParametersPerfCrossMutrRFMC[1]], ignore_index=True)
    allParametersPerformancePerModel[13] = pd.concat([allParametersPerformancePerModel[13], allParametersPerfCrossMutrRFMM[1]], ignore_index=True)
    allParametersPerformancePerModel[14] = pd.concat([allParametersPerformancePerModel[14], allParametersPerfCrossMutrRFMC[2]], ignore_index=True)
    allParametersPerformancePerModel[14] = pd.concat([allParametersPerformancePerModel[14], allParametersPerfCrossMutrRFMM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFMC[3]], ignore_index=True)
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFMM[3]], ignore_index=True)

    allParametersPerformancePerModel[16] = allParametersPerformancePerModel[16] + allParametersPerfCrossMutrGradBMC[0] + allParametersPerfCrossMutrGradBMM[0]
    
    allParametersPerformancePerModel[17] = pd.concat([allParametersPerformancePerModel[17], allParametersPerfCrossMutrGradBMC[1]], ignore_index=True)
    allParametersPerformancePerModel[17] = pd.concat([allParametersPerformancePerModel[17], allParametersPerfCrossMutrGradBMM[1]], ignore_index=True)
    allParametersPerformancePerModel[18] = pd.concat([allParametersPerformancePerModel[18], allParametersPerfCrossMutrGradBMC[2]], ignore_index=True)
    allParametersPerformancePerModel[18] = pd.concat([allParametersPerformancePerModel[18], allParametersPerfCrossMutrGradBMM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBMC[3]], ignore_index=True)
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBMM[3]], ignore_index=True)

    addKNN = addGradB

    addLR = addKNN + setMaxLoopValue[40] + setMaxLoopValue[34] + setMaxLoopValue[28] + setMaxLoopValue[22]

    addMLP = addLR + setMaxLoopValue[39] + setMaxLoopValue[33] + setMaxLoopValue[27] + setMaxLoopValue[21]

    addRF = addMLP + setMaxLoopValue[38] + setMaxLoopValue[32] + setMaxLoopValue[26] + setMaxLoopValue[20]

    addGradB = addRF + setMaxLoopValue[37] + setMaxLoopValue[31] + setMaxLoopValue[25] + setMaxLoopValue[19]


    return 'Everything Okay'

def InitializeFirstStageCM (RemainingIds, setMaxLoopValue):
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

    global randomSearchVar
    greater = randomSearchVar*5

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
    paramAllAlgs = PreprocessingParam()

    KNNIntIndex = []
    LRIntIndex = []
    MLPIntIndex = []
    RFIntIndex = []
    GradBIntIndex = []
    
    localCrossMutr = []
    allParametersPerfCrossMutrKNNC = []
    for dr in KNNIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
        else:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countKNN < setMaxLoopValue[16]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNNC', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[16]

    for loop in range(setMaxLoopValue[16] - 1):
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
    for dr in KNNIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
        else:
            KNNIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countKNN < setMaxLoopValue[10]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNNM', AlgorithmsIDsEnd)
            countKNN += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[10]

    for loop in range(setMaxLoopValue[10] - 1):
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
    for dr in LRIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
        else:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countLR < setMaxLoopValue[15]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LRC', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[15]

    for loop in range(setMaxLoopValue[15] - 1):
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
    for dr in LRIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
        else:
            LRIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countLR < setMaxLoopValue[9]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LRM', AlgorithmsIDsEnd)
            countLR += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[9]

    for loop in range(setMaxLoopValue[9] - 1):
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
    for dr in MLPIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
        else:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countMLP < setMaxLoopValue[14]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLPC', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[14]

    for loop in range(setMaxLoopValue[14] - 1):
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
    for dr in MLPIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
        else:
            MLPIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countMLP < setMaxLoopValue[8]:

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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLPM', AlgorithmsIDsEnd)
            countMLP += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[8]

    for loop in range(setMaxLoopValue[8] - 1):
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
    for dr in RFIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
        else:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countRF < setMaxLoopValue[13]:

        RFPickPair = random.sample(RFIntIndex,2)

        pairDF = paramAllAlgs.iloc[RFPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['max_depth'] == crossoverDF['max_depth'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'max_depth': [crossoverDF['max_depth'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RFC', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[13]

    for loop in range(setMaxLoopValue[13] - 1):
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
    for dr in RFIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
        else:
            RFIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countRF < setMaxLoopValue[7]:

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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['max_depth'] == crossoverDF['max_depth'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'max_depth': [crossoverDF['max_depth'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RFM', AlgorithmsIDsEnd)
            countRF += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[7]

    for loop in range(setMaxLoopValue[7] - 1):
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
    for dr in GradBIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
        else:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countGradB < setMaxLoopValue[12]:

        GradBPickPair = random.sample(GradBIntIndex,2)

        pairDF = paramAllAlgs.iloc[GradBPickPair]
        crossoverDF = pd.DataFrame()
        for column in pairDF:
            listData = []
            randomZeroOne = random.randint(0, 1)
            valuePerColumn = pairDF[column].iloc[randomZeroOne]
            listData.append(valuePerColumn)
            crossoverDF[column] = listData
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['loss'] == crossoverDF['loss'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['subsample'] == crossoverDF['subsample'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'loss': [crossoverDF['loss'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'subsample': [crossoverDF['subsample'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradBC', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[12]

    for loop in range(setMaxLoopValue[12] - 1):
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
    for dr in GradBIDs:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
        else:
            GradBIntIndex.append(int(re.findall('\d+', dr)[0]))
    while countGradB < setMaxLoopValue[6]:

        GradBPickPair = random.sample(GradBIntIndex,1)
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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['loss'] == crossoverDF['loss'].iloc[0]) & (paramAllAlgs['learning_rate'] == crossoverDF['learning_rate'].iloc[0]) & (paramAllAlgs['subsample'] == crossoverDF['subsample'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'loss': [crossoverDF['loss'].iloc[0]], 'learning_rate': [crossoverDF['learning_rate'].iloc[0]], 'subsample': [crossoverDF['subsample'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countGradB
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradBM', AlgorithmsIDsEnd)
            countGradB += 1
            crossoverDF = pd.DataFrame()

    countAllModels = countAllModels + setMaxLoopValue[6]

    for loop in range(setMaxLoopValue[6] - 1):
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

    global allParametersPerformancePerModelEnsem
    allParametersPerformancePerModelEnsem = allParametersPerformancePerModel.copy()

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrKNNC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrKNNM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrKNNC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrKNNM[2]], ignore_index=True)

    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrKNNC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrKNNM[3]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrLRC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrLRM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrLRC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrLRM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrLRC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrLRM[3]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrMLPC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrMLPM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrMLPC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrMLPM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrMLPC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrMLPM[3]], ignore_index=True)

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrRFC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrRFM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrRFC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrRFM[2]], ignore_index=True)
    
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrRFC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrRFM[3]], ignore_index=True)

    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrGradBC[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[17] = pd.concat([allParametersPerformancePerModelEnsem[17], allParametersPerfCrossMutrGradBM[1]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrGradBC[2]], ignore_index=True)
    allParametersPerformancePerModelEnsem[18] = pd.concat([allParametersPerformancePerModelEnsem[18], allParametersPerfCrossMutrGradBM[2]], ignore_index=True)

    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrGradBC[3]], ignore_index=True)
    allParametersPerformancePerModelEnsem[19] = pd.concat([allParametersPerformancePerModelEnsem[19], allParametersPerfCrossMutrGradBM[3]], ignore_index=True)

    allParametersPerfCrossMutr = allParametersPerfCrossMutrKNNC + allParametersPerfCrossMutrKNNM + allParametersPerfCrossMutrLRC + allParametersPerfCrossMutrLRM + allParametersPerfCrossMutrMLPC + allParametersPerfCrossMutrMLPM + allParametersPerfCrossMutrRFC + allParametersPerfCrossMutrRFM + allParametersPerfCrossMutrGradBC + allParametersPerfCrossMutrGradBM
    allParametersPerformancePerModel[0] = allParametersPerformancePerModel[0] + allParametersPerfCrossMutrKNNC[0] + allParametersPerfCrossMutrKNNM[0]

    allParametersPerformancePerModel[1] = pd.concat([allParametersPerformancePerModel[1], allParametersPerfCrossMutrKNNC[1]], ignore_index=True)
    allParametersPerformancePerModel[1] = pd.concat([allParametersPerformancePerModel[1], allParametersPerfCrossMutrKNNM[1]], ignore_index=True)
    allParametersPerformancePerModel[2] = pd.concat([allParametersPerformancePerModel[2], allParametersPerfCrossMutrKNNC[2]], ignore_index=True)
    allParametersPerformancePerModel[2] = pd.concat([allParametersPerformancePerModel[2], allParametersPerfCrossMutrKNNM[2]], ignore_index=True)

    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNC[3]], ignore_index=True)
    allParametersPerformancePerModel[3] = pd.concat([allParametersPerformancePerModel[3], allParametersPerfCrossMutrKNNM[3]], ignore_index=True)
    
    allParametersPerformancePerModel[4] = allParametersPerformancePerModel[4] + allParametersPerfCrossMutrLRC[0] + allParametersPerfCrossMutrLRM[0]
    
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRC[1]], ignore_index=True)
    allParametersPerformancePerModel[5] = pd.concat([allParametersPerformancePerModel[5], allParametersPerfCrossMutrLRM[1]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRC[2]], ignore_index=True)
    allParametersPerformancePerModel[6] = pd.concat([allParametersPerformancePerModel[6], allParametersPerfCrossMutrLRM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRC[3]], ignore_index=True)
    allParametersPerformancePerModel[7] = pd.concat([allParametersPerformancePerModel[7], allParametersPerfCrossMutrLRM[3]], ignore_index=True)

    allParametersPerformancePerModel[8] = allParametersPerformancePerModel[8] + allParametersPerfCrossMutrMLPC[0] + allParametersPerfCrossMutrMLPM[0]
    
    allParametersPerformancePerModel[9] = pd.concat([allParametersPerformancePerModel[9], allParametersPerfCrossMutrMLPC[1]], ignore_index=True)
    allParametersPerformancePerModel[9] = pd.concat([allParametersPerformancePerModel[9], allParametersPerfCrossMutrMLPM[1]], ignore_index=True)
    allParametersPerformancePerModel[10] = pd.concat([allParametersPerformancePerModel[10], allParametersPerfCrossMutrMLPC[2]], ignore_index=True)
    allParametersPerformancePerModel[10] = pd.concat([allParametersPerformancePerModel[10], allParametersPerfCrossMutrMLPM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPC[3]], ignore_index=True)
    allParametersPerformancePerModel[11] = pd.concat([allParametersPerformancePerModel[11], allParametersPerfCrossMutrMLPM[3]], ignore_index=True)

    allParametersPerformancePerModel[12] = allParametersPerformancePerModel[12] + allParametersPerfCrossMutrRFC[0] + allParametersPerfCrossMutrRFM[0]
    
    allParametersPerformancePerModel[13] = pd.concat([allParametersPerformancePerModel[13], allParametersPerfCrossMutrRFC[1]], ignore_index=True)
    allParametersPerformancePerModel[13] = pd.concat([allParametersPerformancePerModel[13], allParametersPerfCrossMutrRFM[1]], ignore_index=True)
    allParametersPerformancePerModel[14] = pd.concat([allParametersPerformancePerModel[14], allParametersPerfCrossMutrRFC[2]], ignore_index=True)
    allParametersPerformancePerModel[14] = pd.concat([allParametersPerformancePerModel[14], allParametersPerfCrossMutrRFM[2]], ignore_index=True)
    
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFC[3]], ignore_index=True)
    allParametersPerformancePerModel[15] = pd.concat([allParametersPerformancePerModel[15], allParametersPerfCrossMutrRFM[3]], ignore_index=True)

    allParametersPerformancePerModel[16] = allParametersPerformancePerModel[16] + allParametersPerfCrossMutrGradBC[0] + allParametersPerfCrossMutrGradBM[0]

    allParametersPerformancePerModel[17] = pd.concat([allParametersPerformancePerModel[17], allParametersPerfCrossMutrGradBC[1]], ignore_index=True)
    allParametersPerformancePerModel[17] = pd.concat([allParametersPerformancePerModel[17], allParametersPerfCrossMutrGradBM[1]], ignore_index=True)
    allParametersPerformancePerModel[18] = pd.concat([allParametersPerformancePerModel[18], allParametersPerfCrossMutrGradBC[2]], ignore_index=True)
    allParametersPerformancePerModel[18] = pd.concat([allParametersPerformancePerModel[18], allParametersPerfCrossMutrGradBM[2]], ignore_index=True)

    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBC[3]], ignore_index=True)
    allParametersPerformancePerModel[19] = pd.concat([allParametersPerformancePerModel[19], allParametersPerfCrossMutrGradBM[3]], ignore_index=True)

    global stage1addKNN
    global stage1addLR
    global stage1addMLP
    global stage1addRF
    global stage1addGradB
    global stageTotalReached
    global randomSearch

    addKNN = addGradB

    addLR = addKNN + setMaxLoopValue[16] + setMaxLoopValue[10]

    addMLP = addLR + setMaxLoopValue[15] + setMaxLoopValue[9]

    addRF = addMLP + setMaxLoopValue[14] + setMaxLoopValue[8]

    addGradB = addRF + setMaxLoopValue[13] + setMaxLoopValue[7]

    addAllNew = setMaxLoopValue[16] + setMaxLoopValue[10] + setMaxLoopValue[15] + setMaxLoopValue[9] + setMaxLoopValue[14] + setMaxLoopValue[8] + setMaxLoopValue[13] + setMaxLoopValue[7] + setMaxLoopValue[12] + setMaxLoopValue[6]

    stage1addKNN = addKNN
    stage1addLR = addLR
    stage1addMLP = addMLP
    stage1addRF = addRF
    stage1addGradB = addGradB
    stageTotalReached = stageTotalReached + addAllNew

    print(stageTotalReached)

    return 'Everything Okay'

def crossoverMutation(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd):
    print(eachAlgor)
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
    metrics = metrics.filter(['mean_test_accuracy','mean_test_precision_macro','mean_test_recall_macro','mean_test_f1_macro','mean_test_roc_auc_ovo']) 

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

        resultsWeighted.append(geometric_mean_score(yData, yPredict, average='macro'))
        resultsCorrCoef.append(matthews_corrcoef(yData, yPredict))
        resultsLogLoss.append(log_loss(yData, yPredictProb, normalize=True))

    maxLog = max(resultsLogLoss)
    minLog = min(resultsLogLoss)
    for each in resultsLogLoss:
        resultsLogLossFinal.append((each-minLog)/(maxLog-minLog))

    metrics.insert(5,'geometric_mean_score_macro',resultsWeighted)
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

def PreprocessingIDsCMSecond():
    dicKNNCC = allParametersPerfCrossMutr[0]
    dicKNNCM = allParametersPerfCrossMutr[4]
    dicLRCC = allParametersPerfCrossMutr[8]
    dicLRCM = allParametersPerfCrossMutr[12]
    dicMLPCC = allParametersPerfCrossMutr[16]
    dicMLPCM = allParametersPerfCrossMutr[20]
    dicRFCC = allParametersPerfCrossMutr[24]
    dicRFCM = allParametersPerfCrossMutr[28]
    dicGradBCC = allParametersPerfCrossMutr[32]
    dicGradBCM = allParametersPerfCrossMutr[36]
    dicKNNMC = allParametersPerfCrossMutr[40]
    dicKNNMM = allParametersPerfCrossMutr[44]
    dicLRMC = allParametersPerfCrossMutr[48]
    dicLRMM = allParametersPerfCrossMutr[52]
    dicMLPMC = allParametersPerfCrossMutr[56]
    dicMLPMM = allParametersPerfCrossMutr[60]
    dicRFMC = allParametersPerfCrossMutr[64]
    dicRFMM = allParametersPerfCrossMutr[68]
    dicGradBMC = allParametersPerfCrossMutr[72]
    dicGradBMM = allParametersPerfCrossMutr[76]

    df_concatIDs = dicKNNCC + dicKNNCM + dicLRCC + dicLRCM + dicMLPCC + dicMLPCM + dicRFCC + dicRFCM + dicGradBCC + dicGradBCM + dicKNNMC + dicKNNMM + dicLRMC + dicLRMM + dicMLPMC + dicMLPMM + dicRFMC + dicRFMM + dicGradBMC + dicGradBMM
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

def PreprocessingMetricsCMSecond():
    dicKNNCC = allParametersPerfCrossMutr[2]
    dicKNNCM = allParametersPerfCrossMutr[6]
    dicLRCC = allParametersPerfCrossMutr[10]
    dicLRCM = allParametersPerfCrossMutr[14]
    dicMLPCC = allParametersPerfCrossMutr[18]
    dicMLPCM = allParametersPerfCrossMutr[22]
    dicRFCC = allParametersPerfCrossMutr[26]
    dicRFCM = allParametersPerfCrossMutr[30]
    dicGradBCC = allParametersPerfCrossMutr[34]
    dicGradBCM = allParametersPerfCrossMutr[38]
    dicKNNMC = allParametersPerfCrossMutr[42]
    dicKNNMM = allParametersPerfCrossMutr[46]
    dicLRMC = allParametersPerfCrossMutr[50]
    dicLRMM = allParametersPerfCrossMutr[54]
    dicMLPMC = allParametersPerfCrossMutr[58]
    dicMLPMM = allParametersPerfCrossMutr[62]
    dicRFMC = allParametersPerfCrossMutr[66]
    dicRFMM = allParametersPerfCrossMutr[70]
    dicGradBMC = allParametersPerfCrossMutr[74]
    dicGradBMM = allParametersPerfCrossMutr[78]

    dfKNNCC = pd.DataFrame.from_dict(dicKNNCC)
    dfKNNCM = pd.DataFrame.from_dict(dicKNNCM)
    dfLRCC = pd.DataFrame.from_dict(dicLRCC)
    dfLRCM = pd.DataFrame.from_dict(dicLRCM)
    dfMLPCC = pd.DataFrame.from_dict(dicMLPCC)
    dfMLPCM = pd.DataFrame.from_dict(dicMLPCM)
    dfRFCC = pd.DataFrame.from_dict(dicRFCC)
    dfRFCM = pd.DataFrame.from_dict(dicRFCM)
    dfGradBCC = pd.DataFrame.from_dict(dicGradBCC)
    dfGradBCM = pd.DataFrame.from_dict(dicGradBCM)
    dfKNNMC = pd.DataFrame.from_dict(dicKNNMC)
    dfKNNMM = pd.DataFrame.from_dict(dicKNNMM)
    dfLRMC = pd.DataFrame.from_dict(dicLRMC)
    dfLRMM = pd.DataFrame.from_dict(dicLRMM)
    dfMLPMC = pd.DataFrame.from_dict(dicMLPMC)
    dfMLPMM = pd.DataFrame.from_dict(dicMLPMM)
    dfRFMC = pd.DataFrame.from_dict(dicRFMC)
    dfRFMM = pd.DataFrame.from_dict(dicRFMM)
    dfGradBMC = pd.DataFrame.from_dict(dicGradBMC)
    dfGradBMM = pd.DataFrame.from_dict(dicGradBMM)

    df_concatMetrics = pd.concat([dfKNNCC, dfKNNCM, dfLRCC, dfLRCM, dfMLPCC, dfMLPCM, dfRFCC, dfRFCM, dfGradBCC, dfGradBCM, dfKNNMC, dfKNNMM, dfLRMC, dfLRMM, dfMLPMC, dfMLPMM, dfRFMC, dfRFMM, dfGradBMC, dfGradBMM])
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

    global storeClass0
    global storeClass1
    global yDataSorted

    firstElKNN = []
    firstElLR = []
    firstElMLP = []
    firstElRF = []
    firstElGradB = []
    firstElPredAv = []
    lastElKNN = []
    lastElLR = []
    lastElMLP = []
    lastElRF = []
    lastElGradB = []
    lastElPredAv = []
    yDataSortedFirst = []
    yDataSortedLast = []

    for index, item in enumerate(yData):
        if (item == 0):
            if (len(predictionsKNN[index]) != 0):
                firstElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                firstElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                firstElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                firstElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                firstElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                firstElPredAv.append(predictions[index][item]*100)
            yDataSortedFirst.append(item)
        else:
            if (len(predictionsKNN[index]) != 0):
                lastElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                lastElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                lastElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                lastElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                lastElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                lastElPredAv.append(predictions[index][item]*100)
            yDataSortedLast.append(item)

    if (storeClass0 > 169 & storeClass1 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)
        lastElKNN = computeClusters(lastElKNN)
        firstElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass0 < 169 & storeClass1 > 169):
                
        lastElKNN = computeClusters(lastElKNN)
        lastElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass1 < 169 & storeClass0 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)

    else:
        pass

    predictionsKNN = firstElKNN + lastElKNN
    predictionsLR = firstElLR + lastElLR        
    predictionsMLP = firstElMLP + lastElMLP
    predictionsRF = firstElRF + lastElRF
    predictionsGradB = firstElGradB + lastElGradB
    predictions = firstElPredAv + lastElPredAv
    yDataSorted = yDataSortedFirst + yDataSortedLast

    return [predictionsKNN, predictionsLR, predictionsMLP, predictionsRF, predictionsGradB, predictions]

def PreprocessingPredCMSecond():
    dicKNNCC = allParametersPerfCrossMutr[3]
    dicKNNCM = allParametersPerfCrossMutr[7]
    dicLRCC = allParametersPerfCrossMutr[11]
    dicLRCM = allParametersPerfCrossMutr[15]
    dicMLPCC = allParametersPerfCrossMutr[19]
    dicMLPCM = allParametersPerfCrossMutr[23]
    dicRFCC = allParametersPerfCrossMutr[27]
    dicRFCM = allParametersPerfCrossMutr[31]
    dicGradBCC = allParametersPerfCrossMutr[35]
    dicGradBCM = allParametersPerfCrossMutr[39]
    dicKNNMC = allParametersPerfCrossMutr[43]
    dicKNNMM = allParametersPerfCrossMutr[47]
    dicLRMC = allParametersPerfCrossMutr[51]
    dicLRMM = allParametersPerfCrossMutr[55]
    dicMLPMC = allParametersPerfCrossMutr[59]
    dicMLPMM = allParametersPerfCrossMutr[63]
    dicRFMC = allParametersPerfCrossMutr[67]
    dicRFMM = allParametersPerfCrossMutr[71]
    dicGradBMC = allParametersPerfCrossMutr[75]
    dicGradBMM = allParametersPerfCrossMutr[79]

    dfKNNCC = pd.DataFrame.from_dict(dicKNNCC)
    dfKNNCM = pd.DataFrame.from_dict(dicKNNCM)
    dfLRCC = pd.DataFrame.from_dict(dicLRCC)
    dfLRCM = pd.DataFrame.from_dict(dicLRCM)
    dfMLPCC = pd.DataFrame.from_dict(dicMLPCC)
    dfMLPCM = pd.DataFrame.from_dict(dicMLPCM)
    dfRFCC = pd.DataFrame.from_dict(dicRFCC)
    dfRFCM = pd.DataFrame.from_dict(dicRFCM)
    dfGradBCC = pd.DataFrame.from_dict(dicGradBCC)
    dfGradBCM = pd.DataFrame.from_dict(dicGradBCM)
    dfKNNMC = pd.DataFrame.from_dict(dicKNNMC)
    dfKNNMM = pd.DataFrame.from_dict(dicKNNMM)
    dfLRMC = pd.DataFrame.from_dict(dicLRMC)
    dfLRMM = pd.DataFrame.from_dict(dicLRMM)
    dfMLPMC = pd.DataFrame.from_dict(dicMLPMC)
    dfMLPMM = pd.DataFrame.from_dict(dicMLPMM)
    dfRFMC = pd.DataFrame.from_dict(dicRFMC)
    dfRFMM = pd.DataFrame.from_dict(dicRFMM)
    dfGradBMC = pd.DataFrame.from_dict(dicGradBMC)
    dfGradBMM = pd.DataFrame.from_dict(dicGradBMM)

    dfKNN = pd.concat([dfKNNCC, dfKNNCM, dfKNNMC, dfKNNMM])

    dfLR = pd.concat([dfLRCC, dfLRCM, dfLRMC, dfLRMM])

    dfMLP = pd.concat([dfMLPCC, dfMLPCM, dfMLPMC, dfMLPMM])

    dfRF = pd.concat([dfRFCC, dfRFCM, dfRFMC, dfRFMM])

    dfGradB = pd.concat([dfGradBCC, dfGradBCM, dfGradBMC, dfGradBMM])

    df_concatProbs = pd.concat([dfKNNCC, dfKNNCM, dfLRCC, dfLRCM, dfMLPCC, dfMLPCM, dfRFCC, dfRFCM, dfGradBCC, dfGradBCM, dfKNNMC, dfKNNMM, dfLRMC, dfLRMM, dfMLPMC, dfMLPMM, dfRFMC, dfRFMM, dfGradBMC, dfGradBMM])

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

    global storeClass0
    global storeClass1
    global yDataSorted

    firstElKNN = []
    firstElLR = []
    firstElMLP = []
    firstElRF = []
    firstElGradB = []
    firstElPredAv = []
    lastElKNN = []
    lastElLR = []
    lastElMLP = []
    lastElRF = []
    lastElGradB = []
    lastElPredAv = []
    yDataSortedFirst = []
    yDataSortedLast = []

    for index, item in enumerate(yData):
        if (item == 0):
            if (len(predictionsKNN[index]) != 0):
                firstElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                firstElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                firstElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                firstElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                firstElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                firstElPredAv.append(predictions[index][item]*100)
            yDataSortedFirst.append(item)
        else:
            if (len(predictionsKNN[index]) != 0):
                lastElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                lastElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                lastElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                lastElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                lastElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                lastElPredAv.append(predictions[index][item]*100)
            yDataSortedLast.append(item)

    if (storeClass0 > 169 & storeClass1 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)
        lastElKNN = computeClusters(lastElKNN)
        firstElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass0 < 169 & storeClass1 > 169):
                
        lastElKNN = computeClusters(lastElKNN)
        lastElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass1 < 169 & storeClass0 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)

    else:
        pass

    predictionsKNN = firstElKNN + lastElKNN
    predictionsLR = firstElLR + lastElLR        
    predictionsMLP = firstElMLP + lastElMLP
    predictionsRF = firstElRF + lastElRF
    predictionsGradB = firstElGradB + lastElGradB
    predictions = firstElPredAv + lastElPredAv
    yDataSorted = yDataSortedFirst + yDataSortedLast

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

def PreprocessingParamCMSecond():
    dicKNNCC = allParametersPerfCrossMutr[1]
    dicKNNCM = allParametersPerfCrossMutr[5]
    dicLRCC = allParametersPerfCrossMutr[9]
    dicLRCM = allParametersPerfCrossMutr[13]
    dicMLPCC = allParametersPerfCrossMutr[17]
    dicMLPCM = allParametersPerfCrossMutr[21]
    dicRFCC = allParametersPerfCrossMutr[25]
    dicRFCM = allParametersPerfCrossMutr[29]
    dicGradBCC = allParametersPerfCrossMutr[33]
    dicGradBCM = allParametersPerfCrossMutr[37]
    dicKNNMC = allParametersPerfCrossMutr[41]
    dicKNNMM = allParametersPerfCrossMutr[45]
    dicLRMC = allParametersPerfCrossMutr[49]
    dicLRMM = allParametersPerfCrossMutr[53]
    dicMLPMC = allParametersPerfCrossMutr[57]
    dicMLPMM = allParametersPerfCrossMutr[61]
    dicRFMC = allParametersPerfCrossMutr[65]
    dicRFMM = allParametersPerfCrossMutr[69]
    dicGradBMC = allParametersPerfCrossMutr[73]
    dicGradBMM = allParametersPerfCrossMutr[77]

    dicKNNCC = dicKNNCC['params']
    dicKNNCM = dicKNNCM['params']
    dicLRCC = dicLRCC['params']
    dicLRCM = dicLRCM['params']
    dicMLPCC = dicMLPCC['params']
    dicMLPCM = dicMLPCM['params']
    dicRFCC = dicRFCC['params']
    dicRFCM = dicRFCM['params']
    dicGradBCC = dicGradBCC['params']
    dicGradBCM = dicGradBCM['params']
    dicKNNMC = dicKNNMC['params']
    dicKNNMM = dicKNNMM['params']
    dicLRMC = dicLRMC['params']
    dicLRMM = dicLRMM['params']
    dicMLPMC = dicMLPMC['params']
    dicMLPMM = dicMLPMM['params']
    dicRFMC = dicRFMC['params']
    dicRFMM = dicRFMM['params']
    dicGradBMC = dicGradBMC['params']
    dicGradBMM = dicGradBMM['params']
    
    dicKNNCC = {int(k):v for k,v in dicKNNCC.items()}
    dicKNNCM = {int(k):v for k,v in dicKNNCM.items()}
    dicLRCC = {int(k):v for k,v in dicLRCC.items()}
    dicLRCM = {int(k):v for k,v in dicLRCM.items()}
    dicMLPCC = {int(k):v for k,v in dicMLPCC.items()}
    dicMLPCM = {int(k):v for k,v in dicMLPCM.items()}
    dicRFCC = {int(k):v for k,v in dicRFCC.items()}
    dicRFCM = {int(k):v for k,v in dicRFCM.items()}
    dicGradBCC = {int(k):v for k,v in dicGradBCC.items()}
    dicGradBCM = {int(k):v for k,v in dicGradBCM.items()}
    dicKNNMC = {int(k):v for k,v in dicKNNMC.items()}
    dicKNNMM = {int(k):v for k,v in dicKNNMM.items()}
    dicLRMC = {int(k):v for k,v in dicLRMC.items()}
    dicLRMM = {int(k):v for k,v in dicLRMM.items()}
    dicMLPMC = {int(k):v for k,v in dicMLPMC.items()}
    dicMLPMM = {int(k):v for k,v in dicMLPMM.items()}
    dicRFMC = {int(k):v for k,v in dicRFMC.items()}
    dicRFMM = {int(k):v for k,v in dicRFMM.items()}
    dicGradBMC = {int(k):v for k,v in dicGradBMC.items()}
    dicGradBMM = {int(k):v for k,v in dicGradBMM.items()}

    dfKNNCC = pd.DataFrame.from_dict(dicKNNCC)
    dfKNNCM = pd.DataFrame.from_dict(dicKNNCM)
    dfLRCC = pd.DataFrame.from_dict(dicLRCC)
    dfLRCM = pd.DataFrame.from_dict(dicLRCM)
    dfMLPCC = pd.DataFrame.from_dict(dicMLPCC)
    dfMLPCM = pd.DataFrame.from_dict(dicMLPCM)
    dfRFCC = pd.DataFrame.from_dict(dicRFCC)
    dfRFCM = pd.DataFrame.from_dict(dicRFCM)
    dfGradBCC = pd.DataFrame.from_dict(dicGradBCC)
    dfGradBCM = pd.DataFrame.from_dict(dicGradBCM)
    dfKNNMC = pd.DataFrame.from_dict(dicKNNMC)
    dfKNNMM = pd.DataFrame.from_dict(dicKNNMM)
    dfLRMC = pd.DataFrame.from_dict(dicLRMC)
    dfLRMM = pd.DataFrame.from_dict(dicLRMM)
    dfMLPMC = pd.DataFrame.from_dict(dicMLPMC)
    dfMLPMM = pd.DataFrame.from_dict(dicMLPMM)
    dfRFMC = pd.DataFrame.from_dict(dicRFMC)
    dfRFMM = pd.DataFrame.from_dict(dicRFMM)
    dfGradBMC = pd.DataFrame.from_dict(dicGradBMC)
    dfGradBMM = pd.DataFrame.from_dict(dicGradBMM)

    dfKNNCC = dfKNNCC.T
    dfKNNCM = dfKNNCM.T
    dfLRCC = dfLRCC.T
    dfLRCM = dfLRCM.T
    dfMLPCC = dfMLPCC.T
    dfMLPCM = dfMLPCM.T
    dfRFCC = dfRFCC.T
    dfRFCM = dfRFCM.T
    dfGradBCC = dfGradBCC.T
    dfGradBCM = dfGradBCM.T
    dfKNNMC = dfKNNMC.T
    dfKNNMM = dfKNNMM.T
    dfLRMC = dfLRMC.T
    dfLRMM = dfLRMM.T
    dfMLPMC = dfMLPMC.T
    dfMLPMM = dfMLPMM.T
    dfRFMC = dfRFMC.T
    dfRFMM = dfRFMM.T
    dfGradBMC = dfGradBMC.T
    dfGradBMM = dfGradBMM.T

    df_params = pd.concat([dfKNNCC, dfKNNCM, dfLRCC, dfLRCM, dfMLPCC, dfMLPCM, dfRFCC, dfRFCM, dfGradBCC, dfGradBCM, dfKNNMC, dfKNNMM, dfLRMC, dfLRMM, dfMLPMC, dfMLPMM, dfRFMC, dfRFMM, dfGradBMC, dfGradBMM])
    df_params = df_params.reset_index(drop=True)
    return df_params

def PreprocessingParamSepCM():
    dicKNNCC = allParametersPerfCrossMutr[1]
    dicKNNCM = allParametersPerfCrossMutr[5]
    dicLRCC = allParametersPerfCrossMutr[9]
    dicLRCM = allParametersPerfCrossMutr[13]
    dicMLPCC = allParametersPerfCrossMutr[17]
    dicMLPCM = allParametersPerfCrossMutr[21]
    dicRFCC = allParametersPerfCrossMutr[25]
    dicRFCM = allParametersPerfCrossMutr[29]
    dicGradBCC = allParametersPerfCrossMutr[33]
    dicGradBCM = allParametersPerfCrossMutr[37]
    dicKNNMC = allParametersPerfCrossMutr[41]
    dicKNNMM = allParametersPerfCrossMutr[45]
    dicLRMC = allParametersPerfCrossMutr[49]
    dicLRMM = allParametersPerfCrossMutr[53]
    dicMLPMC = allParametersPerfCrossMutr[57]
    dicMLPMM = allParametersPerfCrossMutr[61]
    dicRFMC = allParametersPerfCrossMutr[65]
    dicRFMM = allParametersPerfCrossMutr[69]
    dicGradBMC = allParametersPerfCrossMutr[73]
    dicGradBMM = allParametersPerfCrossMutr[77]

    dicKNNCC = dicKNNCC['params']
    dicKNNCM = dicKNNCM['params']
    dicLRCC = dicLRCC['params']
    dicLRCM = dicLRCM['params']
    dicMLPCC = dicMLPCC['params']
    dicMLPCM = dicMLPCM['params']
    dicRFCC = dicRFCC['params']
    dicRFCM = dicRFCM['params']
    dicGradBCC = dicGradBCC['params']
    dicGradBCM = dicGradBCM['params']
    dicKNNMC = dicKNNMC['params']
    dicKNNMM = dicKNNMM['params']
    dicLRMC = dicLRMC['params']
    dicLRMM = dicLRMM['params']
    dicMLPMC = dicMLPMC['params']
    dicMLPMM = dicMLPMM['params']
    dicRFMC = dicRFMC['params']
    dicRFMM = dicRFMM['params']
    dicGradBMC = dicGradBMC['params']
    dicGradBMM = dicGradBMM['params']
    
    dicKNNCC = {int(k):v for k,v in dicKNNCC.items()}
    dicKNNCM = {int(k):v for k,v in dicKNNCM.items()}
    dicLRCC = {int(k):v for k,v in dicLRCC.items()}
    dicLRCM = {int(k):v for k,v in dicLRCM.items()}
    dicMLPCC = {int(k):v for k,v in dicMLPCC.items()}
    dicMLPCM = {int(k):v for k,v in dicMLPCM.items()}
    dicRFCC = {int(k):v for k,v in dicRFCC.items()}
    dicRFCM = {int(k):v for k,v in dicRFCM.items()}
    dicGradBCC = {int(k):v for k,v in dicGradBCC.items()}
    dicGradBCM = {int(k):v for k,v in dicGradBCM.items()}
    dicKNNMC = {int(k):v for k,v in dicKNNMC.items()}
    dicKNNMM = {int(k):v for k,v in dicKNNMM.items()}
    dicLRMC = {int(k):v for k,v in dicLRMC.items()}
    dicLRMM = {int(k):v for k,v in dicLRMM.items()}
    dicMLPMC = {int(k):v for k,v in dicMLPMC.items()}
    dicMLPMM = {int(k):v for k,v in dicMLPMM.items()}
    dicRFMC = {int(k):v for k,v in dicRFMC.items()}
    dicRFMM = {int(k):v for k,v in dicRFMM.items()}
    dicGradBMC = {int(k):v for k,v in dicGradBMC.items()}
    dicGradBMM = {int(k):v for k,v in dicGradBMM.items()}

    dfKNNCC = pd.DataFrame.from_dict(dicKNNCC)
    dfKNNCM = pd.DataFrame.from_dict(dicKNNCM)
    dfLRCC = pd.DataFrame.from_dict(dicLRCC)
    dfLRCM = pd.DataFrame.from_dict(dicLRCM)
    dfMLPCC = pd.DataFrame.from_dict(dicMLPCC)
    dfMLPCM = pd.DataFrame.from_dict(dicMLPCM)
    dfRFCC = pd.DataFrame.from_dict(dicRFCC)
    dfRFCM = pd.DataFrame.from_dict(dicRFCM)
    dfGradBCC = pd.DataFrame.from_dict(dicGradBCC)
    dfGradBCM = pd.DataFrame.from_dict(dicGradBCM)
    dfKNNMC = pd.DataFrame.from_dict(dicKNNMC)
    dfKNNMM = pd.DataFrame.from_dict(dicKNNMM)
    dfLRMC = pd.DataFrame.from_dict(dicLRMC)
    dfLRMM = pd.DataFrame.from_dict(dicLRMM)
    dfMLPMC = pd.DataFrame.from_dict(dicMLPMC)
    dfMLPMM = pd.DataFrame.from_dict(dicMLPMM)
    dfRFMC = pd.DataFrame.from_dict(dicRFMC)
    dfRFMM = pd.DataFrame.from_dict(dicRFMM)
    dfGradBMC = pd.DataFrame.from_dict(dicGradBMC)
    dfGradBMM = pd.DataFrame.from_dict(dicGradBMM)

    dfKNNCC = dfKNNCC.T
    dfKNNCM = dfKNNCM.T
    dfLRCC = dfLRCC.T
    dfLRCM = dfLRCM.T
    dfMLPCC = dfMLPCC.T
    dfMLPCM = dfMLPCM.T
    dfRFCC = dfRFCC.T
    dfRFCM = dfRFCM.T
    dfGradBCC = dfGradBCC.T
    dfGradBCM = dfGradBCM.T
    dfKNNMC = dfKNNMC.T
    dfKNNMM = dfKNNMM.T
    dfLRMC = dfLRMC.T
    dfLRMM = dfLRMM.T
    dfMLPMC = dfMLPMC.T
    dfMLPMM = dfMLPMM.T
    dfRFMC = dfRFMC.T
    dfRFMM = dfRFMM.T
    dfGradBMC = dfGradBMC.T
    dfGradBMM = dfGradBMM.T

    return [dfKNNCC, dfKNNCM, dfLRCC, dfLRCM, dfMLPCC, dfMLPCM, dfRFCC, dfRFCM, dfGradBCC, dfGradBCM, dfKNNMC, dfKNNMM, dfLRMC, dfLRMM, dfMLPMC, dfMLPMM, dfRFMC, dfRFMM, dfGradBMC, dfGradBMM]

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
        if sum(factors) == 0:
            sumPerClassifier = 0
        else:
            sumPerClassifier.append(rowSum/sum(factors) * 100)
    return sumPerClassifier

def preProcsumPerMetricCMSecond(factors):
    sumPerClassifier = []
    loopThroughMetrics = PreprocessingMetricsCMSecond()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    loopThroughMetrics.loc[:, 'log_loss'] = 1 - loopThroughMetrics.loc[:, 'log_loss']
    for row in loopThroughMetrics.iterrows():
        rowSum = 0
        name, values = row
        for loop, elements in enumerate(values):
            rowSum = elements*factors[loop] + rowSum
        if sum(factors) == 0:
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
    metricsPerModelColl.append(loopThroughMetrics['geometric_mean_score_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_precision_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_recall_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_f1_macro'])
    metricsPerModelColl.append(loopThroughMetrics['matthews_corrcoef'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_roc_auc_ovo'])
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


def preProcMetricsAllAndSelCMSecond():
    loopThroughMetrics = PreprocessingMetricsCMSecond()
    loopThroughMetrics = loopThroughMetrics.fillna(0)
    global factors
    metricsPerModelColl = []
    metricsPerModelColl.append(loopThroughMetrics['mean_test_accuracy'])
    metricsPerModelColl.append(loopThroughMetrics['geometric_mean_score_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_precision_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_recall_macro'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_f1_macro'])
    metricsPerModelColl.append(loopThroughMetrics['matthews_corrcoef'])
    metricsPerModelColl.append(loopThroughMetrics['mean_test_roc_auc_ovo'])
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
    global CurStage
    if (CurStage == 1):
        PreProcessingInitial()
        response = {    
            'OverviewResultsCM': ResultsCM
        }
    else:
        PreProcessingSecond()
        response = {    
            'OverviewResultsCM': ResultsCMSecond
        }
    return jsonify(response)

def PreProcessingInitial(): 
    XModels = PreprocessingMetricsCM()
    global allParametersPerfCrossMutr

    global factors
    XModels = XModels.fillna(0)
    dropMetrics = []
    for index, element in enumerate(factors):
        if (element == 0):
            dropMetrics.append(index)
    
    XModels.drop(XModels.columns[dropMetrics], axis=1, inplace=True)

    ModelSpaceMDSCM = FunMDS(XModels)
    ModelSpaceTSNECM = FunTsne(XModels)
    ModelSpaceTSNECM = ModelSpaceTSNECM.tolist()
    ModelSpaceUMAPCM = FunUMAP(XModels)

    PredictionProbSelCM = PreprocessingPredCM()

    CrossMutateResults(ModelSpaceMDSCM,ModelSpaceTSNECM,ModelSpaceUMAPCM,PredictionProbSelCM)

def PreProcessingSecond(): 
    XModels = PreprocessingMetricsCMSecond()

    global factors
    XModels = XModels.fillna(0)
    dropMetrics = []
    for index, element in enumerate(factors):
        if (element == 0):
            dropMetrics.append(index)
    
    XModels.drop(XModels.columns[dropMetrics], axis=1, inplace=True)

    ModelSpaceMDSCMSecond = FunMDS(XModels)
    ModelSpaceTSNECMSecond = FunTsne(XModels)
    ModelSpaceTSNECMSecond = ModelSpaceTSNECMSecond.tolist()
    ModelSpaceUMAPCMSecond = FunUMAP(XModels)

    PredictionProbSelCMSecond = PreprocessingPredCMSecond()

    CrossMutateResultsSecond(ModelSpaceMDSCMSecond,ModelSpaceTSNECMSecond,ModelSpaceUMAPCMSecond,PredictionProbSelCMSecond)

def CrossMutateResults(ModelSpaceMDSCM,ModelSpaceTSNECM,ModelSpaceUMAPCM,PredictionProbSelCM):

    global ResultsCM
    global AllTargets
    global names_labels
    global yDataSorted
    ResultsCM = []

    parametersGenCM = PreprocessingParamCM()
    metricsPerModelCM = preProcMetricsAllAndSelCM()
    sumPerClassifierCM = preProcsumPerMetricCM(factors)
    ModelsIDsCM = PreprocessingIDsCM()
    parametersGenPDGM = parametersGenCM.to_json(orient='records')
    XDataJSONEntireSet = XData.to_json(orient='records')
    XDataColumns = XData.columns.tolist()

    ResultsCM.append(json.dumps(ModelsIDsCM))
    ResultsCM.append(json.dumps(sumPerClassifierCM))
    ResultsCM.append(json.dumps(parametersGenPDGM))
    ResultsCM.append(json.dumps(metricsPerModelCM))
    ResultsCM.append(json.dumps(XDataJSONEntireSet))
    ResultsCM.append(json.dumps(XDataColumns))
    ResultsCM.append(json.dumps(yData))
    ResultsCM.append(json.dumps(target_names))
    ResultsCM.append(json.dumps(AllTargets))
    ResultsCM.append(json.dumps(ModelSpaceMDSCM))
    ResultsCM.append(json.dumps(ModelSpaceTSNECM))
    ResultsCM.append(json.dumps(ModelSpaceUMAPCM))
    ResultsCM.append(json.dumps(PredictionProbSelCM))
    ResultsCM.append(json.dumps(names_labels))
    ResultsCM.append(json.dumps(yDataSorted))

    return ResultsCM

def CrossMutateResultsSecond(ModelSpaceMDSCMSecond,ModelSpaceTSNECMSecond,ModelSpaceUMAPCMSecond,PredictionProbSelCMSecond):

    global ResultsCMSecond
    global AllTargets
    global names_labels
    global yDataSorted
    ResultsCMSecond = []

    parametersGenCMSecond = PreprocessingParamCMSecond()
    metricsPerModelCMSecond = preProcMetricsAllAndSelCMSecond()
    sumPerClassifierCMSecond = preProcsumPerMetricCMSecond(factors)
    ModelsIDsCMSecond = PreprocessingIDsCMSecond()
    parametersGenPDGMSecond = parametersGenCMSecond.to_json(orient='records')
    XDataJSONEntireSet = XData.to_json(orient='records')
    XDataColumns = XData.columns.tolist()

    ResultsCMSecond.append(json.dumps(ModelsIDsCMSecond))
    ResultsCMSecond.append(json.dumps(sumPerClassifierCMSecond))
    ResultsCMSecond.append(json.dumps(parametersGenPDGMSecond))
    ResultsCMSecond.append(json.dumps(metricsPerModelCMSecond))
    ResultsCMSecond.append(json.dumps(XDataJSONEntireSet))
    ResultsCMSecond.append(json.dumps(XDataColumns))
    ResultsCMSecond.append(json.dumps(yData))
    ResultsCMSecond.append(json.dumps(target_names))
    ResultsCMSecond.append(json.dumps(AllTargets))
    ResultsCMSecond.append(json.dumps(ModelSpaceMDSCMSecond))
    ResultsCMSecond.append(json.dumps(ModelSpaceTSNECMSecond))
    ResultsCMSecond.append(json.dumps(ModelSpaceUMAPCMSecond))
    ResultsCMSecond.append(json.dumps(PredictionProbSelCMSecond))
    ResultsCMSecond.append(json.dumps(names_labels))
    ResultsCMSecond.append(json.dumps(yDataSorted))

    return ResultsCMSecond

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
            if ((items[0] == "KNN") | (items[0] == "KNNC") | (items[0] == "KNNM") | (items[0] == "KNNCC") | (items[0] == "KNNCM") | (items[0] == "KNNMC") | (items[0] == "KNNMM")):
                numberIDKNN.append(int(items[1]) - addKNN)
            elif ((items[0] == "LR") | (items[0] == "LRC") | (items[0] == "LRM") | (items[0] == "LRCC") | (items[0] == "LRCM") | (items[0] == "LRMC") | (items[0] == "LRMM")):
                numberIDLR.append(int(items[1]) - addLR)
            elif ((items[0] == "MLP") | (items[0] == "MLPC") | (items[0] == "MLPM") | (items[0] == "MLPCC") | (items[0] == "MLPCM") | (items[0] == "MLPMC") | (items[0] == "MLPMM")):
                numberIDMLP.append(int(items[1]) - addMLP)
            elif ((items[0] == "RF") | (items[0] == "RFC") | (items[0] == "RFM") | (items[0] == "RFCC") | (items[0] == "RFCM") | (items[0] == "RFMC") | (items[0] == "RFMM")):
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

    dfKNN.index += addKNN

    dfLR = pd.DataFrame.from_dict(dicLR)
    dfLR = dfLR.loc[numberIDLR]

    dfLR.index += addLR

    dfMLP = pd.DataFrame.from_dict(dicMLP)
    dfMLP = dfMLP.loc[numberIDMLP]

    dfMLP.index += addMLP

    dfRF = pd.DataFrame.from_dict(dicRF)

    dfRF = dfRF.loc[numberIDRF]

    dfRF.index += addRF

    dfGradB = pd.DataFrame.from_dict(dicGradB)

    dfGradB = dfGradB.loc[numberIDGradB]

    dfGradB.index += addGradB

    df_concatProbs = pd.concat([dfKNN, dfLR, dfMLP, dfRF, dfGradB])
    df_concatProbs = df_concatProbs.reset_index(drop=True)

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

    global storeClass0
    global storeClass1
    global yDataSorted

    firstElKNN = []
    firstElLR = []
    firstElMLP = []
    firstElRF = []
    firstElGradB = []
    firstElPredAv = []
    lastElKNN = []
    lastElLR = []
    lastElMLP = []
    lastElRF = []
    lastElGradB = []
    lastElPredAv = []
    yDataSortedFirst = []
    yDataSortedLast = []

    for index, item in enumerate(yData):
        if (item == 0):
            if (len(predictionsKNN[index]) != 0):
                firstElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                firstElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                firstElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                firstElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                firstElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                firstElPredAv.append(predictions[index][item]*100)
            yDataSortedFirst.append(item)
        else:
            if (len(predictionsKNN[index]) != 0):
                lastElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                lastElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                lastElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                lastElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                lastElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                lastElPredAv.append(predictions[index][item]*100)
            yDataSortedLast.append(item)

    if (storeClass0 > 169 & storeClass1 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)
        lastElKNN = computeClusters(lastElKNN)
        firstElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass0 < 169 & storeClass1 > 169):
                
        lastElKNN = computeClusters(lastElKNN)
        lastElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass1 < 169 & storeClass0 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)

    else:
        pass

    predictionsKNN = firstElKNN + lastElKNN
    predictionsLR = firstElLR + lastElLR        
    predictionsMLP = firstElMLP + lastElMLP
    predictionsRF = firstElRF + lastElRF
    predictionsGradB = firstElGradB + lastElGradB
    predictions = firstElPredAv + lastElPredAv
    yDataSorted = yDataSortedFirst + yDataSortedLast
 
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
            if ((items[0] == "KNN") | (items[0] == "KNNC") | (items[0] == "KNNM") | (items[0] == "KNNCC") | (items[0] == "KNNCM") | (items[0] == "KNNMC") | (items[0] == "KNNMM")):
                numberIDKNN.append(int(items[1]))
            elif ((items[0] == "LR") | (items[0] == "LRC") | (items[0] == "LRM") | (items[0] == "LRCC") | (items[0] == "LRCM") | (items[0] == "LRMC") | (items[0] == "LRMM")):
                numberIDLR.append(int(items[1]))
            elif ((items[0] == "MLP") | (items[0] == "MLPC") | (items[0] == "MLPM") | (items[0] == "MLPCC") | (items[0] == "MLPCM") | (items[0] == "MLPMC") | (items[0] == "MLPMM")):
                numberIDMLP.append(int(items[1]))
            elif ((items[0] == "RF") | (items[0] == "RFC") | (items[0] == "RFM") | (items[0] == "RFCC") | (items[0] == "RFCM") | (items[0] == "RFMC") | (items[0] == "RFMM")):
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

    global storeClass0
    global storeClass1
    global yDataSorted

    firstElKNN = []
    firstElLR = []
    firstElMLP = []
    firstElRF = []
    firstElGradB = []
    firstElPredAv = []
    lastElKNN = []
    lastElLR = []
    lastElMLP = []
    lastElRF = []
    lastElGradB = []
    lastElPredAv = []
    yDataSortedFirst = []
    yDataSortedLast = []

    for index, item in enumerate(yData):
        if (item == 0):
            if (len(predictionsKNN[index]) != 0):
                firstElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                firstElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                firstElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                firstElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                firstElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                firstElPredAv.append(predictions[index][item]*100)
            yDataSortedFirst.append(item)
        else:
            if (len(predictionsKNN[index]) != 0):
                lastElKNN.append(predictionsKNN[index][item]*100)
            if (len(predictionsLR[index]) != 0):
                lastElLR.append(predictionsLR[index][item]*100)
            if (len(predictionsMLP[index]) != 0):
                lastElMLP.append(predictionsMLP[index][item]*100)
            if (len(predictionsRF[index]) != 0):
                lastElRF.append(predictionsRF[index][item]*100)
            if (len(predictionsGradB[index]) != 0):
                lastElGradB.append(predictionsGradB[index][item]*100)
            if (len(predictions[index]) != 0):
                lastElPredAv.append(predictions[index][item]*100)
            yDataSortedLast.append(item)

    if (storeClass0 > 169 & storeClass1 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)
        lastElKNN = computeClusters(lastElKNN)
        firstElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass0 < 169 & storeClass1 > 169):
                
        lastElKNN = computeClusters(lastElKNN)
        lastElLR = computeClusters(lastElLR)
        lastElMLP = computeClusters(lastElMLP)
        lastElRF = computeClusters(lastElRF)
        lastElGradB = computeClusters(lastElGradB)
        lastElPredAv = computeClusters(lastElPredAv)

    elif (storeClass1 < 169 & storeClass0 > 169):
                
        firstElKNN = computeClusters(firstElKNN)
        firstElLR = computeClusters(firstElLR)
        firstElMLP = computeClusters(firstElMLP)
        firstElRF = computeClusters(firstElRF)
        firstElGradB = computeClusters(firstElGradB)
        firstElPredAv = computeClusters(firstElPredAv)

    else:
        pass

    predictionsKNN = firstElKNN + lastElKNN
    predictionsLR = firstElLR + lastElLR        
    predictionsMLP = firstElMLP + lastElMLP
    predictionsRF = firstElRF + lastElRF
    predictionsGradB = firstElGradB + lastElGradB
    predictions = firstElPredAv + lastElPredAv
    yDataSorted = yDataSortedFirst + yDataSortedLast

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

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRequestSelPoin', methods=["GET", "POST"])
def RetrieveSelClassifiersID():
    global EnsembleActive
    ClassifierIDsList = request.get_data().decode('utf8').replace("'", '"')
    #ComputeMetricsForSel(ClassifierIDsList)
    ClassifierIDCleaned = json.loads(ClassifierIDsList)
    ClassifierIDCleaned = ClassifierIDCleaned['ClassifiersList']

    EnsembleActive = []
    EnsembleActive = ClassifierIDCleaned.copy()
    EnsembleIDs()
    EnsembleModel(ClassifierIDsList, 1)

    return 'Everything Okay'

@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
@app.route('/data/ServerRemoveFromEnsemble', methods=["GET", "POST"])
def RetrieveSelClassifiersIDandRemoveFromEnsemble():
    global EnsembleActive
    ClassifierIDsList = request.get_data().decode('utf8').replace("'", '"')
    ClassifierIDsList = json.loads(ClassifierIDsList)
    ClassifierIDsListCleaned = ClassifierIDsList['ClassifiersList']
    
    EnsembleActive = []
    EnsembleActive = ClassifierIDsListCleaned.copy()

    return 'Everything Okay'