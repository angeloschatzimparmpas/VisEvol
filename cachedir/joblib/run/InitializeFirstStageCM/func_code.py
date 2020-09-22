# first line: 1312
@memory.cache
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

    while countKNN < setMaxLoopValue[16]:
        for dr in KNNIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNN_C', AlgorithmsIDsEnd)
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

    while countKNN < setMaxLoopValue[10]:
        for dr in KNNIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                KNNIntIndex.append(int(re.findall('\d+', dr)[0])-addKNN)
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'KNN_M', AlgorithmsIDsEnd)
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

    while countLR < setMaxLoopValue[15]:
        for dr in LRIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LR_C', AlgorithmsIDsEnd)
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

    while countLR < setMaxLoopValue[9]:
        for dr in LRIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                LRIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar))
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'LR_M', AlgorithmsIDsEnd)
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

    while countMLP < setMaxLoopValue[14]:
        for dr in MLPIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLP_C', AlgorithmsIDsEnd)
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

    while countMLP < setMaxLoopValue[8]:
        for dr in MLPIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                MLPIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*2))
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'MLP_M', AlgorithmsIDsEnd)
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

    while countRF < setMaxLoopValue[13]:
        for dr in RFIDs:
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
        if (((paramAllAlgs['n_estimators'] == crossoverDF['n_estimators'].iloc[0]) & (paramAllAlgs['criterion'] == crossoverDF['criterion'].iloc[0])).any()):
            crossoverDF = pd.DataFrame()
        else:
            clf = RandomForestClassifier(random_state=RANDOM_SEED)
            params = {'n_estimators': [crossoverDF['n_estimators'].iloc[0]], 'criterion': [crossoverDF['criterion'].iloc[0]]}
            AlgorithmsIDsEnd = countAllModels + countRF
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RF_C', AlgorithmsIDsEnd)
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

    while countRF < setMaxLoopValue[7]:
        if (int(re.findall('\d+', dr)[0]) >= greater):
            RFIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*3))
        else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'RF_M', AlgorithmsIDsEnd)
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

    while countGradB < setMaxLoopValue[12]:
        for dr in GradBIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradB_C', AlgorithmsIDsEnd)
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
            
    while countGradB < setMaxLoopValue[6]:
        for dr in GradBIDs:
            if (int(re.findall('\d+', dr)[0]) >= greater):
                GradBIntIndex.append(int(re.findall('\d+', dr)[0])-(addKNN-randomSearchVar*4))
            else:
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
            localCrossMutr = crossoverMutation(XData, yData, clf, params, 'GradB_M', AlgorithmsIDsEnd)
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

    addKNN = addGradB

    addLR = addKNN + setMaxLoopValue[16] + setMaxLoopValue[10]

    addMLP = addLR + setMaxLoopValue[15] + setMaxLoopValue[9]

    addRF = addMLP + setMaxLoopValue[14] + setMaxLoopValue[8]

    addGradB = addRF + setMaxLoopValue[13] + setMaxLoopValue[7]

    return 'Everything Okay'
