# first line: 741
@memory.cache
def randomSearch(XData, yData, clf, params, eachAlgor, AlgorithmsIDsEnd,crossValidation,randomSear):
    print('inside')
    print(clf)
    search = RandomizedSearchCV(    
        estimator=clf, param_distributions=params, n_iter=randomSear,
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
