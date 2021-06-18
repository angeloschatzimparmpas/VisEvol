# VisEvol: Visual Analytics to Support Hyperparameter Search through Evolutionary Optimization

This Git repository contains the code that accompanies the research paper "VisEvol: Visual Analytics to Support Hyperparameter Search through Evolutionary Optimization". The details of the experiments and the research outcome are described in [the paper](https://diglib.eg.org/handle/10.1111/cgf14300).

**Note:** VisEvol is optimized to work better for standard resolutions (such as 1440p/QHD (Quad High Definition) and 1080p). Any other resolution might need manual adjustment of your browser's zoom level to work properly.

**Note:** The tag `paper-version` matches the implementation at the time of the paper's publication. The current version might look significantly different depending on how much time has passed since then.

**Note:** As any other software, the code is not bug free. There might be limitations in the views and functionalities of the tool that could be addressed in a future code update.

# Data Sets #
All publicly available data sets used in the paper are in the `data` folder, formatted as comma separated values (csv). 
They are also available online from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php): Heart Disease and QSAR Biodegradation.

# Requirements #
For the backend:
- [Python 3](https://www.python.org/downloads/)
- [Flask](https://palletsprojects.com/p/flask/)
- Other packages: `pymongo`, `numpy`, `scipy`, `scikit-learn`, `sk-dist`, `eli5`, and `pandas`.

You can install all the backend requirements with the following command:
```
pip install -r requirements.txt
```

For the frontend:
- [Node.js](https://nodejs.org/en/)
- [D3.js](https://d3js.org/)
- [Plotly.js](https://github.com/plotly/plotly.js/)

There is no need to install anything for the frontend, since all modules are in the repository.

# Usage #
Below is an example of how you can get VisEvol running using Python for both frontend and backend. The frontend is written in JavaScript/HTML, so it could be hosted in any other web server of your preference. The only hard requirement (currently) is that both frontend and backend must be running on the same machine. 
```
# first terminal: hosting the visualization side (client)
# with Node.js
cd frontend
npm run dev
```

```
# second terminal: hosting the computational side (server)
FLASK_APP=run.py flask run

# (optional) recommendation: use insertMongo script to add a data set in Mongo database
# for Python3
python3 insertMongo.py
```

Then, open your browser and point it to `localhost:8080`. We recommend using an up-to-date version of Google Chrome.

# Hyper-Parameters per Algorithm #
**Random Search:**
- **K-Nearest Neighbor:** {'n_neighbors': list(range(1, 100)), 'metric': ['chebyshev', 'manhattan', 'euclidean', 'minkowski'], 'algorithm': ['brute', 'kd_tree', 'ball_tree'], 'weights': ['uniform', 'distance']}
- **Logistic Regression:** {'C': list(np.arange(1,100,1)), 'max_iter': list(np.arange(50,500,50)), 'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'], 'penalty': ['l2', 'none']}
- **Multilayer Perceptron:** {'hidden_layer_sizes': ranges,'alpha': list(np.arange(0.00001,0.001,0.0002)), 'tol': list(np.arange(0.00001,0.001,0.0004)), 'max_iter': list(np.arange(100,200,100)), 'activation': ['relu', 'identity', 'logistic', 'tanh'], 'solver' : ['adam', 'sgd']}, where ranges=[(n, random.randint(1,3)) for n in range(start=60, stop=120, step=1)] with RANDOM_SEED=42
- **Random Forests:** {'n_estimators': list(range(20, 100)), 'max_depth': list(range(2, 20)), 'criterion': ['gini', 'entropy']}
- **Gradient Boosting:** {'n_estimators': list(range(20, 100)), 'loss': ['deviance','exponential'], 'learning_rate': list(np.arange(0.01,0.56,0.11)), 'subsample': list(np.arange(0.1,1,0.1)), 'criterion': ['friedman_mse', 'mse', 'mae']}

**Crossover**: 
- It happens by mixing randomly models (and their hyperparameters) originating from the same algorithms.
- Only the unselected models by the user are transformed with this process.

**Mutation**: 
- It happens by picking randomly a new (outside of the previous ranges) value for the primary hyperparameter (according to Scikit-learn) of each algorithm.
- Only the unselected models by the user are transformed with this process.

# Corresponding Author #
For any questions with regard to the implementation or the paper, feel free to contact [Angelos Chatzimparmpas](mailto:angelos.chatzimparmpas@lnu.se).
