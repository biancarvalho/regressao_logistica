import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

###################################################################################

df = pd.read_csv("stars_data.csv")

df = df[(df['Spectral Class'] == 'M') | (df['Spectral Class'] == 'B')]
df['Spectral Class'].replace(to_replace='M', value=1, inplace=True)
df['Spectral Class'].replace(to_replace='B', value=0, inplace=True)
df = df.dropna()


df[['Temperature (K)']] = df[['Temperature (K)']]/df[['Temperature (K)']].mean()
df[['Radius(R/Ro)']] = df[['Radius(R/Ro)']]/df[['Radius(R/Ro)']].mean()

X = df[['Temperature (K)', 'Radius(R/Ro)']].to_numpy()
y = df[['Spectral Class']].to_numpy()
y = np.hstack((y)).T

#############################################################################

regressor = LogisticRegression(learning_rate=0.1, n_iters=2000)
regressor.fit(X, y)
predictions = regressor.predict(X)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print(accuracy(y, predictions))

#############################################################################

# cara descobri isso só testando vários valores aleatoriamente e deu certo KKKKKKKKKKK
slope = -(regressor.weights[0]/regressor.weights[1])
intercept = -(regressor.bias/regressor.weights[1])

sns.set_style('white')
sns.scatterplot(x = X[:,0], y= X[:,1], hue=y.reshape(-1), style=predictions.reshape(-1));

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");


