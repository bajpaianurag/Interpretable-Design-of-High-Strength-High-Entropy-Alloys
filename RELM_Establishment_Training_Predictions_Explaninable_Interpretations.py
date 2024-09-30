# Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.font_manager as font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from sklearn.neighbors import KernelDensity
from tensorflow.keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import shap
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import csv


# Data input and preprocessing 
data = pd.read_csv('input_dataset.csv')

# split into features and target variable
X = data.drop('YS', axis=1)
y = data ['YS']

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale the dataset
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Bayesian optimization of Hyperparameters with Regressor Training and Prediction
models = {
    "Random Forest": (RandomForestRegressor(), {
        'n_estimators': (10, 200),
        'max_depth': (2, 30),
        'min_samples_split': (2, 30),
        'min_samples_leaf': (1, 20),
        'max_features': (0.1, 1.0),
        'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
    }),
    "SVR": (SVR(), {
        'C': (0.001, 100.0),                        
        'gamma': ('scale', 'auto'),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }),
    "Lasso Regression": (Lasso(), {
        'alpha': (0.000001, 1.0),                     
        'selection': ['cyclic', 'random']
    }),
    "Ridge Regression": (Ridge(), {
        'alpha': (0.000001, 10.0),                     
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }),
    "Elastic Net": (ElasticNet(), {
        'alpha': (0.000001, 10.0),                     
        'l1_ratio': (0.001, 0.99),                    
        'selection': ['cyclic', 'random']
    }),
    "K-Nearest Neighbors": (KNeighborsRegressor(), {
        'n_neighbors': (2, 50),                      
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }),
    "Gradient Boosting": (GradientBoostingRegressor(), {
        'n_estimators': (10, 200),
        'learning_rate': (0.00001, 1.0),              
        'max_depth': (1, 30),                        
        'min_samples_split': (2, 30),                               
        'max_features': (0.1, 1.0),  
    })
}


# Train and evaluate each model
best_model = None
best_score = float('inf')
predictions = {}
predictions_train = {}

# Set font properties
font_prop = {'family': 'Arial', 'size': 28}

# Define a list of colors for scatter plots
scatter_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'g']
color_index = 0
scatter_point_size = 150
scatter_alpha = 0.7

prediction_uncertainties = {}

for name, (model, search_space) in models1.items():
    # Perform hyperparameter tuning using Bayesian optimization
    model = BayesSearchCV(model, search_spaces=search_space, n_iter=1000, n_jobs=-1, cv=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Print best hyperparameters
    print(f"{name} Best Hyperparameters: {model.best_params_}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    y_pred_train = model.predict(X_train)   
    predictions_train[name] = y_pred_train

    # Reshape predicted values to 1D array
    y_pred = y_pred.reshape(-1)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{name} MSE: {mse}")
    print(f"{name} R2 Score: {r2}")
    print(f"{name} MAE: {mae}")
    print()
    
    gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10, random_state=42)
    gpr.fit(X_train, y_train)
    
    y_std, _ = gpr.predict(X_test, return_std=True)
    prediction_uncertainties[name] = y_std
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 9))
    plt.scatter(y_train, y_pred_train, color=scatter_colors[color_index+1], label=name, s=scatter_point_size, alpha=scatter_alpha)
    plt.scatter(y_test, y_pred, color=scatter_colors[color_index], label=name, s=scatter_point_size, alpha=scatter_alpha)
    plt.plot([100, 1200], [100, 1200], 'k--')
    color_index = (color_index + 1) % len(scatter_colors)
    plt.xlabel("Actual Yield Strength (MPa)", fontdict=font_prop)
    plt.ylabel("Predicted Yield Strength (MPa)", fontdict=font_prop)
    plt.xlim([100, 1200])
    plt.ylim([100, 1200])
    plt.xticks(fontproperties='Arial', fontsize=20)
    plt.yticks(fontproperties='Arial', fontsize=20)
    plt.show()
    
    # Keep track of the best performing model
    if mse < best_score:
        best_score = mse
        best_model = model.best_estimator_

# Initialize the residuals dictionary to store errors for each model
residuals = {}

# Define a list of colors for scatter plots
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
color_index = 0
font_prop = {'family': 'Arial', 'size': 28}

for name, (model, search_space) in models1.items():
    residuals[name] = y_test - predictions[name]
    
    # Define a list of colors for scatter plots
    plt.figure(figsize=(10, 9))
    plt.hist(residuals[name], bins=20, alpha=0.7, label=name, color=colors[color_index])
    color_index = (color_index + 1) % len(scatter_colors)
    plt.xlabel('Residuals', fontdict=font_prop)
    plt.ylabel('Frequency', fontdict=font_prop)
    plt.title('Histogram of Residuals', fontdict=font_prop)
    plt.xticks(fontproperties='Arial', fontsize=20)
    plt.yticks(fontproperties='Arial', fontsize=20)
    plt.legend()
    plt.show()

# Define the custom regressor class
class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layers=1, neurons_per_layer=10, activation='relu',
                 optimizer=Adam, learning_rate=0.00001, batch_size=16, epochs=3000):
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.loss_history = []

    def create_model(self):
        model = Sequential()
        model.add(Dense(self.neurons_per_layer, input_dim=X_train.shape[1], activation=self.activation))
        for _ in range(self.hidden_layers - 1):
            model.add(Dense(self.neurons_per_layer, activation=self.activation))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=self.optimizer(learning_rate=self.learning_rate))
        return model

    def fit(self, X, y):
        self.model = self.create_model()
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.loss_history = history.history['loss']

    def predict(self, X):
        return self.model.predict(X)

search_space = {
    'hidden_layers': (1, 4),             
    'neurons_per_layer': (10, 100),          
    'activation': ['relu', 'sigmoid', 'tanh'],        
    'optimizer': [Adam],                      
    'learning_rate': (0.00001, 0.1, 'log-uniform'),  
    'batch_size': (8, 64)                   
}

opt = BayesSearchCV(
    NeuralNetworkRegressor(),
    search_space,
    n_iter=1000,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    optimizer_kwargs={'base_estimator': 'GP'},
    verbose=1
)

opt.fit(X_train_scaled, y_train)
best_params_nn = opt.best_params_
model_NN = NeuralNetworkRegressor(**best_params_nn)
print('Best Hyperparameters:', best_params_nn)

# Set font properties
font_prop = {'family': 'Arial', 'size': 28}

plt.figure(figsize=(10, 9))
plt.plot(range(1, len(model_NN.loss_history) + 1), model_NN.loss_history, color = 'b')
if hasattr(model_NN, 'validation_loss_history'):
    plt.plot(range(1, len(model_NN.validation_loss_history) + 1), model_NN.validation_loss_history, label='Validation Loss', color='r')

    plt.xlabel('Number of Epochs', fontdict=font_prop)
plt.ylabel('Loss Value', fontdict=font_prop)
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.show()

loss_history = model_NN.loss_history

y_pred_NN = model_NN.predict(X_test_scaled)
y_pred_NN = y_pred_NN.reshape(-1)

y_pred_NN_train = model_NN.predict(X_train_scaled)

mse_NN = mean_squared_error(y_test, y_pred_NN)
r2_NN = r2_score(y_test, y_pred_NN)
mae_NN = mean_absolute_error(y_test, y_pred_NN)    
print("NN MSE:", mse_NN)
print("NN R2 Score:", r2_NN)
print("NN MAE:", mae_NN)
print()

# Set font properties
font_prop = {'family': 'Arial', 'size': 28}

# Define a list of colors for scatter plots
scatter_point_size = 150
scatter_alpha = 0.7

# Plot actual vs predicted values
plt.figure(figsize=(10, 9))
plt.scatter(y_test, y_pred_NN, color='crimson', label=name, s=scatter_point_size, alpha=scatter_alpha)
plt.xlabel("Actual Yield Strength (MPa)", fontdict=font_prop)
plt.ylabel("Predicted Yield Strength (MPa)", fontdict=font_prop)
plt.xlim([100, 1000])
plt.ylim([100, 1000])
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.show()

residuals_NN = {}
font_prop = {'family': 'Arial', 'size': 28}
residuals_NN = y_test - y_pred_NN

plt.figure(figsize=(10, 9))
plt.hist(residuals_NN, bins=20, alpha=0.5, label='Neural Network', color='b')
plt.xlabel('Residuals', fontdict=font_prop)
plt.ylabel('Frequency', fontdict=font_prop)
plt.title('Histogram of Residuals', fontdict=font_prop)
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.legend()
plt.show()

# Create the RELM
RELM = best_model

# Train the RELM using the two best performing models
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()

# Fit the Random Forest model on the training data with the best hyperparameters
rf_model.fit(X_train_scaled, y_train)

# Fit the Gradient Boosting model on the residuals of the Random Forest predictions with the best hyperparameters
rf_predictions = rf_model.predict(X_test_scaled)
residuals = y_test - rf_predictions
gb_model.fit(X_test_scaled, residuals)

# Make predictions with the RELM
rf_predictions_test = rf_model.predict(X_test_scaled)
RELM_predictions_test = rf_predictions_test + gb_model.predict(X_test_scaled)

rf_predictions_train = rf_model.predict(X_train_scaled)
RELM_predictions_train = rf_predictions_train + gb_model.predict(X_train_scaled)

# Compute the MSE, R2 score, and MAE of the RELM
mse_RELM = mean_squared_error(y_test, RELM_predictions_test)
r2_RELM = r2_score(y_test, RELM_predictions_test)
mae_RELM = mean_absolute_error(y_test, RELM_predictions_test)

print(f"RELM MSE: {mse_RELM}")
print(f"RELM R2 Score: {r2_RELM}")
print(f"RELM MAE: {mae_RELM}")

# Plot actual vs predicted values for the RELM

font_prop = {'family': 'Arial', 'size': 28}
color_index = 0
scatter_point_size = 150
scatter_alpha = 0.7

squared_diff = (y_test - RELM_predictions_test) ** 2
variance = np.var(squared_diff)

# Plot actual vs predicted values
plt.figure(figsize=(10, 9))
plt.scatter(y_train, RELM_predictions_train, color='b', label=name, s=scatter_point_size, alpha=scatter_alpha)
plt.scatter(y_test, RELM_predictions_test, color='r', label=name, s=scatter_point_size, alpha=scatter_alpha)
plt.plot([100, 1200], [100, 1200], 'k--')
color_index = (color_index + 1) % len(scatter_colors)
plt.xlabel("Actual Yield Strength (MPa)", fontdict=font_prop)
plt.ylabel("Predicted Yield Strength (MPa)", fontdict=font_prop)
plt.xlim([100, 1200])
plt.ylim([100, 1200])
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.show()

residuals_RELM = {}
residuals_RELM = y_test - RELM_predictions_test

# Define a list of colors for scatter plots
plt.figure(figsize=(10, 9))
plt.hist(residuals_RELM, bins=20, alpha=0.5, label='Neural Network', color='b')
plt.xlabel('Residuals', fontdict=font_prop)
plt.ylabel('Frequency', fontdict=font_prop)
plt.title('Histogram of Residuals', fontdict=font_prop)
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.legend()
plt.show()

## Prediction of new alloys with processing parameters
# load new compositions dataset
X_data_pred = pd.read_csv('data_for_prediction.csv')
rf_predictions_pred = rf_model.predict(X_data_pred)
RELM_predictions_pred = rf_predictions_pred + gb_model.predict(X_data_pred)


### Error Analysis

# Create a KDE plot for the dataset
plt.figure(figsize=(10, 6))
sns.kdeplot(dataset, shade=True, label="Dataset")

plt.xlabel("RMSE")
plt.ylabel("Probability")
plt.legend()
plt.show()

# Calculate KDE values separately
kde_values = sns.kdeplot(dataset).get_lines()[0].get_data()
x_values, y_values = kde_values[0], kde_values[1]


### Explaining the RELM

feature_names = X_train_scaled.columns.tolist()

explainer = shap.Explainer(gb_model)
shap_values = explainer(X_train_scaled)

max_display = 32  

plt.figure(figsize=(12, 12))
shap.summary_plot(shap_values, X_train_scaled, plot_type='bar', color = 'b', feature_names=feature_names, max_display=max_display)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_names, plot_type='violin', show=False)
plt.show()


# Create an empty DataFrame to store SHAP values
shap_df = pd.DataFrame(columns=feature_names)

# Loop through each data point and calculate SHAP values
for i, row in X_test_scaled.iterrows():
    shap_values = explainer.shap_values(row)

    # Append SHAP values as a new row to the DataFrame
    shap_df = shap_df.append(pd.Series(shap_values[0], index=feature_names), ignore_index=True)

# Export the DataFrame to an Excel file
shap_df.to_excel('shap_values_per_data_point_full.xlsx', index=False, engine='openpyxl')



# Iterate through all possible combinations of two features to develop PPD plots
for feature1, feature2 in combinations(feature_names, 2):
    # Create a grid of values for the two features with reduced density for a sharper plot
    feature1_range = np.linspace(X_test_scaled[feature1].min(), X_test_scaled[feature1].max(), num=50)
    feature2_range = np.linspace(X_test_scaled[feature2].min(), X_test_scaled[feature2].max(), num=50)
    X, Y = np.meshgrid(feature1_range, feature2_range)

    # Initialize an empty array to store predicted values
    Z = np.zeros_like(X)

    # Iterate through the grid of feature values and predict for each combination
    for i in range(len(feature1_range)):
        for j in range(len(feature2_range)):
            # Copy the test data to keep it constant for other features
            X_test_copy = X_test_scaled.copy()

            # Set the feature values in the copied data
            X_test_copy[feature1] = X[i, j]
            X_test_copy[feature2] = Y[i, j]

            # Predict using the RELM
            predictions_test = rf_model.predict(X_test_copy) + gb_model.predict(X_test_copy)

            # Calculate the average prediction (you can choose other aggregation methods)
            Z[i, j] = predictions_test.mean()

    # Create a 3D plot with improved aesthetics
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with a higher resolution colormap, smoother shading, and reduced line width
    surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.9, rstride=2, cstride=2, linewidth=0.5, antialiased=True)

    # Add labels and a color bar with improved font settings
    ax.set_xlabel(feature1, fontsize=24, fontname='Arial')
    ax.set_ylabel(feature2, fontsize=24, fontname='Arial')
    ax.set_zlabel('Predicted Yield Strength (MPa)', fontsize=24, fontname='Arial')

    # Set the tick label font size
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)

    # Customize axis colors and thickness
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.set_linewidth(0.5)
    ax.yaxis.pane.set_linewidth(0.5)
    ax.zaxis.pane.set_linewidth(0.5)

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.05)
    cbar.ax.tick_params(labelsize=20)

    # Set the view to make the plot face outwards
    ax.view_init(elev=45, azim=240)

    # Save the 3D plot as a JPEG image with 600 DPI
    filename = f'3d_partial_dependence_plot_{feature1}_{feature2}_2.jpg'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    
    plt.close(fig)
