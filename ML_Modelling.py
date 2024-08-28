#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from itertools import permutations
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")


# ### Data input and preprocessing 

# In[2]:


# load dataset
data = pd.read_csv('regression_complete_for_SHAP.csv')


# In[3]:


# split into features and target variable
X = data.drop('YS', axis=1)
y = data ['YS']


# In[4]:


# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


#scale the dataset
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


# In[6]:


X_train_scaled = X_train
X_test_scaled = X_test


# In[7]:


X_test_scaled.to_excel('X_test_scaled.xlsx', index=False, engine='openpyxl')


# In[8]:


print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of X_test_scaled:", X_test_scaled.shape)


# ### Bayesian optimization of Hyperparameters with Model Training and Prediction

# In[9]:


models = {
    "Random Forest": (RandomForestRegressor(), {
        'n_estimators': (10, 1000),
        'max_depth': (2, 50),
        'min_samples_split': (2, 50),
        'min_samples_leaf': (1, 30),
        'max_features': (0.1, 1.0),
        'bootstrap': [True, False],
        'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
        'min_impurity_decrease': (0.0, 0.1),  
        'min_weight_fraction_leaf': (0.0, 0.5),  
        'max_leaf_nodes': (10, 200),
        'ccp_alpha': (0.0, 0.2)
    }),
    "SVR": (SVR(), {
        'C': (0.001, 300.0),                        
        'gamma': ('scale', 'auto'),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }),
    "Lasso Regression": (Lasso(), {
        'alpha': (0.00000001, 1.0),                     
        'selection': ['cyclic', 'random']
    }),
    "Ridge Regression": (Ridge(), {
        'alpha': (0.00000001, 10.0),                     
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }),
    "Elastic Net": (ElasticNet(), {
        'alpha': (0.00000001, 10.0),                     
        'l1_ratio': (0.00001, 0.99),                    
        'selection': ['cyclic', 'random']
    }),
    "K-Nearest Neighbors": (KNeighborsRegressor(), {
        'n_neighbors': (2, 150),                      
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }),
    "Gradient Boosting": (GradientBoostingRegressor(), {
        'n_estimators': (10, 500),
        'learning_rate': (0.0000001, 1.0),              
        'max_depth': (1, 60),                        
        'min_samples_split': (2, 50),                               
        'max_features': (0.1, 1.0),  
    })
}


# In[54]:


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

top_five_results = []

prediction_uncertainties = {}

for name, (model, search_space) in models1.items():
    # Perform hyperparameter tuning using Bayesian optimization
    model = BayesSearchCV(model, search_spaces=search_space, n_iter=50, n_jobs=-1, cv=5, random_state=42)
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
    
    # Append the results to the top_five_results list
    top_five_results.append({
        'Model': name,
        'Best Hyperparameters': model.best_params_,
        'MSE': mse,
        'R2 Score': r2,
        'MAE': mae
    })    
    
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
    
    # Save the figure
    fig_filename = f"{name}_actual_vs_predicted.png"
    plt.savefig(fig_filename, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Keep track of the best performing model
    if mse < best_score:
        best_score = mse
        best_model = model.best_estimator_

top_five_results.sort(key=lambda x: x['MSE'])
top_five_results = top_five_results[:5]

import csv

with open('top_five_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Model', 'Best Hyperparameters', 'MSE', 'R2 Score', 'MAE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in top_five_results:
        writer.writerow(result)

# Print the top five results to the console
for i, result in enumerate(top_five_results):
    print(f"Top {i+1} Model: {result['Model']}")
    print(f"Best Hyperparameters: {result['Best Hyperparameters']}")
    print(f"MSE: {result['MSE']}")
    print(f"R2 Score: {result['R2 Score']}")
    print(f"MAE: {result['MAE']}")
    print()


# In[55]:


# Initialize the residuals dictionary to store errors for each model
residuals = {}

# Define a list of colors for scatter plots
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
color_index = 0

# Set font properties
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
    
    # Save the figure
    fig_filename = f"{name}_residual_histogram_final_dataset.png"
    plt.savefig(fig_filename, bbox_inches='tight')
    
    plt.show()


# In[13]:


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


# In[14]:


search_space = {
    'hidden_layers': (1, 5),             
    'neurons_per_layer': (10, 100),          
    'activation': ['relu', 'sigmoid', 'tanh'],        
    'optimizer': [Adam],                      
    'learning_rate': (0.00001, 0.1, 'log-uniform'),  
    'batch_size': (8, 128)                   
}


# In[15]:


opt = BayesSearchCV(
    NeuralNetworkRegressor(),
    search_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    optimizer_kwargs={'base_estimator': 'GP'},
    verbose=1
)


# In[16]:


opt.fit(X_train_scaled, y_train)


# In[17]:


best_params_nn = opt.best_params_


# In[18]:


model_NN = NeuralNetworkRegressor(**best_params_nn)
print('Best Hyperparameters:', best_params_nn)


# In[19]:


model_NN.fit(X_train_scaled, y_train)


# In[20]:


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


# In[21]:


loss_history = model_NN.loss_history

# Create a DataFrame from the loss history
loss_df = pd.DataFrame(loss_history, columns=['Loss'])
excel_filename = 'loss_history_final_dataset.xlsx'

# Save the DataFrame to an Excel file
loss_df.to_excel(excel_filename, index=False)
print(f"Loss history exported to {excel_filename}")


# In[22]:


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


# In[23]:


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

# Save the figure
fig_filename = "NN_actual_vs_predicted_final_dataset.png"
plt.savefig(fig_filename, bbox_inches='tight')
plt.show()


# In[24]:


# Initialize the residuals dictionary to store errors for each model
residuals_NN = {}

# Set font properties
font_prop = {'family': 'Arial', 'size': 28}

residuals_NN = y_test - y_pred_NN

# Define a list of colors for scatter plots
plt.figure(figsize=(10, 9))
plt.hist(residuals_NN, bins=20, alpha=0.5, label='Neural Network', color='b')
plt.xlabel('Residuals', fontdict=font_prop)
plt.ylabel('Frequency', fontdict=font_prop)
plt.title('Histogram of Residuals', fontdict=font_prop)
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.legend()

# Save the figure
fig_filename = "NN_residual_histogram_final_dataset.png"
plt.savefig(fig_filename, bbox_inches='tight')

plt.show()


# In[56]:


df = pd.DataFrame.from_dict(predictions)
df.to_excel('predictions_final_dataset.xlsx', index=False)

df1 = pd.DataFrame.from_dict(predictions_train)
df1.to_excel('predictions_train_final_dataset.xlsx', index=False)

df2 = pd.DataFrame.from_dict(residuals)
df2.to_excel('residuals_final_dataset.xlsx', index=False)

df3 = pd.DataFrame.from_dict(residuals_NN)
df3.to_excel('residuals_NN_final_dataset.xlsx', index=False)

df4 = pd.DataFrame.from_dict(y_pred_NN)
df4.to_excel('predictions_NN_final_dataset.xlsx', index=False)

df5 = pd.DataFrame.from_dict(y_pred_NN_train)
df5.to_excel('predictions_NN_train_final_dataset.xlsx', index=False)


# In[57]:


# Create a DataFrame with the y_test values
df6 = pd.DataFrame({'Actual': y_test})

# Export the DataFrame to an Excel file
df6.to_excel('y_test_final_dataset.xlsx', index=False)

# Create a DataFrame with the y_test values
df7 = pd.DataFrame(y_train)

# Export the DataFrame to an Excel file
df7.to_excel('y_train_final_dataset.xlsx', index=False)


# In[46]:


# Create the hybrid model
hybrid_model = best_model

# Train the hybrid model with the two best performing models
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()


# In[47]:


# Fit the Random Forest model on the training data with the best hyperparameters
rf_model.fit(X_train_scaled, y_train)


# In[48]:


# Fit the Gradient Boosting model on the residuals of the Random Forest predictions with the best hyperparameters
rf_predictions = rf_model.predict(X_test_scaled)
residuals = y_test - rf_predictions
gb_model.fit(X_test_scaled, residuals)


# In[49]:


# Make predictions with the hybrid model
rf_predictions_test = rf_model.predict(X_test_scaled)
hybrid_predictions_test = rf_predictions_test + gb_model.predict(X_test_scaled)

rf_predictions_train = rf_model.predict(X_train_scaled)
hybrid_predictions_train = rf_predictions_train + gb_model.predict(X_train_scaled)


# In[50]:


# Compute the MSE, R2 score, and MAE of the hybrid model
mse_hybrid = mean_squared_error(y_test, hybrid_predictions_test)
r2_hybrid = r2_score(y_test, hybrid_predictions_test)
mae_hybrid = mean_absolute_error(y_test, hybrid_predictions_test)

print(f"Hybrid Model MSE: {mse_hybrid}")
print(f"Hybrid Model R2 Score: {r2_hybrid}")
print(f"Hybrid Model MAE: {mae_hybrid}")


# In[51]:


# Compute the MSE, R2 score, and MAE of the hybrid model
mse_hybrid_train = mean_squared_error(y_train, hybrid_predictions_train)
r2_hybrid_train = r2_score(y_train, hybrid_predictions_train)
mae_hybrid_train = mean_absolute_error(y_train, hybrid_predictions_train)

print(f"Hybrid Model MSE: {mse_hybrid_train}")
print(f"Hybrid Model R2 Score: {r2_hybrid_train}")
print(f"Hybrid Model MAE: {mae_hybrid_train}")


# In[52]:


# Plot actual vs predicted values for the hybrid model

font_prop = {'family': 'Arial', 'size': 28}

color_index = 0
scatter_point_size = 150
scatter_alpha = 0.7

squared_diff = (y_test - hybrid_predictions_test) ** 2

variance = np.var(squared_diff)

# Plot actual vs predicted values
plt.figure(figsize=(10, 9))
plt.scatter(y_train, hybrid_predictions_train, color='b', label=name, s=scatter_point_size, alpha=scatter_alpha)
plt.scatter(y_test, hybrid_predictions_test, color='r', label=name, s=scatter_point_size, alpha=scatter_alpha)
plt.plot([100, 1200], [100, 1200], 'k--')
color_index = (color_index + 1) % len(scatter_colors)
plt.xlabel("Actual Yield Strength (MPa)", fontdict=font_prop)
plt.ylabel("Predicted Yield Strength (MPa)", fontdict=font_prop)
plt.xlim([100, 1200])
plt.ylim([100, 1200])
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)

# Save the figure
fig_filename = f"{name}_actual_vs_predicted.png"
plt.savefig(fig_filename, bbox_inches='tight')
plt.show()


# In[34]:


residuals_hybrid = {}
residuals_hybrid = y_test - hybrid_predictions_test

# Define a list of colors for scatter plots
plt.figure(figsize=(10, 9))
plt.hist(residuals_hybrid, bins=20, alpha=0.5, label='Neural Network', color='b')
plt.xlabel('Residuals', fontdict=font_prop)
plt.ylabel('Frequency', fontdict=font_prop)
plt.title('Histogram of Residuals', fontdict=font_prop)
plt.xticks(fontproperties='Arial', fontsize=20)
plt.yticks(fontproperties='Arial', fontsize=20)
plt.legend()

# Save the figure
fig_filename = "hybrid_residual_histogram_final_dataset.png"
plt.savefig(fig_filename, bbox_inches='tight')

plt.show()


# In[35]:


df6 = pd.DataFrame.from_dict(residuals_hybrid)
df6.to_excel('residuals_hybrid_final_dataset.xlsx', index=False)


# In[36]:


df7 = pd.DataFrame.from_dict(hybrid_predictions_test)
df7.to_excel('hybrid_predictions_final_dataset.xlsx', index=False)


# In[37]:


df8 = pd.DataFrame.from_dict(hybrid_predictions_train)
df8.to_excel('hybrid_predictions_train_final_dataset.xlsx', index=False)


# ## Generate new compositions

# In[ ]:


# Load data
data = pd.read_csv('final_comp_data.csv')

# Preprocess data
compositions = data.iloc[:, :13].values
yield_strength = data.iloc[:, 13].values

compositions_normalized = compositions

scaler_yield = MinMaxScaler()
yield_normalized = scaler_yield.fit_transform(yield_strength.reshape(-1, 1))


# In[ ]:


# Generator model
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='tanh'))
    return model

# Discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Encoder model
def build_encoder(input_dim, latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(latent_dim))
    return model

# Compile GAN
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    z = Input(shape=(latent_dim,))
    alloy = generator(z)
    discriminator.trainable = False
    validity = discriminator(alloy)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return combined


# In[ ]:


latent_dim = 2
output_dim = 13
epochs = 1000
batch_size = 64
sample_interval = 1000

generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)
gan = build_gan(generator, discriminator)
encoder = build_encoder(output_dim, latent_dim)

# Train the GAN
half_batch = int(batch_size / 2)

d_losses = []
g_losses = []

for epoch in range(epochs):
    idx = np.random.randint(0, compositions_normalized.shape[0], half_batch)
    real_compositions = compositions_normalized[idx]
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    fake_compositions = generator.predict(noise)
    
    d_loss_real = discriminator.train_on_batch(real_compositions, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_compositions, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.array([1] * batch_size)
    g_loss = gan.train_on_batch(noise, valid_y)
    
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# Autoencoder for reconstruction
real_compositions_input = Input(shape=(output_dim,))
encoded_compositions = encoder(real_compositions_input)
reconstructed_compositions = generator(encoded_compositions)

autoencoder = Model(real_compositions_input, reconstructed_compositions)
autoencoder.compile(loss='mse', optimizer=Adam(0.0002, 0.5))

# Train the autoencoder and capture loss values
autoencoder_history = autoencoder.fit(compositions_normalized, compositions_normalized, epochs=1000, batch_size=batch_size, verbose=1)

# Project known compositions into the latent space
latent_vectors = encoder.predict(compositions_normalized)
print("Latent vectors for known compositions:")
print(latent_vectors)

# Plot the autoencoder training loss
plt.figure(figsize=(10, 5))
plt.plot(autoencoder_history.history['loss'], label='Autoencoder Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('los_epoch_VAE.jpg', dpi=600)
plt.show()

# Plot the latent space and save the figure
plt.figure(figsize=(10, 7))
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=yield_strength, cmap='viridis', s=100)
plt.colorbar(label='Yield Strength')
plt.title('2D Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.savefig('latent_space_plot.jpg', dpi=600)
plt.show()


# In[ ]:


# Generate 10,000 new compositions
num_samples = 10000
noise = np.random.normal(0, 1, (num_samples, latent_dim))
new_compositions = generator.predict(noise)

# Denormalize the new compositions
new_compositions_denormalized = scaler_compositions.inverse_transform(new_compositions)

# Export to Excel
new_compositions_df = pd.DataFrame(new_compositions_denormalized, columns=[f'Element_{i+1}' for i in range(output_dim)])
new_compositions_df.to_excel('generated_compositions.xlsx', index=False)

print("Generated compositions exported to 'generated_compositions.xlsx'")


# In[ ]:


# Project the generated compositions into the latent space
generated_latent_vectors = encoder.predict(new_compositions)


# In[ ]:


# Plot the latent space and save the figure
plt.figure(figsize=(10, 7))
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=yield_strength, cmap='viridis', s=0, alpha=0.5, label='Known Compositions')
plt.scatter(generated_latent_vectors[:, 0], generated_latent_vectors[:, 1], c='red', s=20, alpha=0.5, label='Generated Compositions')
plt.colorbar(label='Yield Strength')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.savefig('latent_space_plot_with_generated.jpg', dpi=600)
plt.show()


# ## Prediction of new alloys with processing parameters

# In[38]:


# load new compositions dataset
X_data_pred = pd.read_csv('data_for_prediction.csv')


# In[39]:


rf_predictions_pred = rf_model.predict(X_data_pred)
hybrid_predictions_pred = rf_predictions_pred + gb_model.predict(X_data_pred)


# In[40]:


df15 = pd.DataFrame.from_dict(hybrid_predictions_pred)
df15.to_excel('new_predictions_with_processing_parameters.xlsx', index=False)


# In[41]:


X_data_pred_1 = pd.read_csv('AlCoCrFeMnNi_compositions_to_predict_new_alloys.csv')
rf_predictions_pred = rf_model.predict(X_data_pred_1)
hybrid_predictions_pred = rf_predictions_pred + gb_model.predict(X_data_pred_1)
df15 = pd.DataFrame.from_dict(hybrid_predictions_pred)
df15.to_excel('new_predictions_AlCoCrFeMnNi.xlsx', index=False)


# In[42]:


X_data_pred_2 = pd.read_csv('AlCoCrFeMnNiSi_compositions_to_predict_new_alloys.csv')
rf_predictions_pred = rf_model.predict(X_data_pred_2)
hybrid_predictions_pred = rf_predictions_pred + gb_model.predict(X_data_pred_2)
df15 = pd.DataFrame.from_dict(hybrid_predictions_pred)
df15.to_excel('new_predictions_AlCoCrFeMnNiSi.xlsx', index=False)


# In[43]:


# Read data from Excel file
file_path = 'YS_compositions_to_predict_new_alloys_1.xlsx'
sheet_name = 'Sheet2'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract input columns (assuming columns are named 'X', 'Y', 'Z')
X = data['X'].values
Y = data['Y'].values
Z = data['Z'].values
# Extract output column (assuming the output column is named 'Output')
Output = data['Output'].values

# Create a grid for contour plot
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
zi = np.linspace(Z.min(), Z.max(), 100)
xi, yi, zi = np.meshgrid(xi, yi, zi)

# Interpolate the data onto the grid
from scipy.interpolate import griddata
output_grid = griddata((X, Y, Z), Output, (xi, yi, zi), method='linear')

# Plot the contour plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create contour plot
contour = ax.contourf(xi, yi, zi, output_grid, cmap='viridis')
fig.colorbar(contour, ax=ax, label='Output')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Color Contour Plot')

plt.show()


# ### Explaining the ML Models

# In[ ]:


import shap

feature_names = X_train_scaled.columns.tolist()

explainer = shap.Explainer(gb_model)
shap_values = explainer(X_train_scaled)

# Define the maximum number of features to display
max_display = 32  

plt.figure(figsize=(12, 12))
shap.summary_plot(shap_values, X_train_scaled, plot_type='bar', color = 'b', feature_names=feature_names, max_display=max_display)
plt.tight_layout()
plt.savefig('shap_bar_full.png', dpi=600, bbox_inches='tight')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, X_train_scaled)
plt.savefig('shap_real_full.jpg', dpi=600, bbox_inches='tight')


# In[ ]:


shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_names, plot_type='violin', show=False)
plt.savefig('shap_violin_full.png', dpi=600, bbox_inches='tight')
plt.show()


# In[ ]:


# Create an empty DataFrame to store SHAP values
shap_df = pd.DataFrame(columns=feature_names)

# Loop through each data point and calculate SHAP values
for i, row in X_test_scaled.iterrows():
    shap_values = explainer.shap_values(row)
    
    # Apply threshold to SHAP values
    #shap_values[shap_values > 1] = 1
    #shap_values[shap_values < -1] = -1
   
    # Append SHAP values as a new row to the DataFrame
    shap_df = shap_df.append(pd.Series(shap_values[0], index=feature_names), ignore_index=True)

# Export the DataFrame to an Excel file
shap_df.to_excel('shap_values_per_data_point_full.xlsx', index=False, engine='openpyxl')


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

feature1 = 'Time '
feature2 = 'PD'

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
        
        # Predict using the hybrid model (or any other model you want)
        predictions_test = rf_model.predict(X_test_copy) + gb_model.predict(X_test_copy)
        
        # Calculate the average prediction (you can choose other aggregation methods)
        Z[i, j] = predictions_test.mean()

# Create a 3D plot with improved aesthetics
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a higher resolution colormap, smoother shading, and reduced line width
surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8, rstride=2, cstride=2, linewidth=0.5, antialiased=True)

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
ax.view_init(elev=45, azim=55)

# Save the 3D plot as a JPEG image with 600 DPI
plt.savefig('3d_partial_dependence_plot_full_Time_PD.jpg', dpi=600, bbox_inches='tight')

# Show the 3D plot
plt.show()


# In[ ]:


from itertools import combinations

# Iterate through all possible combinations of two features
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

            # Predict using the hybrid model (or any other model you want)
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


# In[ ]:




