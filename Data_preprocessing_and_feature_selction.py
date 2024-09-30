import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import random

# Load the dataset
data = pd.read_csv("your_csv_file.csv")

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Descriptive statistics
print("Descriptive statistics:\n", data.describe())

# Histogram of target variable
sns.set_style('white')
plt.figure(figsize=(10, 8))
sns.histplot(data['target'], kde=False, color = 'blue', edgecolor='black')
boxprops = dict(linewidth=2)
plt.xticks(fontname="Times New Roman", fontsize=24)
plt.yticks(fontname="Times New Roman", fontsize=24)
plt.xlabel("Target", fontname="Times New Roman", fontsize=27)
plt.ylabel("Counts", fontname="Times New Roman", fontsize=27)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)

ax = plt.gca()
ax.tick_params(axis="both", which="major", width=3)
sns.despine()
plt.savefig('histogram_YS.png', dpi=1000, bbox_inches='tight')

# Create a grid of sub scatter plots
plt.figure(figsize=(12, 8))
sns.set_style('white')
data1 = data.drop('target', axis=1)
features = data1.columns
num_rows = 4  
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for idx, feature in enumerate(features):
    row = idx // num_cols
    col = idx % num_cols
    ax = axes[row, col]

    # Create box plot for the current feature
    boxplot = ax.boxplot(data1[feature], patch_artist=True, boxprops=dict(facecolor='blue'))

    # Set the color of the whiskers and caps
    for whisker in boxplot['whiskers']:
        whisker.set(color='black')
    for cap in boxplot['caps']:
        cap.set(color='black')

    ax.set_xticklabels([feature], fontname="Times New Roman", fontsize=18)
    ax.set_yticklabels([], fontname="Times New Roman", fontsize=18)
 
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax.tick_params(axis="both", which="major", width=2)

plt.tight_layout() 
plt.savefig('boxplots_all_features.png', dpi=600)
plt.show()

# Calculate skewness for each feature
skewness = data1.skew()
print(skewness)

# Apply Yeo-Johnson transformation to descriptors
cols_to_transform = ['Hmix','Smix','delta','Vm','E','K','G','VEC','Tm','rho','X_P','Omega','eta','E_LD','E_C','F_PN']
pt = PowerTransformer(method='yeo-johnson', standardize=True)
data[cols_to_transform] = pt.fit_transform(data[cols_to_transform])
print(data.head())

plt.figure(figsize=(12, 8))
sns.set_style('white')
data1 = data.drop('target', axis=1)
features = data1.columns
num_rows = 4  
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for idx, feature in enumerate(features):
    row = idx // num_cols
    col = idx % num_cols
    ax = axes[row, col]

    # Create box plot for the current feature
    boxplot = ax.boxplot(data1[feature], patch_artist=True, boxprops=dict(facecolor='aquamarine'))

    # Set the color of the whiskers and caps
    for whisker in boxplot['whiskers']:
        whisker.set(color='black')
    for cap in boxplot['caps']:
        cap.set(color='black')

    ax.set_xticklabels([feature], fontname="Times New Roman", fontsize=18)
    ax.set_yticklabels([], fontname="Times New Roman", fontsize=18)
 
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax.tick_params(axis="both", which="major", width=2)

plt.tight_layout()  
plt.savefig('boxplots_transformed_all_features.png', dpi=600)
plt.show()

# Calculate skewness for each transformed feature
skewness = data1.skew()
print(skewness)

# Create the pairplot with custom styling
sns.set_style('white')
plt.figure(figsize=(12, 12))
sns.set(font_scale=1.2, rc={'lines.linewidth': 1.5})
plot = sns.pairplot(data, diag_kind='hist', diag_kws={'color': 'maroon'}, plot_kws={'color': 'maroon'})

# Set black borders for the plot
plot = plot.map_upper(plt.scatter, edgecolor="k", s=10)
plot = plot.map_lower(sns.kdeplot, colors="k", linewidths=0.5)
plot = plot.map_diag(sns.histplot, color="maroon", edgecolor="k")

# Increase font size for all axes and tick labels
for ax in plot.axes.flatten():
    ax.tick_params(labelsize=20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(24)

plt.show()


# Feature Selection

### Spearman Rank Correlation

# Calculate Spearman rank correlation matrix
corr_matrix_spearman, _ = spearmanr(data)
sns.heatmap(corr_matrix_spearman, annot=False, cmap='rainbow', vmin=-1, vmax=1)
plt.title("Spearman Rank Correlation Matrix Heatmap")
plt.show()


### Mutual Importance
X = data.drop('target', axis=1)
y = data['target']

mi_scores = mutual_info_regression(X, y)
for i in range(len(mi_scores)):
    print('Feature %d: %.3f' % (i+1, mi_scores[i]))


### RF based Feature Selection

model = RandomForestRegressor(random_state=0)
model.fit(X, y)
importances = model.feature_importances_


### GB based Feature Selection

gbr = GradientBoostingRegressor(random_state=0)
gbr.fit(X, y)
importances = gbr.feature_importances_


### Permutation Importance

model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Calculate baseline Mean Squared Error (MSE) on the test set
y_pred_baseline = model.predict(X_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)

# Calculate permutation feature importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# Calculate normalized importance scores
normalized_importance = perm_importance.importances_mean / mse_baseline

# Create a DataFrame to display the results with feature names
df_importance = pd.DataFrame({'Feature': X.columns, 'Importance': normalized_importance})

# Sort the DataFrame by importance score
df_importance = df_importance.sort_values(by='Importance', ascending=False)

# Print the feature importance
print(df_importance)


### Recursive Feature Elimination

model = RandomForestRegressor()

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=model)
fit = rfe.fit(X, y)

# Get the rankings of features
feature_rankings = fit.ranking_

# Create a DataFrame to display the rankings with feature names
df_rankings = pd.DataFrame({'Feature': X.columns, 'Ranking': feature_rankings})

# Sort the DataFrame by ranking (1 = most important, higher = less important)
df_rankings = df_rankings.sort_values(by='Ranking')

# Print the feature rankings
print(df_rankings)


### Backward Elimination

selected_features = list(X.columns)

# Define a stopping criterion
stopping_criterion = 0.01

# Initialize lists to store results
mse_values = []
num_features_selected = []

while len(selected_features) > 1:
    # Fit the model using the selected features
    model.fit(X_train[selected_features], y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test[selected_features])
    
    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Store the current MSE and number of selected features
    mse_values.append(mse)
    num_features_selected.append(len(selected_features))
    
    # Store the current MSE and selected features
    best_mse = mse
    best_features = selected_features.copy()
    
    # Iterate over each feature and check its impact on performance
    for feature in selected_features:
        # Temporarily remove the current feature
        temp_features = selected_features.copy()
        temp_features.remove(feature)
        
        # Fit the model with the updated set of features
        model.fit(X_train[temp_features], y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test[temp_features])
        
        # Calculate the Mean Squared Error (MSE)
        temp_mse = mean_squared_error(y_test, y_pred)
        
        # If removing the feature improves performance, update best_mse and best_features
        if temp_mse < best_mse:
            best_mse = temp_mse
            best_features = temp_features
    
    # Check if removing a feature improved performance beyond the stopping criterion
    if (mse - best_mse) < stopping_criterion:
        break
    
    # Update selected_features with the best_features
    selected_features = best_features

# Print the final selected features
print("Selected Features:", selected_features)

# Plot the MSE values against the number of selected features
plt.figure(figsize=(10, 6))
plt.plot(be_results_df['Number of Features'], be_results_df['MSE'], marker='o', linestyle='-')
plt.xlabel('Number of Features')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. Number of Features')
plt.grid(True)
plt.show()


### Genetic Algorithm Feature Selection

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the fitness function for the genetic algorithm
def fitness_function(selected_features):
    # Train a Random Forest Regressor model and return its performance
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled[:, selected_features], y_train)
    y_pred = model.predict(X_test_scaled[:, selected_features])
    mse = mean_squared_error(y_test, y_pred)
    return -mse

# Genetic Algorithm parameters
population_size = 16
num_generations = 1000
mutation_rate = 0.2

# Initialize population randomly
population = [random.sample(range(X_train.shape[1]), 5) for _ in range(population_size)]

# Initialize lists to store information across generations
best_fitness_scores = []
average_fitness_scores = []
best_individuals = []

for generation in range(num_generations):
    # Evaluate fitness of each individual in the population
    fitness_scores = [fitness_function(individual) for individual in population]
    
    # Record statistics for this generation
    best_fitness_scores.append(max(fitness_scores))
    average_fitness_scores.append(np.mean(fitness_scores))
    best_individuals.append(population[np.argmax(fitness_scores)])
    
    # Select parents for the next generation using tournament selection
    selected_parents = []
    for _ in range(population_size):
        tournament_indices = np.random.choice(len(population), size=5, replace=False)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        selected_parents.append(population[tournament_indices[np.argmin(tournament_scores)]])
    
    # Create the next generation through crossover and mutation
    new_population = []
    for parent1, parent2 in zip(selected_parents[::2], selected_parents[1::2]):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(child1) - 1)
            child1[mutation_point] = random.randint(0, X_train.shape[1] - 1)
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(child2) - 1)
            child2[mutation_point] = random.randint(0, X_train.shape[1] - 1)
        
        new_population.extend([child1, child2])
    
    population = new_population

# Find the best individual (subset of features) in the final population
best_individual = max(population, key=fitness_function)

# Print the best features
print("Best features:", best_individual)

# Plot the fitness scores
plt.figure(figsize=(10, 5))
plt.plot(best_fitness_scores, label='Best Fitness')
plt.plot(average_fitness_scores, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.title('Genetic Algorithm Fitness Evolution')
plt.legend()
plt.grid(True)
plt.show()

# Create a matrix to store selected features' presence across generations
selected_features_matrix = np.zeros((num_generations, X_train.shape[1]))

# Fill in the matrix with selected feature presence information
for generation, individual in enumerate(best_individuals):
    for feature in individual:
        selected_features_matrix[generation, feature] = 1

# Plot selected features' presence across generations
plt.figure(figsize=(10, 6))
plt.imshow(selected_features_matrix.T, cmap='gray', aspect='auto', interpolation='nearest')
plt.xlabel('Generation')
plt.ylabel('Feature Index')
plt.title('Selected Features Evolution')
plt.colorbar(label='Selected')
plt.grid(False)
plt.show()


### Best-Subset Based Feature Selection

num_features = X.shape[1]
all_feature_indices = list(range(num_features))

# Iterate over all possible feature combinations
for subset_size in range(1, num_features + 1):
    # Generate all possible combinations of feature indices
    feature_combinations = list(itertools.combinations(all_feature_indices, subset_size))
    
    for feature_indices in feature_combinations:
        # Select the subset of features
        X_subset = X_train_scaled[:, feature_indices]
        
        # Train the regression model on the subset of features
        model.fit(X_subset, y_train)
        
        # Make predictions on the test set
        X_test_subset = X_test_scaled[:, feature_indices]
        y_pred = model.predict(X_test_subset)
        
        # Evaluate the model's performance using Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        
        # Add the results to the DataFrame
        feature_subset = ",".join([str(i) for i in feature_indices])
        results_df = results_df.append({"Feature Subset": feature_subset, "Mean Squared Error": mse}, ignore_index=True)

# Extract the number of features in each subset
num_features_in_subset = [len(subset.split(",")) for subset in results_df["Feature Subset"]]

# Extract the MSE values
mse_values = results_df["Mean Squared Error"]

# Create a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(num_features_in_subset, mse_values, marker='o', s=30, alpha=0.5)
plt.title("MSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()
