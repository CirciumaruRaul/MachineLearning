import pandas as pd 
import lightgbm as lgb 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

'''
  Loading the train data
'''
train = 'data/train.csv'
train_df = pd.read_csv(train)
print(train_df.isna().sum())
X_train, X_val, y_train, y_val = train_test_split(train_df.drop(columns = ["Activity"]), train_df['Activity'], test_size=0.2, random_state=13)

'''
  Initial look at the data
'''
missing_vals = train_df.isnull().sum().sum()
print(f"Total Missing Values: {missing_vals}")

if missing_vals > 0:
    print(train_df.isnull().sum()) 

duplicate_rows = train_df.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_rows}")

'''
  Distribution plotting
'''
sns.countplot(x='Activity', data=train_df)
plt.title("Class Distribution")
plt.show()

'''
  Plot distribution of features to acess the starting point
'''
train_df.iloc[:, 1:10].hist(figsize=(12, 8), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

'''
  Plotting the heatmap
'''
corr_matrix = train_df.iloc[:, 1:].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()

'''
  Scale the features
'''
minMaxScaler = MinMaxScaler()   
train_df_transformed = minMaxScaler.fit_transform(X_train)
val_df_transformed = minMaxScaler.transform(X_val)

'''
  Extract features via PCA/VT
'''
varThresholder = VarianceThreshold(threshold = 0.1)
pca = PCA(n_components=10)

train_df_thresh = varThresholder.fit_transform(train_df_transformed)
train_df_PCA = pca.fit_transform(train_df_transformed)

val_df_thresh = varThresholder.transform(val_df_transformed)
val_df_PCA = pca.transform(val_df_transformed)

accuracy = []

'''
  Tunning the models for each approach
'''

''' 
  LightGBM VARIANCE
'''
lgbm = lgb.LGBMClassifier()
param_grid = {
    'num_leaves': [20, 31, 40], 
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.5, 0.7, 0.9],
    'max_depth': [-1, 5, 10],
    'boosting_type': ['gbdt', 'dart'], 
    'objective': ['binary'],  
    'metric': ['binary_error']  
}



grid_search_lgb_thresh = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_lgb_thresh.fit(train_df_thresh, y_train)

# Print best parameters, score
print(f"Best parameters: {grid_search_lgb_thresh.best_params_}")
print(f"Best cross-validation score: {grid_search_lgb_thresh.best_score_}")

# Validation
y_pred_val_thresh_lgb = grid_search_lgb_thresh.best_estimator_.predict(val_df_thresh)
acc_thresh_lgb = accuracy_score(y_val, y_pred_val_thresh_lgb)
accuracy.append(acc_thresh_lgb)
print(f'Validation Accuracy: {acc_thresh_lgb}')

'''
  LightGBM PCA
'''
lgbm = lgb.LGBMClassifier()
param_grid = {
    'num_leaves': [20, 31, 40],  
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.5, 0.7, 0.9],
    'max_depth': [-1, 5, 10],
    'boosting_type': ['gbdt', 'dart'],
    'objective': ['binary'],
    'metric': ['binary_error'] 
}

grid_search_lgb_pca = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_lgb_pca.fit(train_df_PCA, y_train)

# Print best parameters, score
print(f"Best parameters: {grid_search_lgb_pca.best_params_}")
print(f"Best cross-validation score: {grid_search_lgb_pca.best_score_}")

# Validation
y_pred_val_pca_lgb = grid_search_lgb_pca.best_estimator_.predict(val_df_PCA)
acc_pca_lgb = accuracy_score(y_val, y_pred_val_pca_lgb)
accuracy.append(acc_pca_lgb)
print(f'Validation Accuracy: {acc_pca_lgb}')


'''
  ExtraTreesClassifier VARIANCE
'''
etc = ExtraTreesClassifier(random_state=13)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4], 
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False],  
    'criterion': ['gini', 'entropy']  
}

grid_search_etc_thresh = GridSearchCV(estimator=etc, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_etc_thresh.fit(train_df_thresh, y_train)

# Print best parameters, score
print(f"Best parameters: {grid_search_etc_thresh.best_params_}")
print(f"Best cross-validation score: {grid_search_etc_thresh.best_score_}")

# Validation
y_pred_val_etc_thresh = grid_search_etc_thresh.best_estimator_.predict(val_df_thresh)
acc_thresh_etc = accuracy_score(y_val, y_pred_val_etc_thresh)
accuracy.append(acc_thresh_etc)
print(f'Validation Accuracy: {acc_thresh_etc}')

'''
  ExtraTreesClassifier PCA
''' 
etc = ExtraTreesClassifier(random_state=13)
param_grid = {
    'n_estimators': [50, 100, 200], 
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False],  
    'criterion': ['gini', 'entropy']  
}

grid_search_pca_etc = GridSearchCV(estimator=etc, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_pca_etc.fit(train_df_PCA, y_train)

# Print best parameters, score
print(f"Best parameters: {grid_search_pca_etc.best_params_}")
print(f"Best cross-validation score: {grid_search_pca_etc.best_score_}")

# Validation
y_pred_val_etc_pca = grid_search_pca_etc.best_estimator_.predict(val_df_PCA)
acc_pca_etc = accuracy_score(y_val, y_pred_val_etc_pca)
accuracy.append(acc_pca_etc)
print(f'Validation Accuracy: {acc_pca_etc}')

'''
  Plotting results with obtained scores
'''

results_lgb_thresh = pd.DataFrame(grid_search_lgb_thresh.cv_results_)
results_lgb_pca = pd.DataFrame(grid_search_lgb_pca.cv_results_)

results_etc_thresh = pd.DataFrame(grid_search_etc_thresh.cv_results_)
results_etc_pca = pd.DataFrame(grid_search_pca_etc.cv_results_)

# Sort by mean test score
results_lgb_thresh = results_lgb_thresh.sort_values(by="mean_test_score", ascending=False)
results_lgb_pca = results_lgb_pca.sort_values(by="mean_test_score", ascending=False)

results_etc_thresh = results_etc_thresh.sort_values(by="mean_test_score", ascending=False)
results_etc_pca = results_etc_pca.sort_values(by="mean_test_score", ascending=False)

results_light = {
    "LightGBM - Variance": results_lgb_thresh,
    "LightGBM - PCA": results_lgb_pca
}
results_etc = {
    "ExtraTreesClassifier - Variance": results_etc_thresh,
    "ExtraTreesClassifier - PCA": results_etc_pca
}

fig, axes = plt.subplots(1, 2, figsize=(15, 8)) 

for ax, (name, df) in zip(axes, results_light.items()):
    df["mean_test_score"] = df["mean_test_score"].astype(float)
    sns.lineplot(data=df, x="param_num_leaves", y="mean_test_score", marker="o", ax=ax)
    ax.set_title(f"{name} - Mean Test Score vs Number of Leaves")
    ax.set_xlabel("Number of Leaves")
    ax.set_ylabel("Mean Test Score")
    ax.grid(True)
    
plt.tight_layout() 
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 8)) 

for ax, (name, df) in zip(axes, results_etc.items()):
    if "param_n_estimators" in df.columns:
        df["param_n_estimators"] = df["param_n_estimators"].astype(float)  
        sns.lineplot(data=df, x="param_n_estimators", y="mean_test_score", marker="o", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel("Mean Test Score")
        ax.grid(True)
        
plt.tight_layout()
plt.show()

'''
  Plotting confusion matrix
'''
predictions = {
    "LightGBM - Variance": y_pred_val_thresh_lgb,
    "LightGBM - PCA": y_pred_val_pca_lgb,
    "ExtraTreesClassifier - Variance": y_pred_val_etc_thresh,
    "ExtraTreesClassifier - PCA": y_pred_val_etc_pca
}

fig, axes = plt.subplots(2, 2, figsize=(15, 8)) 
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()