#!/usr/bin/env python
# coding: utf-8

# Load Dataset

# In[1]:


import pandas as pd

df = pd.read_csv("telco_customer_churn.csv")
df.head()


# In[2]:


df.info()


# Understanding the data

# In[3]:


import numpy as np

for column in df:
    unique_vals = np.unique(df[column].fillna('0'))
    nr_values = len(unique_vals)
    if nr_values <= 12:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


# In[4]:


#checking null values

df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.describe(include=object)


# In[7]:


#remove unnecessary columns


df = df.drop(columns=['customerID'])

df.head()


# In[8]:


#Spit data
from sklearn.model_selection import train_test_split

target = ['Churn']
cat_var = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
num_var = ['tenure', 'MonthlyCharges', 'TotalCharges']
df['tenure'] = df['tenure'].astype(float)
df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
df['TotalCharges'] = df['tenure'].astype(float)

X = df.drop(columns=target) 
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#check the shape of the data


# In[10]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Data Visualize

# In[9]:


from matplotlib import pyplot as plt
import seaborn as sb

# Count Plot of our Y - Check the balance of the dataset
plt.figure(figsize=(8, 6))
sb.countplot(data=df, x="Churn")
plt.title("Count Plot of Churn", fontsize=16)
plt.xlabel("Churn", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks([0, 1], labels=["Not Churn", "Churn"])
plt.show()

y_train.value_counts(normalize=True)


# In[10]:


#distribution of cat_var
plt.figure(figsize=(20, 15))

for i, col in enumerate(cat_var):
    plt.subplot(6,3,i+1)
    if len(X_train[col].unique()) <=5:
        sb.countplot(data=X_train, x = col, alpha = 0.5)
        plt.ylabel("Count", fontdict={'fontsize':16})
    else:
        sb.countplot(df=X_train, y=col, order=X_train[col].value_counts().index, alpha=0.5)
    plt.title(f"""Distribution of {col}""", fontdict={'fontsize':22})
    plt.xlabel
    plt.ylabel

plt.tight_layout()
plt.savefig("Categorical Distribution")
plt.show()


# In[11]:


#investigate target by churn
for f in df:
    plt.figure()
    ax = sb.countplot(data=df, x=f, hue="Churn", palette="Set1")


# In[12]:


df.head()


# In[13]:


#indetify numerical values

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
n_variables = df.select_dtypes(include=numerics).columns

# Increases the size of sns plots
sb.set(rc={'figure.figsize':(8,5)})

for c in n_variables:    
    x = df[c].values
    ax = sb.boxplot(x, color = '#D1EC46')
    print('The meadian is: ', df[c].median())
    plt.title(c)
    plt.show()


# In[14]:


X_train[num_var].head()


# In[15]:


#corelation matrix

corr= X_train[num_var].corr()

plt.figure(figsize=(25, 12))
mask = np.tril(np.ones_like(corr))
sb.heatmap(corr, annot=True, fmt="2f", mask=mask, square=True)
plt.show()


# In[16]:


print(df["Churn"].value_counts())


# In[17]:


import plotly.graph_objects as go

# Combine features and target
df_train = pd.concat([X_train, y_train], axis=1)

# Ensure the target variable is binary numeric
if df_train['Churn'].dtype == object:
    df_train['Churn'] = df_train['Churn'].map({'Yes': 1, 'No': 0})

# Loop through categorical variables
for col in cat_var:
    if col in df_train.columns:
        plt.figure(figsize=(12, 5))
        if len(X_train[col].unique()) < 4:
            sb.histplot(binwidth=0.5, x=col, hue=df_train['Churn'], data=df_train, stat="count", multiple="stack", bins=10)
        else:
            sb.histplot(binwidth=0.5, x=col, hue=df_train['Churn'], data=df_train)
        plt.title(f"""Distribution of {col} vs Churn""", fontdict={'fontsize':22})
        plt.show()

        print('Overall frequency: ')
        display(X_train[col].value_counts(normalize=True))

        # Compute odds ratio
        unique_values = df_train[col].unique()
        odds_dict = {}
        for unique_val in unique_values:
            is_churned = len(df_train[(df_train["Churn"] == 1) & (df_train[col] == unique_val)])
            not_churned = len(df_train[(df_train["Churn"] == 0) & (df_train[col] == unique_val)])
            odds_dict[unique_val] = round((is_churned / (not_churned + 1e-6)) * 100, 2)

        odds_df = pd.Series(odds_dict, name=f'{col}_odds_ratio - Churn/Not Churn (%)').reset_index() \
            .rename(columns={'index': 'Categories'})

        # Display table using Plotly
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(odds_df.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[odds_df[val] for val in odds_df.columns], fill_color='lavender', align='left'))
        ])
        fig.update_layout(autosize=False)
        fig.show()

        print('*' * 50)


# In[18]:


X_train.head()


# In[19]:


from sklearn.preprocessing import OneHotEncoder

cat_val = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

encoder = OneHotEncoder(sparse_output=False)
cat_mat=encoder.fit_transform(X_train[cat_val])
cat_df = pd.DataFrame(cat_mat,
                          columns=encoder.get_feature_names_out(),
                          index=X_train.index)

X_train = X_train.drop(columns=cat_val)
X_train = pd.concat([X_train, cat_df], axis=1)


# In[20]:


X_train.info()


# In[21]:


test_cat_mat = encoder.transform(X_test[cat_val])
test_cat_df = pd.DataFrame(test_cat_mat,
                               columns=encoder.get_feature_names_out(),
                               index=X_test.index)

X_test=X_test.drop(columns=cat_val)
X_test= pd.concat([X_test, test_cat_df], axis=1)

X_test.head()


# In[27]:


#corelation matrix

corr= X_train.corr()
plt.figure(figsize=(100, 52))
mask = np.tril(np.ones_like(corr))
sb.heatmap(corr, annot=True, fmt="2f", mask=mask, square=True)
plt.show()


# In[26]:


#corelation matrix

corr= X_test.corr()
plt.figure(figsize=(50, 23))
mask = np.tril(np.ones_like(corr))
sb.heatmap(corr, annot=True, fmt="2f", mask=mask, square=True)
plt.show()


# In[24]:


from sklearn.preprocessing import MinMaxScaler

#minmax scaller
sc = MinMaxScaler()
X_trainSc = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
X_testSc =  pd.DataFrame(sc.fit_transform(X_test), columns=X_test.columns)

X_trainSc.head()


# Oversample Using SMOTE

# In[25]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(
    sampling_strategy='auto',
    random_state=0,
    k_neighbors=5
)

X_res, y_res = sm.fit_resample(X_trainSc, y_train)


# Training AI

# In[27]:


# Convert labels
y_res_num = y_res['Churn'].map({'No': 0, 'Yes': 1})
y_test_num = y_test['Churn'].map({'No': 0, 'Yes': 1})

# To collect results from both models
results = []


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Random Forest tuning and evaluation...")

# Define parameters for tuning
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

# GridSearchCV
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_res, y_res_num)

# Best model
rf = rf_grid.best_estimator_

# Predictions
pred_train_rf = rf.predict(X_res)
pred_test_rf = rf.predict(X_testSc)

# Accuracy scores
train_accuracy = accuracy_score(y_res_num, pred_train_rf)
test_accuracy = accuracy_score(y_test_num, pred_test_rf)

# Plot accuracy comparison
metrics = ['Train Accuracy', 'Test Accuracy']
values = [train_accuracy, test_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color='skyblue')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.4f}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', fontsize=12)

plt.ylim(0, 1.05)
plt.title("Random Forest Accuracy Comparison")
plt.ylabel("Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Evaluation metrics
f1 = f1_score(y_test_num, pred_test_rf, pos_label=1)
recall = recall_score(y_test_num, pred_test_rf, pos_label=1)
precision = precision_score(y_test_num, pred_test_rf, pos_label=1)



metrics = ['F1 Score', 'Recall', 'Precision']
values = [f1, recall, precision]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['steelblue', 'orange', 'green'])
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', fontsize=12)

plt.ylim(0, 1.05)
plt.title("Random Forest Evaluation Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Best Parameters:", rf_grid.best_params_)




# In[42]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

print("XGBoost tuning and evaluation...")

# Parameters grid
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.2]
}

# GridSearchCV
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    xgb_params, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_res, y_res_num)

# Best model
xgb = xgb_grid.best_estimator_

# Predictions
pred_train_xgb = xgb.predict(X_res)
pred_test_xgb = xgb.predict(X_testSc)

# Accuracy scores
train_accuracy = accuracy_score(y_res_num, pred_train_xgb)
test_accuracy = accuracy_score(y_test_num, pred_test_xgb)

# Plot accuracy comparison
metrics = ['Train Accuracy', 'Test Accuracy']
values = [train_accuracy, test_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color='skyblue')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.4f}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', fontsize=12)

plt.ylim(0, 1.05)
plt.title("XGBoost Accuracy Comparison")
plt.ylabel("Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Evaluation metrics
f1 = f1_score(y_test_num, pred_test_xgb, pos_label=1)
recall = recall_score(y_test_num, pred_test_xgb, pos_label=1)
precision = precision_score(y_test_num, pred_test_xgb, pos_label=1)

metrics = ['F1 Score', 'Recall', 'Precision']
values = [f1, recall, precision]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['steelblue', 'orange', 'green'])
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', fontsize=12)

plt.ylim(0, 1.05)
plt.title("XGBoost Evaluation Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Best Parameters:", xgb_grid.best_params_)


# In[35]:


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

cat_clf = CatBoostClassifier(random_state=42, verbose=0, train_dir='/tmp/catboost_info')


# In[48]:


print("CatBoost tuning and evaluation...")

# CatBoost parameters grid
cat_params = {
    "iterations": [100, 200],
    "depth": [3, 5],
    "learning_rate": [0.1, 0.2]
}

cat_clf = CatBoostClassifier(random_state=42, verbose=0)

grid_search = GridSearchCV(
    cat_clf,
    param_grid=cat_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    error_score='raise'  # raise error instead of silent failure
)

grid_search.fit(X_res, y_res_num)

best_cat = grid_search.best_estimator_

# Predictions
pred_train_cat = best_cat.predict(X_res)
pred_test_cat = best_cat.predict(X_testSc)

# Accuracy scores
train_accuracy = accuracy_score(y_res_num, pred_train_cat)
test_accuracy = accuracy_score(y_test_num, pred_test_cat)

# Plot accuracy comparison
metrics = ['Train Accuracy', 'Test Accuracy']
values = [train_accuracy, test_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color='skyblue')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.4f}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', fontsize=12)

plt.ylim(0, 1.05)
plt.title("CatBoost Accuracy Comparison")
plt.ylabel("Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Evaluation metrics
f1 = f1_score(y_test_num, pred_test_cat, pos_label=1)
recall = recall_score(y_test_num, pred_test_cat, pos_label=1)
precision = precision_score(y_test_num, pred_test_cat, pos_label=1)

metrics = ['F1 Score', 'Recall', 'Precision']
values = [f1, recall, precision]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['steelblue', 'orange', 'green'])
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height), 
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', fontsize=12)

plt.ylim(0, 1.05)
plt.title("CatBoost Evaluation Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Best Parameters:", grid_search.best_params_)


# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define metric labels
metrics = ['Precision', 'Recall', 'F1 Score']

# Collect metric values in a DataFrame
scores_df = pd.DataFrame({
    'Random Forest': [
        precision_score(y_test_num, pred_test_rf, pos_label=1),
        recall_score(y_test_num, pred_test_rf, pos_label=1),
        f1_score(y_test_num, pred_test_rf, pos_label=1)
    ],
    'XGBoost': [
        precision_score(y_test_num, pred_test_xgb, pos_label=1),
        recall_score(y_test_num, pred_test_xgb, pos_label=1),
        f1_score(y_test_num, pred_test_xgb, pos_label=1)
    ],
    'CatBoost': [
        precision_score(y_test_num, pred_test_cat, pos_label=1),
        recall_score(y_test_num, pred_test_cat, pos_label=1),
        f1_score(y_test_num, pred_test_cat, pos_label=1)
    ]
}, index=metrics)

# Plotting
x = np.arange(len(metrics))  # label locations
width = 0.25  # width of bars

plt.figure(figsize=(10, 6))
plt.bar(x - width, scores_df['Random Forest'], width, label='Random Forest', color='skyblue')
plt.bar(x, scores_df['XGBoost'], width, label='XGBoost', color='orange')
plt.bar(x + width, scores_df['CatBoost'], width, label='CatBoost', color='green')

# Add score labels
for i in range(len(metrics)):
    plt.text(x[i] - width, scores_df['Random Forest'][i] + 0.01, f"{scores_df['Random Forest'][i]:.2f}", ha='center')
    plt.text(x[i], scores_df['XGBoost'][i] + 0.01, f"{scores_df['XGBoost'][i]:.2f}", ha='center')
    plt.text(x[i] + width, scores_df['CatBoost'][i] + 0.01, f"{scores_df['CatBoost'][i]:.2f}", ha='center')

plt.xticks(x, metrics)
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[49]:


import joblib

# Save the best XGBoost model
joblib.dump(xgb, 'best_xgb_model.pkl')
print("Model saved as 'best_xgb_model.pkl'")

