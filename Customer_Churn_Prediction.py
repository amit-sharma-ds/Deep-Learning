#!/usr/bin/env python
# coding: utf-8

# # Import Essential Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler     ## Standardization Technique


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Import Warning Library
import warnings
warnings.filterwarnings("ignore")


# # Read the Dataset

# In[2]:


df = pd.read_csv("Churn_Prediction.csv")


# In[3]:


# Display First 5 rows of the dataset
df.head()


# In[4]:


# Shape of the dataset
df.shape


# In[5]:


# Check Dtype and Missing Values
df.info()


# # Feature Engineering

# # Data Cleaning

# In[6]:


# Check Missing Values
df.isnull().sum()


# In[7]:


# Check Duplicated Values
df.duplicated().sum()


# # EDA (Exploratory Data Analysis)

# In[8]:


# Drop irrelevant columns for analysis
df_clean = df.drop(columns=["RowNumber", "CustomerId", "Surname"])


# In[9]:


# Convert categorical columns to numeric for correlation
df_clean = pd.get_dummies(df_clean,columns = ['Geography','Gender'], drop_first=True)


# In[10]:


df_clean


# In[11]:


# Correlation heatmap
plt.figure(figsize=(12,8))

sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.show()


# # Check Distribution of the dataset

# In[12]:


numeric_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns


# In[13]:


for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.distplot(df_clean[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# # Data Preprocessing

# In[15]:


# Separate Independent and Dependent Variable
X = df_clean.drop('Exited', axis=1)   
y = df_clean['Exited']                

# Apply standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame (for easy reading)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# In[16]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Build the ANN Model

# In[17]:


model = Sequential([
    Dense(3, activation = 'sigmoid',input_dim = 11),
    Dense(1, activation = 'sigmoid'),
    ])


# In[18]:


model.summary()


# In[19]:


# Import Adam Optimizer for gradient algorithms
from tensorflow.keras.optimizers import Adam


# In[20]:


# Compile the model
model.compile(loss = 'binary_crossentropy',optimizer = 'Adam')

# Train the model
model.fit(X_train, y_train, epochs=20, validation_split = 0.2)


# In[21]:


# for Layer 1 weights and bias 
model.layers[0].get_weights()


# In[22]:


# for layers 2 weights and bias
model.layers[1].get_weights()


# In[23]:


y_log = model.predict(X_test)


# In[24]:


y_pred = np.where(y_log>0.5,1,0)


# In[25]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# # How can we Improve our accuracy score

# In[66]:


# 1. Hidden Layer main Activation Function main 'Relu' rakho.
# 2. You can increase No of epochs e.g. 100 , 1000
# 3. You can increase no of nodes
# 4. You can increase no of hidden layer


# In[93]:


model = Sequential([
    Dense(11, activation = 'relu',input_dim = 11),
    Dense(11, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
    ])


# In[94]:


model.summary()


# In[95]:


# Compile the model
model.compile(loss = 'binary_crossentropy',optimizer = 'Adam',metrics = ['accuracy'])


# In[96]:


# Train the model
history = model.fit(X_train, y_train, epochs=200,validation_split = 0.2)


# In[97]:


y_log = model.predict(X_test)


# In[98]:


y_pred = np.where(y_log>0.5,1,0)


# In[99]:


accuracy_score(y_test,y_pred)


# In[1]:


# Hidden Layer = ReLU, Tanh
# Output Layer (binary classification) = Sigmoid
# Output layer (Multi-Class classification) = Softmax
# Output Layer (regression) = No Activation fun


# # Steps in Forward propagation

# In[2]:


# 1. Multiply inputs by weights
# 2. Add bias
# 3. Apply activation fun (Tanh, ReLU)
# 4. pass output to next layer
# 5. Get final output


# # Steps in Backward Propagation

# In[3]:


# 1.Calculate the error at output
# 2. Find how much each weight contributed to the error (using derivatives)
# 3. Update weights and bias to minimize error
# 4. Repaet over many iterations (epochs) to improve accuracy


# # Inshort Forward & Backward Propagation

# In[ ]:


# predict output from input
# # Learn by reducing error

