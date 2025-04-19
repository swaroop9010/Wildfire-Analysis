#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


df = pd.read_csv("/Users/saiswaroop/Downloads/California_Fire_Incidents.csv")
df.head()


# In[4]:


#summary statistics
df.describe()


# In[7]:


fires = pd.read_csv('/Users/saiswaroop/Downloads/Fires_pruned.csv')
fires.head()


# In[3]:


# Convert 'Started' to datetime and extract year
df['Started'] = pd.to_datetime(df['Started'], errors='coerce')
df['Year'] = df['Started'].dt.year

# Summarize total acres burned per year
acres_burned_per_year = df.groupby('Year')['AcresBurned'].sum().reset_index()

# Count the number of incidents per year
incidents_per_year = df.groupby('Year').size().reset_index(name='Incidents')

# Merge the summaries
trends = pd.merge(acres_burned_per_year, incidents_per_year, on='Year')

# Display the merged summary
trends


# In[4]:


# Displaying the column names to identify the correct column for fire causes
list(df.columns)


# In[5]:


# basic information
df.info()


# In[8]:


total_fires_per_year = fires['FIRE_YEAR'].value_counts()
total_fires_per_year.sort_index()


# In[7]:


# Revisiting the correlation matrix
corr


# In[9]:


total_fires_per_year.sort_index().plot(kind = 'bar', figsize = (10, 8));


# In[8]:


# Plotting the distribution of AcresBurned
plt.figure(figsize=(8, 4))
sns.histplot(df['AcresBurned'].dropna(), bins=50, kde=True, color='red')
plt.title('Distribution of Acres Burned in Wildfires')
plt.xlabel('Acres Burned (log scale)')
plt.ylabel('Frequency')
plt.xscale('log')
plt.show()


# In[10]:


total_fires_per_year.sort_index().plot();


# In[9]:


# Plotting the number of wildfires over the years
plt.figure(figsize=(11, 5))
sns.countplot(x='ArchiveYear', data=df, palette='inferno')
plt.title('Number of Wildfires per Year')
plt.xlabel('Year')
plt.ylabel('Number of Wildfires')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[10]:


# Plotting the geographical distribution of wildfires
plt.figure(figsize=(9, 5))
sns.scatterplot(x='Longitude', y='Latitude', data=df, hue='ArchiveYear', palette='cool', s=50, alpha=0.6)
plt.title('Geographical Distribution of Wildfires')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Plotting the total acres burned per year
df.groupby('ArchiveYear')['AcresBurned'].sum().plot(kind='bar', figsize=(8, 4), color='orange')
plt.title('Total Acres Burned per Year')
plt.xlabel('Year')
plt.ylabel('Total Acres Burned')
plt.xticks(rotation=45)
plt.show()


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the total number of wildfires by county
plt.figure(figsize=(8, 3))
df['Counties'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title('Top 10 Cities by Number of Wildfires')
plt.xlabel('Cities')
plt.ylabel('Number of Wildfires')
plt.xticks(rotation=45)
plt.show()


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/saiswaroop/Downloads/California_Fire_Incidents.csv'
df = pd.read_csv(file_path)


# Plotting the boxplot of AcresBurned and StructuresDestroyed
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[['AcresBurned', 'StructuresDestroyed']].dropna(), orient='h', palette='Set2')
plt.title('Boxplot of AcresBurned and StructuresDestroyed')
plt.xlabel('Value')
plt.show()


# In[11]:


# fires_bystate = fires.groupby(['STATE']).size().reset_index().rename(columns={0:'NUMBER_OF_FIRES'})
# fires_bystate
fires_bystate = fires['STATE'].value_counts().head(25)
fires_bystate


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score  # Include accuracy_score here

# Load the dataset
df = pd.read_csv("/Users/saiswaroop/Downloads/California_Fire_Incidents.csv")


# In[12]:


fires_bystate.plot(x = 'STATE', kind = 'bar', figsize=(100,100), fontsize=100)


# In[23]:


#Convert 'Started' to datetime and extract year if not already done
df['Started'] = pd.to_datetime(df['Started'], errors='coerce')
df['Year'] = df['Started'].dt.year

# Handle missing values
df.dropna(subset=['AcresBurned', 'Latitude', 'Longitude', 'Year'], inplace=True)

# Example of preparing data, transform 'AcresBurned' to reduce skewness
df['AcresBurned'] = np.log1p(df['AcresBurned'])


# In[4]:


# Plotting the correlation matrix to see relationships between numerical variables
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score  # Include accuracy_score here

# Load the dataset
df = pd.read_csv("/Users/saiswaroop/Downloads/California_Fire_Incidents.csv")


# In[25]:


#Convert 'Started' to datetime and extract year if not already done
df['Started'] = pd.to_datetime(df['Started'], errors='coerce')
df['Year'] = df['Started'].dt.year

# Handle missing values
df.dropna(subset=['AcresBurned', 'Latitude', 'Longitude', 'Year'], inplace=True)

# Example of preparing data, transform 'AcresBurned' to reduce skewness
df['AcresBurned'] = np.log1p(df['AcresBurned'])


# In[26]:


# Features and Target
feature_columns = ['Year', 'Latitude', 'Longitude']
X = df[feature_columns]
y = df['AcresBurned']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, lr_predictions))

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, rf_predictions))

# Assuming KNN is applicable (Note: typically used for classification)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train.astype(float), y_train.astype(int))  # Ensure data types are appropriate
knn_predictions = knn_model.predict(X_test.astype(float))
print("KNN Accuracy:", accuracy_score(y_test.astype(int), knn_predictions))


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, lr_predictions))


# In[29]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, rf_predictions))


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assuming KNN is applicable (Note: typically used for classification)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train.astype(float), y_train.astype(int))  # Ensure data types are appropriate
knn_predictions = knn_model.predict(X_test.astype(float))
print("KNN Accuracy:", accuracy_score(y_test.astype(int), knn_predictions))


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Features and Target
feature_columns = ['Year', 'Latitude', 'Longitude']
X = df[feature_columns]
y = df['AcresBurned']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
print("Linear Regression MSE:", lr_mse)#mean_squared_error
print("Linear Regression MAE:", lr_mae)#mean_absolute_error
print("Linear Regression R2 Score:", lr_r2)#import r2_score


# In[34]:


# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print("Random Forest MSE:", rf_mse)  #mean_squared_error
print("Random Forest MAE:", rf_mae)  #mean_absolute_error
print("Random Forest R2 Score:", rf_r2)  #import r2_score


# In[35]:


# K-Nearest Neighbors (KNN) Regressor
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_r2 = r2_score(y_test, knn_predictions)
print("KNN MSE:", knn_mse)#mean_squared_error
print("KNN MAE:", knn_mae)#mean_absolute_error
print("KNN R2 Score:", knn_r2)#import r2_score


# In[13]:


fires.STAT_CAUSE_CODE.unique()
fires.STAT_CAUSE_DESCR.unique()


# In[14]:


fires_cause = fires['STAT_CAUSE_DESCR'].value_counts()
fires_cause


# In[15]:


fires_cause.head(10).plot(label ='', title = 'Fire Cause', kind = 'pie', figsize = (25,20), autopct='%1.2f')


# In[36]:


import matplotlib.pyplot as plt

# Data for bar plots
models = ['Linear Regression', 'Random Forest', 'KNN']
mse_scores = [lr_mse, rf_mse, knn_mse]
mae_scores = [lr_mae, rf_mae, knn_mae]
r2_scores = [lr_r2, rf_r2, knn_r2]

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Bar plots for MSE
axes[0].bar(models, mse_scores, color='skyblue')
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].set_ylabel('MSE')

# Bar plots for MAE
axes[1].bar(models, mae_scores, color='lightgreen')
axes[1].set_title('Mean Absolute Error (MAE)')
axes[1].set_ylabel('MAE')

# Bar plots for R2
axes[2].bar(models, r2_scores, color='salmon')
axes[2].set_title('R-squared (R2) Score')
axes[2].set_ylabel('R2')

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()

