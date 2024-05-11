#!/usr/bin/env python
# coding: utf-8

# ## Quantitative variables:
# known as numerical variables, represent quantities or measurements that can be expressed as numerical values
# ## Categorical Variables:
# represent qualitative characteristics or attributes that can be divided into distinct categories or groups. These variables do not have a natural order or numeric value associated with them
# 
# this code used to detrmine is data is quantitave or categorical

# In[37]:


import pandas as pd
# Read the CSV file into a DataFrame
dataset = pd.read_csv("F:/Breast_cancer_data.csv",sep=",")
# Dictionary to store variable types
variable_types = {}

for column in dataset.columns:
    values = dataset[column].unique()

    if dataset[column].dtype in ['int64', 'float64']:
        variable_types[column] = 'quantitative'
    else:
        variable_types[column] = 'categorical'

for column, variable_type in variable_types.items():
    print(f"Column '{column}' is {variable_type}")


# ## what is an outliers?
# an outliers is datapoint indata set that is distant from other observations.A data point that lies outside the overall distribution of the dataset
# 

# In[38]:


import numpy as np
from scipy.stats import zscore
z_scores = zscore(dataset)
threshold = 3.0
outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

mask = np.zeros(len(dataset), dtype=bool)
mask[outlier_indices] = True

data = dataset.loc[~mask]
data.reset_index(drop=True, inplace=True)


# ### this code plot the data of one column of my data set before removing outliers and after removing it to show that outliers is acutal removing

# In[39]:


import matplotlib.pyplot as plt

# Visualize data before removing outliers
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(dataset['mean_radius'], bins=20)
plt.title('Histogram Before Removing Outliers')

plt.subplot(1, 2, 2)
plt.hist(data['mean_radius'], bins=20)
plt.title('Histogram After Removing Outliers')

plt.tight_layout()
plt.show()


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# View dimensions of dataset

# In[41]:


data.shape


# Preview the dataset

# In[42]:


data.head()


# *View* summary of datase

# In[43]:


data.info()


# In[44]:


data1 = data.iloc[:,:-1]
print(data1)


# ### Measuring of Central Tendency
# *   Mean
# *   Median
# *   Mode
# 

# In[45]:


mean =data1.mean()
print(mean)


# In[46]:


median =data1.median()
print(median)


# In[47]:


mode= data1.mode()
print(mode)


# #Summary statistics of numerical columns

# In[48]:


data1.describe().transpose()


# ### Measuring of Dispersion or Variability
# *   Range
# *   Variance
# *   Standard deviation
# *   Correlation
# *   Covariance

# In[49]:


maxx= data1.max()
minn = data1.min()
rangee = maxx - minn
print(rangee)


# In[50]:



data_var = data1.var()

print(data_var)


# In[51]:



# compute the std
data_std = data1.std()

print(data_std)


# In[52]:



# compute the correlation
data_corr= data1.corr()

print(data_corr)


# In[53]:



# compute the covariance
data_cov = data1.cov()

print(data_cov)


# ## Standerization of Data
# 

# In[54]:


# Standardize the features using the calculated mean and standard deviation
# data["mean_radius"]=(data["mean_radius"]-data["mean_radius"].mean())/data["mean_radius"].std()
# data["mean_texture"]=(data["mean_texture"]-data["mean_texture"].mean())/data["mean_texture"].std()
# data["mean_perimeter"]=(data["mean_perimeter"]-data["mean_perimeter"].mean())/data["mean_perimeter"].std()
# data["mean_area"]=(data["mean_area"]-data["mean_area"].mean())/data["mean_area"].std()
# data["mean_smoothness"]=(data["mean_smoothness"]-data["mean_smoothness"].mean())/data["mean_smoothness"].std()

data_A=data
data=(data-data.mean())/data.std()
data["diagnosis"]=data_A["diagnosis"]

# Print the standardized data
data.head(10)


# #• Split the data randomly into 2 partitions with a 80%-20% proportion:
# 
# 
# *  The 80% partition is called the training data.
# 
# *  The 20% partition is called the testing data.
# 
# 

# In[55]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2)
# print the sizes of the resulting datasets
print("Training set size:", len(train_set))
print("Testing set size:", len(test_set))


# ### Plot the histogram/distribution For each feature/column in the training data.

# In[56]:


dic = {
    "mean_radius"  : "normal",
    "mean_texture"   :"normal",
    "mean_perimeter"  :"normal",
    "mean_area"       :"normal",
    "mean_smoothness"  :"normal",
    "diagnosis" : " "
}


# In[57]:


for col in train_set.columns:
    # Create a histogram of the column's values
    plt.hist(train_set[col], bins=10)

    # Add a title to the plot
    plt.title(f"{dic[col]} distribution" )


    # Add x and y axis labels
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


# ## Hypothesis Test

# In[58]:


#Statistically test if a feature/column is normally distribute:

import pandas as pd
from scipy.stats import kstest
# Define the null and alternative hypotheses
null_hypothesis = "The column is normally distributed."
alternative_hypothesis = "The column is not normally distributed."

#  iterate over each column and perform the kolmogorov-smirnov
for column in data.columns:
    # Extract the column values
    column_values = data[column]
    # Perform the kolmogorov-smirnov test for normality
    test_statistic, p_value = kstest(column_values,"norm")
   # print(p_value)

    # Compare the p-value with the chosen significance level
    alpha = 0.05
    if p_value < alpha:
        print(f"Column '{column}': Reject the null hypothesis.")
        print(alternative_hypothesis)
    else:
        print(f"Column '{column}': Fail to reject the null hypothesis.")


# ## Plot the conditional distributions of each feature on each target class (label).

# In[59]:


#Plot the conditional distributions of each feature on each target class (label).
import matplotlib.pyplot as plt
import pandas as pd


# Iterate over each column and plot the conditional distribution based on the diagnosis (0 or 1).
for column in data.columns:
    if column != 'diagnosis':
        # Split the data based on the diagnosis (0 or 1)
        diag_0 = data[data['diagnosis'] == 0][column]
        diag_1 = data[data['diagnosis'] == 1][column]

        # Plot the histograms
        plt.figure()
        plt.hist(diag_0, bins=10, alpha=0.5, label='class 0')
        plt.hist(diag_1, bins=10, alpha=0.5, label='class 1')
        plt.title(f'Conditional Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


# In[60]:


data.head(19)


# ## Naive Bayes
# 

# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set_style("darkgrid")
import scipy
from scipy import stats


# In[62]:


data.head(10)  #data after standerization


# ## Check  Features Dependent or Not

# In[63]:



#computing the correlation coefficient between each two variable
#get correlation of data to show strength of relationship between its features
#this is important because for the naive bayes to be applied, the variables must be independet
corr = data.iloc[:,:-1].corr(method="pearson")
cmap=sns.diverging_palette(250,354,80,60,center="dark",as_cmap=True)
sns.heatmap(corr,vmax=1,vmin=-0.5,cmap=cmap,square=True,linewidths=0.2)

#from the result we can find that mean_radius,mean_perimeter ,mean_area are dependent so we take only one of them so we can apply NB


# In[64]:


data = data[["mean_radius", "mean_texture", "mean_smoothness", "diagnosis"]] #take only mean_radius
data.head(10)


# ## Show Normal Distribution of  the Features
# 

# In[65]:


fig,axes=plot.subplots(1,3,figsize=(18,6))  #to confirm all data is normally distributed
sns.histplot(data,ax=axes[0],x='mean_radius',kde=True,color='r')
sns.histplot(data,ax=axes[1],x='mean_texture',kde=True,color='b')
sns.histplot(data,ax=axes[2],x='mean_smoothness',kde=True,color='g')


# ## Calculate Prior Probability

# In[66]:


def calculate_prior(df, Y): #get probability of each value in diagonsis(prior probability)
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:

       #print(df[df[Y]==i])
        prior.append(len(df[df[Y]==i])/len(df))
    return prior


# <h2> Now we are going to calculate conditional probability of observing a specific value x for a feature X, given that the instance belongs to a particular class y <h2>
# <h4>The Conditional Probability formula using Gaussian distribution is give as:<h4>
#     <h3> \[
# P(X=x|Y=y) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
# \]<h3>

# In[67]:


def calculate_likehood(df,feat_name,feat_value,Y,label):
    df=df[df[Y]==label]
    mean,std=df[feat_name].mean(),df[feat_name].std()

    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_value-mean)**2 / (2 * std**2 )))

    return p_x_given_y


#  ##  Calculate 𝑃(𝑋=𝑥1|𝑌=𝑦)⋅𝑃(𝑋=𝑥2|𝑌=𝑦)⋯𝑃(𝑋=𝑥𝑛|𝑌=𝑦)⋅𝑃(𝑌=𝑦) For all Y Classes and Find The Maximum probability

# In[68]:


def naive_bayes(df,X,Y):
    features=list(df.columns)[:-1]
    prior=calculate_prior(df,Y)
    Y_pred=[]

    for x in X:
        labels=sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        #likelihood=[]
        for j in labels:
            for i in range(len(features)):
                likelihood[j]*=calculate_likehood(df,features[i],x[i],Y,j)
        post_prob = [1]*len(labels)

        for j in range(len(labels)):
            post_prob[j]=prior[j]*likelihood[j]

        Y_pred.append(np.argmax(post_prob)) #returns the max value from array

    return np.array(Y_pred)


# ## Testing the model

# In[69]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=.2, random_state=41)
X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values

Y_pred = naive_bayes(train, X=X_test, Y="diagnosis")

from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))


# ## Check The Accuracy

# In[70]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y_pred)
print("accuaracy = ",accuracy*100 ,"%")


# ## Testing The Model using GaussianNB

# In[71]:


from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data['diagnosis'], test_size=0.2, random_state=41)

nb=GaussianNB()   #calssifier naive bayes
nb.fit(X_train,y_train)   #training the data
sklearn_prediction=nb.predict(X_test)
sklearn_accuracy=accuracy_score(y_test,sklearn_prediction)
sklearn_accuracy
print(f"confusion matrix={confusion_matrix (y_test,sklearn_prediction)}")
print("accuaracy = ",sklearn_accuracy*100 ,"%")

