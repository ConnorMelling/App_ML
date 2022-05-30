#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import chart_studio.plotly

# Plotly Packages
import chart_studio.plotly.plotly as py
from plotly import tools
#import plotly.plotly as py
#import plotly.figure_factory as ff
from plotly import graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)

# Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

# Statistical Libraries
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy import stats


# Regression Modeling
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# Other Libraries
import warnings
warnings.filterwarnings("ignore")


# In[27]:


insurance = pd.read_csv("/Users/Connor/Documents/Uni 2nd year/app ml/insurance.csv")
#getting a sample of the data
sample = insurance.head()
#Desc of dataset
desc = insurance.describe()
#dtypes in dataset
dtypes = insurance.dtypes
insurance


# In[5]:


#getting the dist of the charges column
charge_dist = insurance["charges"].values

#Getting the log dist of the charges column
logcharge = np.log(insurance["charges"])

#creating trace for the dist
dist_trace = go.Histogram(
    x=charge_dist,
    histnorm='probability',
    name="Charges Distribution",
    marker = dict())

#creating trace for log dist
log_trace = go.Histogram(
    x=logcharge,
    histnorm='probability',
    name="Log Distribution Charges",
    marker = dict())

#displying the charts & formatting
plot = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=('Charge Distribution','Log Distribution'),
                         print_grid=False)


plot.append_trace(dist_trace, 1, 1)
plot.append_trace(log_trace, 2, 1)

plot['layout'].update(showlegend=True, title='Distribution', bargap=0.05)
iplot(plot)


# In[6]:


plt.figure(figsize=(12,4))
sns.heatmap(insurance.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[7]:


# correlation plot
corr = insurance.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);


# In[15]:


# Dummy variable
categorical_columns = ['sex','children', 'smoker', 'region']
insurance_encode = pd.get_dummies(data = insurance, prefix = 'DVT', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
# Lets verify the dummay variable process
print('Columns in original data frame:\n',insurance.columns.values)
print('\nNumber of rows and columns in the dataset:',insurance.shape)
print('\nColumns in data frame after encoding the dummy variable trap:\n',insurance_encode.columns.values)
print('\nNumber of rows and columns in the dataset:',insurance_encode.shape)


# In[16]:


from scipy.stats import boxcox
y_bc,lam, ci= boxcox(insurance_encode['charges'],alpha=0.05)

#df['charges'] = y_bc  
# it did not perform better for this model, so log transform is used
ci,lam


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[20]:


insurance[['sex', 'smoker', 'region']] = insurance[['sex', 'smoker', 'region']].astype('category')
insurance.dtypes


# In[30]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(insurance.sex.drop_duplicates())
insurance.sex = label.transform(insurance.sex)
label.fit(insurance.smoker.drop_duplicates())
insurance.smoker = label.transform(insurance.smoker)
label.fit(insurance.region.drop_duplicates())
insurance.region = label.transform(insurance.region)
insurance.dtypes


# In[23]:


f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(insurance.corr(), annot=True, cmap='cool')


# In[34]:


from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x = insurance.drop(['charges'], axis = 1)
y = insurance['charges']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)
print("Intercept :", Lin_reg.intercept_)
print("Coef :", Lin_reg.coef_)
print("Rsquared :", Lin_reg.score(x_test, y_test))


# In[37]:


from sklearn.linear_model import Ridge
Ridge = Ridge(alpha=0.5)
Ridge.fit(x_train, y_train)
print("Intercept :", Ridge.intercept_)
print("Coef :", Ridge.coef_)
print("Rsquared :" ,Ridge.score(x_test, y_test))


# In[38]:


from sklearn.linear_model import Lasso
Lasso = Lasso(alpha=0.2, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
              tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
Lasso.fit(x_train, y_train)
print("Intercept :", Lasso.intercept_)
print("Coef :", Lasso.coef_)
print("Rsquared :", Lasso.score(x_test, y_test))


# In[41]:


from sklearn.preprocessing import PolynomialFeatures
x = insurance.drop(['charges', 'sex', 'region'], axis = 1)
y = insurance.charges
pol = PolynomialFeatures (degree = 2)
x_pol = pol.fit_transform(x)
x_train, x_test, y_train, y_test = holdout(x_pol, y, test_size=0.2, random_state=0)
Pol_reg = LinearRegression()
Pol_reg.fit(x_train, y_train)
y_train_pred = Pol_reg.predict(x_train)
y_test_pred = Pol_reg.predict(x_test)
print("Intercept :", Pol_reg.intercept_)
print("Coef :", Pol_reg.coef_)
print("Rsquared :",Pol_reg.score(x_test, y_test))


# In[42]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[43]:


##Predicting the charges
y_test_pred = Pol_reg.predict(x_test)
##Comparing the actual output values with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
df


# In[44]:


df.head()


# In[ ]:




