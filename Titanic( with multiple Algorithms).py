#!/usr/bin/env python
# coding: utf-8

# # Step 1- Exploratory Data Analysis

# In[1]:


import pandas as pd
import seaborn as sb


# In[2]:


df_train = pd.read_csv('titanic_datasets/train.csv')
df_train.head()


# In[3]:


df_test = pd.read_csv('titanic_datasets/test.csv')
df_test.head()


# In[4]:


df_test.info()


# In[5]:


df_train.info()


# # Cabin

# In[6]:


df_train.drop(['Cabin'],axis=1,inplace=True) #'Cabin' has more than 70 percent missing data
df_test.drop(['Cabin'],axis=1,inplace=True)


# In[7]:


df_train.head(2)


# In[8]:


df_test.head(2)


# # PassengerId

# In[9]:


df_train['PassengerId']


# # Survived

# In[10]:


import seaborn as sb
sb.countplot(df_train['Survived'])


# In[11]:


df_train['Survived'].value_counts() #342 people survised out of 891 which is 38% survival rate


# # Pclass

# In[12]:


sb.countplot(df_train['Pclass'], hue = df_train['Survived'])


# In[13]:


df_train.groupby('Pclass').mean()['Survived'] # value closure to 1 means more survived form that class


# In[14]:


#1. Passengers in Pclass 1 (Upper class) are more likely to survive.
#2. Pclass is a good feature for prediction of survival.


# # Name

# In[15]:


df_train['Name']


# # Sex

# In[16]:


sb.countplot(df_train['Sex'], hue = df_train['Survived'])


# In[17]:


df_train.groupby('Sex').mean()['Survived']


# In[18]:


#observations
#1. Proportion of male and female: ~2/3 vs ~1/3

#2. Male is much less likely to survive, with only 20% chance of survival. For female, >70% chance of survival.

#3. Obviously, Sex is an important feature to predict survival.


# # Age

# In[19]:


from matplotlib import pyplot as plt


# In[20]:


plt.hist(df_train['Age'],edgecolor='black',bins=16) #using matplotlib
plt.xlabel('Age')
plt.ylabel('count')
plt.show()


# In[21]:


sb.histplot(df_train['Age'],edgecolor='black',bins=16) #using seaborn
plt.xlabel('Age')
plt.ylabel('count')
plt.show()


# In[22]:


sb.boxplot(x=df_train['Survived'], y= df_train['Age'])


# In[23]:


#observations
#1. Passengers are mainly aged 20–40.

#2. Younger passengers tends to survive.


# # SibSp

# In[24]:


sb.countplot(df_train['SibSp'],hue=df_train['Survived'])


# In[25]:


df_train.groupby('SibSp').mean()['Survived']


# In[26]:


sb.barplot(x=df_train['SibSp'], y=df_train['Survived'])
plt.show()


# In[27]:


#observations
#1. Most of the passengers travel with 1 sibling/spouse.

#2. Passengers having 1 sibling/spouse are more likely to survive compared to those not.

#3. For those more than 1 siblings/spouses, the information is insufficient to provide any insight.


# # Parch

# In[28]:


sb.countplot(df_train['Parch'],hue=df_train['Survived'])


# In[29]:


df_train.groupby('Parch').mean()['Survived']


# In[30]:


#observations
#1. >70% passengers travel without parents/children.
#2. Passengers travelling with parents/children are more likely to survive than those not.


# # Ticket

# In[31]:


df_train['Ticket']


# # Fare

# In[32]:


sb.boxplot(x=df_train['Survived'], y=df_train['Fare'])


# In[33]:


#observations
#The distribution is right-skewed. Outliers are observed.
#For those who survived, their fares are relatively higher.


# # Embarked

# In[34]:


sb.countplot(df_train['Embarked'])


# In[35]:


df_train.groupby('Embarked').mean()['Survived']


# In[36]:


sb.barplot(x='Embarked', y='Survived', data=df_train)
plt.show()


# In[37]:


#observations
#1. >2/3 passengers embarked at Port S.
#2. Passengers embarked at Port C are more likely to survive.


# # Step 2- Imputation of Missing Data/ Outliers

# In[38]:


df_train.info()


# In[39]:


#Age and Embarked columns has the missing data


# # Age

# In[40]:


df_train['Name'].head(10)


# In[41]:


tittles = df_train['Name'].apply(lambda X: X.split(' ')[1]).value_counts()
tittles


# In[42]:


titles = df_test['Name'].apply(lambda X: X.split(' ')[1]).value_counts()[:4].index
titles


# In[43]:


#The common titles are(Mr/Miss/Mrs/Master). Some of the titles (Ms/Lady/Sir…etc.) can be grouped to the common titles. 
#The remaining unclassified titles can be grouped to “Others”.


# In[44]:


df_train['Title'] = df_train['Name'].apply(lambda X:X.split(' ')[1])
df_test['Title'] = df_test['Name'].apply(lambda X:X.split(' ')[1])

df_train['Title'] = df_train['Title'].apply(lambda X: X if X in titles else 'other')
df_test['Title'] = df_test['Title'].apply(lambda X: X if X in titles else 'other')


# In[45]:


df_train.head()


# In[46]:


sb.countplot(df_train['Title'], hue=df_train['Survived'])


# In[47]:


sb.boxplot(x='Title', y='Age', data=df_train)
plt.show()


# In[48]:


#Find the median of Age in each title.
#(Remarks: only use train dataset to avoid data leakage)


# In[49]:


# Handling Missing Data
columns_object = list(df_train.select_dtypes(include='object').columns)
columns_numeric = list(df_train.select_dtypes(exclude='object').columns)
columns_numeric, columns_object


# In[50]:


columns_numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
columns_object = ['Sex', 'Embarked', 'Title']


# In[51]:


from sklearn.impute import SimpleImputer


# In[52]:


imputer_numeric = SimpleImputer(strategy='median')
imputer_object = SimpleImputer(strategy='most_frequent')
imputer_numeric.fit(df_train[columns_numeric])
imputer_object.fit(df_train[columns_object])

df_train[columns_numeric] = imputer_numeric.transform(df_train[columns_numeric])
df_train[columns_object] = imputer_object.transform(df_train[columns_object])


# In[53]:


df_test[columns_numeric] = imputer_numeric.transform(df_test[columns_numeric])
df_test[columns_object] = imputer_object.transform(df_test[columns_object])


# In[54]:


df_train.isna().sum()


# In[55]:


df_test.isna().sum()


# # Step 3 - Data Transformation

# #Encode string to numbers for modelling.

# In[56]:


def get_age_group(X):
    if X>50:
        return 'older'
    if X>25:
        return 'young'
    return 'kid'


# In[57]:


# BINNING => new => categorical
df_train['Age_group'] = df_train['Age'].apply(get_age_group)
df_test['Age_group'] = df_test['Age'].apply(get_age_group)


# In[58]:


columns_object.append('Age_group')


# In[59]:


columns_object


# In[60]:


df_train.head()


# In[61]:


# Feature Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df_train[columns_object])


# In[62]:


temp = encoder.transform(df_train[columns_object]).toarray()
df_temp = pd.DataFrame(temp)
df_train = pd.concat([df_train,df_temp], axis=1)
columns_numeric.extend(list(df_temp.columns))


# In[63]:


temp = encoder.transform(df_test[columns_object]).toarray()
df_temp = pd.DataFrame(temp)
df_test = pd.concat([df_test,df_temp], axis=1)


# In[64]:


df_test


# In[65]:


df_train


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


X_train, X_validate, y_train, y_validate = train_test_split(df_train[columns_numeric],df_train['Survived'], test_size=0.2, random_state=14)


# In[68]:


X_train.shape, X_validate.shape, y_train.shape, y_validate.shape


# In[69]:


X_train.head()


# # Step 4 - model

# # LogisticRegression

# In[70]:


#scaling


# In[71]:


from sklearn.preprocessing import MinMaxScaler


# In[72]:


scaler = MinMaxScaler()


# In[73]:


scaler.fit(X_train) # to find min and max values of every col in X_train


# In[74]:


X_train_scaled = scaler.transform(X_train)
X_validate_scaled = scaler.transform(X_validate)


# In[75]:


X_train_scaled.shape, X_validate_scaled.shape,X_train.shape, X_validate.shape


# In[76]:


pd.DataFrame(X_train_scaled).head()


# In[77]:


from sklearn.linear_model import LogisticRegression


# In[78]:


model_log = LogisticRegression()
model_log.fit(X_train_scaled,y_train)
model_log.score(X_validate_scaled,y_validate)


# # SVM

# In[79]:


from sklearn.svm import SVC


# In[80]:


model_linear = SVC(kernel='linear') # using linear kernel


# In[81]:


model_linear.fit(X_train_scaled,y_train)
model_linear.score(X_validate_scaled,y_validate)


# In[82]:


model_poly = SVC(kernel='poly') # using poly kernel
model_poly.fit(X_train_scaled,y_train)
model_poly.score(X_validate_scaled,y_validate)


# In[83]:


model_rbf = SVC(kernel='rbf') #using rbf kernel
model_rbf.fit(X_train_scaled,y_train)
model_rbf.score(X_validate_scaled,y_validate)


# In[84]:


#poly kernel provided best accuracy


# # Using decision tree

# In[85]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[88]:


model_DT = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100)
model_DT.fit(X_train_scaled,y_train)
model_DT.score(X_validate_scaled,y_validate)


# # Using random forest

# In[89]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[91]:


p = {'n_estimators':[25,50,100,150,200], 'max_depth':[2,3,4,5,6], 'min_samples_leaf':[3,4,5,6,7]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=1),p,n_jobs=-1,cv=5, verbose=3)
grid_cv.fit(X_train_scaled,y_train)


# In[92]:


grid_cv.best_estimator_


# In[93]:


model_RF = RandomForestClassifier(random_state=1, max_depth=6, min_samples_leaf=3, n_estimators=150)
model_RF.fit(X_train_scaled,y_train)
model_RF.score(X_validate_scaled,y_validate)


# In[94]:


#poly kernel provided best accuracy and Random forest has proved best accuray


# # Step -5 Predictions

# In[96]:


df_test[columns_numeric]


# In[101]:


y_test_scaled = scaler.transform(df_test[columns_numeric])
pd.DataFrame(y_test_scaled)


# In[102]:


y = model_RF.predict(y_test_scaled)


# In[103]:


y


# In[104]:


df_test['Survived'] = y


# In[105]:


df_test[['PassengerId','Survived']].to_csv('sub2_dsml14.csv',index=False)

