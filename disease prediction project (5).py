#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


df = pd.read_csv('data.csv')


# In[7]:


df


# In[8]:


df["BodyTemp."]


# In[9]:


df["BodyTemp_cat"]= pd.cut(df["BodyTemp."],bins=[98, 101 ,104,float('inf')], labels =['low', 'normal','high'])


# In[10]:


df.drop("BodyTemp.",axis = 1,inplace =True)


# In[11]:


df=pd.get_dummies(df,columns=["BodyTemp_cat"])


# In[12]:


df


# In[13]:


df.head()


# In[14]:


df.tail()


# In[15]:


df.info()


# In[16]:


df['BodyPain'].value_counts()


# In[17]:


df.describe()


# In[18]:


def data_split(data, ratio):
    np.random.seed(100)
    shuffled = np.random.permutation(len(data))
    test_data_size = int(len(data)*ratio)
    test_data_indices = shuffled[: test_data_size]
    train_data_indices = shuffled[test_data_size :]
    return data.iloc[test_data_indices], data.iloc[train_data_indices]


# In[19]:


test_data, train_data = data_split(df,0.15)


# In[20]:


test_data, train_data = data_split(df, 0.15)


# In[21]:


test_data


# In[22]:


train_data


# In[23]:


x_train = train_data[['BodyTemp_cat_low','BodyTemp_cat_normal','BodyTemp_cat_high', 'Fatigue', 'Cough', 'BodyPain', 'SoreThroat', 'BreathingDifficulty']].to_numpy()


# In[24]:


x_train


# In[25]:


x_test = test_data[['BodyTemp_cat_low','BodyTemp_cat_normal','BodyTemp_cat_high', 'Fatigue', 'Cough', 'BodyPain', 'SoreThroat', 'BreathingDifficulty']].to_numpy()


# In[26]:


x_test


# In[27]:


y_train = train_data[['Infected']].to_numpy().reshape(3400, )
y_test = test_data[['Infected']].to_numpy().reshape(600, )


# In[28]:


y_train


# In[29]:


y_test


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


clf = LogisticRegression()


# In[32]:


clf.fit(x_train, y_train)


# In[33]:


clf.predict_proba([[0,0,1, 1, 1, 1, 1, 1]])


# In[34]:


y_pred = clf.predict_proba(x_test)


# In[35]:


y_pred


# In[36]:


from sklearn import svm


# In[37]:


clf1 = svm.SVC()


# In[38]:


clf1.fit(x_train, y_train)


# In[39]:


y1_pred=clf1.predict([[0,0,1, 1, 1, 1, 1, 1]])


# In[40]:


y1_pred


# In[41]:


from sklearn.metrics import accuracy_score


# In[42]:


logistic_accuracy = accuracy_score(y_test, clf.predict(x_test))


# In[43]:



print("Logistic Regression Accuracy:", logistic_accuracy)


# In[44]:


svm_accuracy = accuracy_score(y_test, clf1.predict(x_test))


# In[45]:


print("SVM Accuracy:", svm_accuracy)


# In[46]:


from sklearn.metrics import log_loss


# In[47]:


log_loss(y_test, y_pred)


# In[59]:





# In[58]:





# In[1]:




   


# In[2]:





# In[ ]:




