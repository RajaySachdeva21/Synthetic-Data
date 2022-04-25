#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
from keras.layers import Dense
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from matplotlib import pyplot
from numpy.random import randn
import pandas as pd


# In[39]:


data = pd.read_csv(r'C:\\Users\\Smrijay\\OneDrive\\Desktop\\Synthetic data Project\diabetes.csv')
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
ouputlabel = ['Outcome']
X = data[columns]
y = data[ouputlabel]

X_true_train, X_true_test, y_true_train, y_true_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf_true = RandomForestClassifier(n_estimators=100)
clf_true.fit(X_true_train,y_true_train)

y_true_pred=clf_true.predict(X_true_test)
print("Real Data :",metrics.accuracy_score(y_true_test, y_true_pred))
print("Real Data classification report:",metrics.classification_report(y_true_test, y_true_pred))


# In[40]:


def Latent_points(dim, data):
	x_input = randn(dim * data)
	x_input = x_input.reshape(data, dim)
	return x_input


# In[41]:


def fakeSamp(grtr,dim,samp):

	x_input = Latent_points(dim,samp)
	X = grtr.predict(x_input)
	y = np.zeros((samp, 1))
	return X, y


# In[42]:


def generate_real_samples(n):
    X = data.sample(n)
    y = np.ones((n, 1))
    return X, y


# In[43]:


def realSamp(a):
  X = data.sample(a)
  y = np.ones((a, 1))
  return X, y


# In[44]:



def grtor(dim, outputs=9):
  model = Sequential()
  model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=dim))
  model.add(Dense(30, activation='relu'))
  model.add(Dense(outputs, activation='linear'))
  return model


# In[45]:


generator1 = grtor(10, 9)


# In[46]:


def distor(n_inputs=9):
  model = Sequential()
  model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


# In[47]:


discriminator1 = distor(9)


# In[48]:


def gan(generator, discriminator):
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model


# In[49]:



def train(gm, dm, gam, ld,e=10000, b=128):
  half_batch = int(b / 2)
  d_history = []
  
  for epoch in range(e):
    xr, yr = realSamp(half_batch)
    xf, yf = fakeSamp(gm, ld, half_batch)
    d_loss_real, d_real_acc = dm.train_on_batch(xr, yr)
    d_loss_fake, d_fake_acc = dm.train_on_batch(xf, yf)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    x_gan = Latent_points(ld,b)
    y_gan = np.ones((b, 1))
    g_loss_fake = gam.train_on_batch(x_gan, y_gan)
    gm.save('model.h5')


# In[50]:



def performance(e, generator, discriminator, latent_dim, n=100):

	xr, yr = realSamp(n)
	_, acc_real = discriminator.evaluate(xr, yr, verbose=0)
	xf, yf = fakeSamp(generator, latent_dim, n)
	_, acc_fake = discriminator.evaluate(xf, yf, verbose=0)
	print(e, acc_real, acc_fake)
	pyplot.scatter(xr[:, 0], color='blue')
	pyplot.scatter(xf[:, 0], color='yellow')
	pyplot.show()


# In[51]:



latent_dim = 10

discriminator = distor()
gtor = grtor(latent_dim)
gam = gan(gtor, discriminator)
train(gtor, discriminator, gam, latent_dim)


# In[52]:


from keras.models import load_model
model = load_model(r'C:\\Users\\Smrijay\\model.h5')
from keras.models import load_model

latent_points = Latent_points(10, 750)
X = model.predict(latent_points)
fakeData = pd.DataFrame(data=X,  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
outcome_mean = fakeData.Outcome.mean()
fakeData['Outcome'] = fakeData['Outcome'] > outcome_mean
fakeData['Outcome']
fakeData["Outcome"] = fakeData["Outcome"].astype(int)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

label = ['Outcome']
Xf = fakeData[features]
yf = fakeData[label]
Xftr, Xftest, yftr, yftest = train_test_split(Xf, yf, test_size=0.30, random_state=42)
clf_fake = RandomForestClassifier(n_estimators=100)
clf_fake.fit(Xftr,yftr)

y_fake_pred=clf_fake.predict(Xftest)
print("Accuracy of fake data model:",metrics.accuracy_score(yftest, y_fake_pred))
print("Classification report of fake data model:",metrics.classification_report(yftest, y_fake_pred))


# In[36]:


from table_evaluator import load_data, TableEvaluator
table_evaluator = TableEvaluator(data, fakeData)
table_evaluator.evaluate(target_col='Outcome')


# In[37]:


table_evaluator.visual_evaluation()


# In[ ]:




