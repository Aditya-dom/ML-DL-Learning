## Fetching data

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x,y = mnist['data'], mnist['target']

#Checking shape --> size
# x.shape
# y.shape

# Plotting graph
import matplotlib
import matplotlib.pyplot as plt

some_digit = x[36000]
#lets reshape it to plot it
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis("off")

x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

import numpy as np
shuffle_index = np.random.permutation(60000)
x_train , y_train = x_train[shuffle_index], y_train[shuffle_index]

## Creating a 2 detector

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==2)
y_test_2  = (y_test==2)

y_train_2 ## array ([false,false,...,false])
y_test_2 ## array ([false,false,...,false])

# probability of getting 2 is 1/10 means 10% chances are there to get 2

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(tol = 0.1 , solver='lbfgs')
clf.fit(x_train,y_test_2)
clf.predict([some_digit])


from sklearn.model_selection import cross_val_predict
a= cross_val_predict(clf,x_train,y_train_2,cv=3,scoring = "accuracy")
a.mean()

##


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf,x_train,y_train_2,cv=3)
# y_train_pred

## Calculationg confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_2, y_train_pred)
confusion_matrix(y_train_2, y_train_2) # idol matrix --> this is confusion matrix for perfect predictions 
#eg:

"""arry([[676   0],
         [0   676]), dtype=int64
"""

## Precision And Recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_2, y_train_pred)
recall_score(y_train_2, y_train_pred)

## F1-Score
from sklearn.metrics import f1_score
f1_score(y_train_2, y_train_pred)

## Precision recall Curve
from sklearn.metrics import precision_recall_curve
y_score = cross_val_predict(clf,x_train,y_train_2,cv=3,method = "decision_function")

precision, recall, thresholds = precision_recall_curve(y_train_2, y_score)

#precision -->array   
#recall -->array  
#threshold -->array  

## PLotting the Precision recall curve
plt.plot(thresholds,precision[:-1], "b--", label="Precision")
plt.plot(thresholds, recall[:-1],"g--", labels="Recall")
plt.xlabel("Thresholds")
plt.legend(loc="upper left")
plt.ylim([0,1])
plt.show()


