# Loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading dataset
iris = datasets.load_iris()

#Printing descriptions and features
#print(iris.DESCR)  """NOT imp to print but we see this because features"""
features = iris.data
labels = iris.target
print(features[0], labels[0])

#Trainging fthe classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)

preds= clf.predict([[41,1,1,1]])
print(preds)