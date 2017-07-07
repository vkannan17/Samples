import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx =[0,50,100]

#Traning data
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx, axis = 0)

#Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print (test_target)
print (clf.predict(test_data))


import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")

