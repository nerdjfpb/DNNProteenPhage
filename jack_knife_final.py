import numpy as np
from sklearn import preprocessing
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn import feature_selection as fs
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import time
from sklearn.model_selection import LeaveOneOut

start_time = time.time()

tf.logging.set_verbosity(tf.logging.INFO)

data=np.genfromtxt('datasub.csv',delimiter=',')

X = data[: , 0:1548]  # 1546 features, +1 output column, +1 for python index


#X_new = SelectKBest(fs.f_classif, k=600).fit_transform(X, y)


#feature select
#feature_numbers=len(X_new[1])+1

kfoldResult=np.array([])


loo = LeaveOneOut()

for train, test in loo.split(X):
    #print(train.shape)
    #print(test.shape)
  	#kfloditeration=kfloditeration+1

	X_train=X[train]
	X_test=X[test]

    #for train set
	train_features=X_train[: , 0:1546]
	train_output=X_train[:,1546]
    #print(train_features.shape)

	
	test_features=X_test[: , 0:1546]
	test_output=X_test[:,1546]

	estimator = SVR(kernel="linear")
	selector = RFE(estimator, 99, step=1.0)
	train_features = selector.fit_transform(train_features, train_output)

	estimator = SVR(kernel="linear")
	selector = RFE(estimator, 99, step=1.0)
	test_features = selector.fit_transform(test_features, test_output)

	#print(test_features.shape)
    #print(X_train_Class)
	#data scalling ........... This do Work Faster
	train_X = preprocessing.scale(train_features)
	test_X	= preprocessing.scale(test_features)
	#print(train_X.shape)

	#print(standardized_X[1,1])


	#parameter for classifier
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=98)]


	a = np.array([])
	  
	p=100
	nTimes=2

	for i in range(1,nTimes):
	#classifier
		classifier = skflow.DNNClassifier(feature_columns=feature_columns,
										hidden_units=[25,15,10], 
										n_classes=2,
										#optimizer=tf.train.GradientDescentOptimizer(0.1))
										optimizer=tf.train.AdamOptimizer)



		#train dataset
		classifier.fit(train_X, train_output, steps=p)
		p=p+1


		#accuracy print
		accuracy_score = classifier.evaluate(test_X, test_output, steps=1)["accuracy"]
		a=np.append(a,accuracy_score)
		print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

	#for i in range(0,nTimes):
	    #print(a[i])

	#print('Max value')
	#print(a.max())

	kfoldResult=np.append(kfoldResult,a.max())


for element in kfoldResult:
    print(element)

#print("Max Value In Kfold : ")
#print(kfoldResult.max())
print("Avrg Value In Jack-Knife: ")
print(kfoldResult.mean())

print("--- %s seconds ---" % (time.time() - start_time))