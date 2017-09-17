#!/usr/bin/python
import math
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

# my_dataset.pkl already contains new features, so set this to false to save time
ADD_NEW_FEATURE = False
# set this to True to show feature selection precoess
FEATURE_SELECTION = False

# The helper method to calculate the fraction of emails
def computeFraction( poi_messages, all_messages ):
	if math.isnan(float(poi_messages)) or math.isnan(float(all_messages)):
		fraction = 0
	else:
		fraction = poi_messages/float(all_messages)
	return fraction

# Add new features
def addFeatures():
	if ADD_NEW_FEATURE:
		# Load the dictionary containing the dataset
		data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
		# Remove outliers
		data_dict.pop('TOTAL',0)
		# Create new feature(s)
		from poi_by_email import predictByEmail
		# Load the text learning classifier
		text_clf = pickle.load(open("text_learn_clf.pkl", "r"))
		# Load the TfidfVectorizer
		vec = pickle.load(open("vectorizer.pkl", "r"))
		# Add new feature named 'text_learn_pred' which is the prediction result from the text learning
		for k,v in data_dict.iteritems():
			 v['text_learn_pred'] = 'NaN'
			 if v['email_address'] == 'NaN':
				 pass
			 else:
				 v['text_learn_pred'] = predictByEmail(v['email_address'], text_clf, vec)[0]
			 if v['text_learn_pred'] == 'N':
				 v['text_learn_pred'] = 'NaN'

		# Add two new features called 'fraction_to_poi' and 'fraction_from_poi'
		for name in data_dict:
			 data_point = data_dict[name]
			 from_poi_to_this_person = data_point["from_poi_to_this_person"]
			 to_messages = data_point["to_messages"]
			 fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
			 data_point["fraction_from_poi"] = fraction_from_poi
			 from_this_person_to_poi = data_point["from_this_person_to_poi"]
			 from_messages = data_point["from_messages"]
			 fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
			 data_point["fraction_to_poi"] = fraction_to_poi
		return data_dict
	else:
		my_dataset = pickle.load(open("my_dataset.pkl","r"))
		return my_dataset

# score function
def score_func(y_true,y_predict):
	true_negatives = 0
	false_negatives = 0
	true_positives = 0
	false_positives = 0

	for prediction, truth in zip(y_predict, y_true):
		if prediction == 0 and truth == 0:
			true_negatives += 1
		elif prediction == 0 and truth == 1:
			false_negatives += 1
		elif prediction == 1 and truth == 0:
			false_positives += 1
		else:
			true_positives += 1
	if true_positives == 0:
		return (0,0,0)
	else:
		precision = 1.0*true_positives/(true_positives+false_positives)
		recall = 1.0*true_positives/(true_positives+false_negatives)
		f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
		return (precision,recall,f1)


def univariateFeatureSelection(f_list, my_dataset):
	result = []
	for feature in f_list:
		# Replace 'NaN' with 0
		for name in my_dataset:
			data_point = my_dataset[name]
			if not data_point[feature]:
				data_point[feature] = 0
			elif data_point[feature] == 'NaN':
				data_point[feature] =0

		data = featureFormat(my_dataset, ['poi',feature], sort_keys = True, remove_all_zeroes = False)
		labels, features = targetFeatureSplit(data)
		features = [abs(x) for x in features]
		from sklearn.cross_validation import StratifiedShuffleSplit
		cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
		features_train = []
		features_test  = []
		labels_train   = []
		labels_test    = []
		for train_idx, test_idx in cv:
			for ii in train_idx:
				features_train.append( features[ii] )
				labels_train.append( labels[ii] )
			for jj in test_idx:
				features_test.append( features[jj] )
				labels_test.append( labels[jj] )
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
		clf.fit(features_train, labels_train)
		predictions = clf.predict(features_test)
		score = score_func(labels_test,predictions)
		result.append((feature,score[0],score[1],score[2]))
	result = sorted(result, reverse=True, key=lambda x: x[3])
	return result

def selectKBest(previous_result, data):
	# remove 'restricted_stock_deferred' and 'director_fees'
	previous_result.pop(4)
	previous_result.pop(4)

	result = []
	_k = 10
	for k in range(0,_k):
		feature_list = ['poi']
		for n in range(0,k+1):
			feature_list.append(previous_result[n][0])

		data = featureFormat(my_dataset, feature_list, sort_keys = True, remove_all_zeroes = False)
		labels, features = targetFeatureSplit(data)
		features = [abs(x) for x in features]
		from sklearn.cross_validation import StratifiedShuffleSplit
		cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
		features_train = []
		features_test  = []
		labels_train   = []
		labels_test    = []
		for train_idx, test_idx in cv:
			for ii in train_idx:
				features_train.append( features[ii] )
				labels_train.append( labels[ii] )
			for jj in test_idx:
				features_test.append( features[jj] )
				labels_test.append( labels[jj] )
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
		clf.fit(features_train, labels_train)
		predictions = clf.predict(features_test)
		score = score_func(labels_test,predictions)
		result.append((k+1,score[0],score[1],score[2]))
	return result

def decisionTree(feature_list,dataset):
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	test_classifier(clf, my_dataset, features_list)
	print '### feature importance'
	print clf.feature_importances_

def GNB(feature_list,dataset):
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	test_classifier(clf, my_dataset, features_list)

def KNN(feature_list,dataset):
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	knn = KNeighborsClassifier()
	# feature scale
	estimators = [('scale', StandardScaler()), ('knn', knn)]
	clf = Pipeline(estimators)
	test_classifier(clf, my_dataset, features_list)

def tuneKNN(feature_list,dataset):
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.grid_search import GridSearchCV
	knn = KNeighborsClassifier()
	# feature scale
	estimators = [('scale', StandardScaler()), ('knn', knn)]
	pipeline = Pipeline(estimators)
	parameters = {'knn__n_neighbors':[1,8],
		'knn__algorithm':('ball_tree','kd_tree','brute','auto')}
	clf = GridSearchCV(pipeline, parameters,scoring = 'recall')
	test_classifier(clf, my_dataset, features_list)
	print '###best_params'
	print clf.best_params_


def tuneDT(feature_list,dataset):
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.grid_search import GridSearchCV
	from sklearn import tree
	tree_clf = tree.DecisionTreeClassifier()
	parameters = {'criterion':('gini', 'entropy'),
		'splitter':('best','random')}
	clf = GridSearchCV(tree_clf, parameters,scoring = 'recall')
	test_classifier(clf, my_dataset, features_list)
	print '###best_params'
	print clf.best_params_

features_list_all = [
	# financial features
	'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
	# email features
	'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi', 'text_learn_pred'
	]

features_list = ['poi','total_stock_value','exercised_stock_options','bonus','deferred_income','long_term_incentive']

my_dataset = addFeatures()

if FEATURE_SELECTION:
	# univariate feature selection
	univariate_result = univariateFeatureSelection(features_list_all,my_dataset)
	print '### univariate feature selection result'
	for l in univariate_result:
		print l

	# select k best
	select_best_result = selectKBest(univariate_result, my_dataset)
	print '### select k best result'
	for l in select_best_result:
		print l

### Try a varity of classifiers
### Gaussian naive bayes
GNB(features_list,my_dataset)
### Decision Tree
decisionTree(features_list,my_dataset)
### K nearest neighbors
KNN(features_list,my_dataset)

# Tuen the algorithms
tuneKNN(features_list,my_dataset)
tuneDT(features_list,my_dataset)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)
### Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
