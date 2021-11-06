from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import dataClass 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

# loading Glove vector files.
glove_input_file = '/content/glove.6B.100d.txt'
word2vec_output_file = '/content/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)




filename='dataset.json'
data=dataClass.prepareData(filename)
data.preprocessData()
data.convertVectors()

print('Total instances')
print('length of dataset: %s', data.getLength())

# split into training and testing sets.
x_train, x_test, y_train, y_val = train_test_split(data.X, data.Y, test_size=0.1, random_state=1000)

print('no. of training instances:', len(x_train))
print('no. of testing instances:', len(x_val))

#Baselines 1
print('Baselines')
print('Word  embedding: frequency based')

# Naive Bayes 
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_val)
print('Naive Bayes accuracy:', accuracy_score(y_val, y_pred))

#SVM's
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = GridSearchCV(SVC(), parameters)
clf.fit(x_val, y_val)
# applying grid search CV to estimate best set of hyper-parameters
grid=GridSearchCV(estimator=SVC(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})

# training SVM
clf = make_pipeline(StandardScaler(), SVC(C = clf.best_params_['C'], kernel = clf.best_params_['kernel']))
clf.fit(x_train, y_train)
y_pred= clf.predict(x_val)
print('SVM accuracy:', accuracy_score(y_val, y_pred))

#Random Forest
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=50,random_state=0) #optimum set of hyper-parameters set after experimentation
distributions = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'])
clf = RandomizedSearchCV(logistic, distributions, random_state=0)
search = clf.fit(x_val, y_val)

# Training the model
x_train, y_train = make_classification(n_informative=2, n_redundant=0,random_state=0, shuffle=False,C = search.best_params_['C'], penalty=search.best_params_['penalty'])
clf = RandomForestClassifier(max_depth=50, random_state=0)
clf.fit(x_train, y_train)
y_pred= clf.predict(x_val)
print('Random Forest accuracy:', accuracy_score(y_val, y_pred))


# Baselines 2
print('Baselines')
print('Word embedding: Glove Embedding')

data=prepareData(filename)
data.preprocessData()
data.convertGloVe()
x_train, x_val, y_train, y_val = train_test_split(data.X, data.Y, test_size=0.1, random_state=1000)

print('no. of training instances:', len(x_train))
print('no. of testing instances:', len(x_val))

# Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_val)
print('Naive Bayes accuracy:', accuracy_score(y_val, y_pred))

#SVM's
clf = make_pipeline(StandardScaler(), SVC(C = clf.best_params_['C'], kernel = clf.best_params_['kernel']))
clf.fit(x_train, y_train)
y_pred= clf.predict(x_val)
print('SVM accuracy:', accuracy_score(y_val, y_pred))

# Random Forest
x_train, y_train = make_classification(,n_informative=2, n_redundant=0,random_state=0, shuffle=False, C = search.best_params_['C'], penalty=search.best_params_['penalty'])
clf = RandomForestClassifier(max_depth=50, random_state=0)
clf.fit(x_train, y_train)
y_pred= clf.predict(x_val)
print('Random Forest accuracy:', accuracy_score(y_val, y_pred))


