# multiclass classification
import pandas as pd
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.externals import joblib
from xgboost import plot_importance

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load data
data = pd.read_csv('bigdata.csv')
#data = data[['open','close','high', 'low', 'SEMA', 'LEMA', 'EMACD','TM']]
#data = (data - data.mean()) / (data.max() - data.min())
dataset = data.values
# split data into X and y
X = dataset[:,2:13]
Y = dataset[:,15]
print(X)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
Y = label_encoder.transform(Y)
print("Transforming Y...")
seed = 7
test_size = 0.33
#X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)
train, test, train_labels, test_labels = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgboost.XGBClassifier(max_depth=10,
                min_child_weight=2,
                subsample=.8,
                colsample_bytree=.99,
                n_estimators=10,
                learning_rate=.3)
print('Fitting model ...')
model.fit(train, train_labels)
print('Model complete..')
joblib.dump(model, 'filename3.pkl')
# make predictions for test data
print('Predicting new values..')
y_pred = model.predict(test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = [1,0]
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

plot_importance(model)
plt.show()