import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
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



model = joblib.load('filename3.pkl')

data = pd.read_csv('bigdata2.csv', header=0)
# data = (data - data.mean()) / (data.max() - data.min())
dataset = data.values
# split data into X and y
X = dataset[:,2:13]
Y = dataset[:,15]
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
Y = label_encoder.transform(Y)
print(Y)
print(X)
print ("test error:")
pred_test = model.predict(X)
print(np.sqrt(np.mean((pred_test - Y)**2)))




plt.figure()
plt.plot(Y, label='actual')
plt.plot(pred_test, label='prediction')
plt.legend(fontsize='x-small')
plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y, pred_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = [1,0]
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

# evaluate predictions
accuracy = accuracy_score(Y, pred_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))