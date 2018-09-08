from sklearn.cross_validation import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy


class CrossMachine(object):

    def __init__(self):
        self.model_cross = OneVsRestClassifier(LinearSVC(random_state=0))

    def fit_and_predict(self,data_t,target_t):
        scores = cross_val_score(self.model_cross,data_t,target_t)
        print(scores)
