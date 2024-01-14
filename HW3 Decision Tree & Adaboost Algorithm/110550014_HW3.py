# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# This function computes the gini impurity of a label array.
def gini(y):
    if len(y) == 0:
        return 0
    return (len(y[y==1]) / len(y)) ** 2 + (len(y[y==0]) / len(y)) ** 2

# This function computes the entropy of a label array.
def entropy(y):
    if len(y) == 0 or len(y[y==1]) == 0 or len(y[y==0]) == 0:
        return 0
    return -(len(y[y==1]) / len(y)) * np.log2(len(y[y==1]) / len(y)) - (len(y[y==0]) / len(y)) * np.log2(len(y[y==0]) / len(y))
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class node():
    def __init__(self, feature=-1, threshold=-1, L=None, R=None, label=-1):
        self.feature = feature
        self.threshold = threshold
        self.L = L
        self.R = R
        self.label = label

class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.root = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.

    def fit(self, X, y, depth=1):
        if len(y) == 0 or len(y[y==0]) == 0 or len(y[y==1]) == 0 or depth >= self.max_depth:
            return node(-1, -1, None, None, len(y[y==1]) >= len(y[y==0]))
        best_gain, best_feature, best_threshold = self.impurity(y), -1, -1
        for feature in range(X.shape[1]):
            thresholds = X[:, feature]
            thresholds = list(dict.fromkeys(thresholds))
            thresholds.sort()
            for i in range(len(thresholds) - 1):
                thresholds[i] = (thresholds[i] + thresholds[i+1]) / 2
            thresholds.pop()
            for threshold in thresholds:
                yL = y[X[:, feature] <= threshold]
                yR = y[X[:, feature] > threshold]
                gain = (len(yL) / len(y)) * self.impurity(yL) + (len(yR) / len(y)) * self.impurity(yR)
                if (self.criterion == 'entropy' and gain < best_gain) or (self.criterion == 'gini' and gain > best_gain):
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        if best_feature == -1 and best_threshold == -1:
            return node(-1, -1, None, None, len(y[y==1]) >= len(y[y==0]))
        Lnode =  self.fit(X[X[:, best_feature] <= best_threshold], y[X[:, best_feature] <= best_threshold], depth+1)
        Rnode =  self.fit(X[X[:, best_feature] > best_threshold], y[X[:, best_feature] > best_threshold], depth+1)
        
        curnode = node(best_feature, best_threshold, Lnode, Rnode, len(y[y==1]) >= len(y[y==0]))
        self.root = curnode
        return curnode


    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred = []
        for x in X:
            cur = self.root
            while(cur.threshold != -1 and cur.feature != -1):
                feature, thershold = cur.feature, cur.threshold
                if x[feature] <= thershold:
                    cur = cur.L
                else:
                    cur = cur.R
            pred.append(cur.label)
        return pred
        
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns=None):
        feature = ['age', 'sex', 'cp', 'fbs', 'thalach' ,'thal']    
        count = np.zeros((6,))
        def inorder(root):
            if root == None:
                return
            if root.feature != -1:
                count[root.feature] += 1
            inorder(root.L)
            inorder(root.R)
        inorder(self.root)
        plt.barh(feature, count)
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.classifier = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        weight = np.ones((len(X),)) / len(X)
        for _ in range(self.n_estimators):
            resample = np.random.choice(len(X), len(X), p=weight)
            X_train, y_train = X[resample], y[resample]
            tree = DecisionTree(self.criterion, 2)
            tree.fit(X_train, y_train)
            pred = tree.predict(X)
            pred = abs(y - pred)
            error = np.sum(pred * weight)
            alpha = 0.5 * np.log((1 - error) / error)
            pred = np.power(-1, pred, dtype='float')
            weight *= np.exp(-alpha * pred)
            weight /= np.sum(weight)


            self.classifier.append([alpha, tree.root])

        



    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred = []
        for x in X:
            y = 0
            for classifier in self.classifier:
                alpha, wc = classifier[0], classifier[1]
                feat, thres = wc.feature, wc.threshold
                if x[feat] <= thres:
                    y += -alpha * np.power(-1, wc.L.label)
                else:
                    y += -alpha * np.power(-1, wc.R.label)
            pred.append(y >= 0)
        return pred

                    


# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))


    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=15):", accuracy_score(y_test, y_pred))


    # tree.plot_feature_importance_img()

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=10)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    
