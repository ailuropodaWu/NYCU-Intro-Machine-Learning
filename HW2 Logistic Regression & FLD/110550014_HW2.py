# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0

        # Add a bias term (intercept) to the input features
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Gradient Descent
        for _ in range(self.iteration):
            linear_model = np.dot(X_b, np.r_[self.intercept, self.weights])
            y_predicted = self.sigmoid(linear_model)

            # Compute the gradient of the cross-entropy loss
            gradient = np.dot(X_b.T, (y_predicted - y)) / n_samples

            # Update the weights and intercept
            self.weights -= self.learning_rate * gradient[1:]
            self.intercept -= self.learning_rate * gradient[0]     
         
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.intercept
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        # Separate data into two classes
        X0 = X[y == 0]
        X1 = X[y == 1]

        # Compute means of the two classes
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        # Compute the within-class scatter matrix
        self.sw = np.dot((X0 - self.m0).T, (X0 - self.m0)) + np.dot((X1 - self.m1).T, (X1 - self.m1))

        # Compute the between-class scatter matrix
        self.sb = np.atleast_2d(self.m1 - self.m0).T.dot(np.atleast_2d(self.m1 - self.m0))

        # Compute the weights
        self.w = np.linalg.inv(self.sw).dot(self.m1 - self.m0)
        self.slope = self.w[1] / self.w[0]

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        y_pred = []
        # Project data
        X_projected = np.dot(X, self.w)
        # Project class means
        m0_projected = np.dot(self.m0, self.w)
        m1_projected = np.dot(self.m1, self.w)

        # Compare distances
        for x in X_projected:
            dist0 = np.abs(x - m0_projected)
            dist1 = np.abs(x - m1_projected)
            y_pred.append(0 if dist0 < dist1 else 1)
        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        y = self.predict(X)
        y = np.array(y)
        
        intercept = 330
        def project_point_to_line(x0, y0):
            slope = self.slope
            x1 = (x0 + slope * y0 - slope * intercept) / (1 + slope**2)
            y1 = (slope * x0 + (slope**2) * y0 + intercept) / (1 + slope**2)
            return x1, y1

        plt.figure(figsize=(10, 10))


        X0 = X[y == 0]
        X1 = X[y == 1]
        plt.scatter(X0.T[0], X0.T[1], c='b', marker='o')
        plt.scatter(X1.T[0], X1.T[1], c='r', marker='o')
        p0x, p0y = project_point_to_line(X0.T[0], X0.T[1])
        p1x, p1y = project_point_to_line(X1.T[0], X1.T[1])
        plt.scatter(p0x, p0y, c='b', marker='o')
        plt.scatter(p1x, p1y, c='r', marker='o')

        for i in range(len(X0)):
            plt.plot([X0[i][0], p0x[i]], [X0[i][1], p0y[i]], c='b', lw=0.1)
        for i in range(len(X1)):
            plt.plot([X1[i][0], p1x[i]], [X1[i][1], p1y[i]], c='r', lw=0.1)
        _x0 = 50
        plt.axline((_x0, intercept + _x0 * self.slope), slope=self.slope, c='k')
        plt.axis('equal')
        plt.title(f'Projection Line: w={self.slope}, b={intercept}')
        plt.show()
    

        
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.00015, iteration=70000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    # FLD.plot_projection(X_test)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

