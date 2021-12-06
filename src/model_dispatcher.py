from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    #"SVC": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    #"MLPClassifier": MLPClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GaussianNB": GaussianNB(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis()
}
