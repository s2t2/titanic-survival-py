
#from sklearn.pipeline import make_pipeline
#from sklearn.impute import SimpleImputer
#from category_encoders import OneHotEncoder
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.importer import training_and_validation_splits

if __name__ == "__main__":

    xtrain, ytrain, xval, yval = training_and_validation_splits()

    model = DecisionTreeClassifier(random_state=89)

    model.fit(training_x, training_y)
    print("Accuracy (Training)", model.score(training_x, training_y))
    print("Accuracy (Validation)", model.score(val_x, val_y))

    #pipeline = make_pipeline(
    #    OneHotEncoder(use_cat_names=True),
    #    SimpleImputer(strategy='median'),
    #    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    #)
    #pipeline.fit(X_train, y_train)
