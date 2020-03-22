# adapted from: https://github.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge

import os
import json
from pprint import pprint
import pickle

from pandas import Series, DataFrame
#import matplotlib.pyplot as plt
import plotly.express as px

#from sklearn.feature_selection import SelectKBest # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
from category_encoders import OneHotEncoder # see: https://contrib.scikit-learn.org/categorical-encoding/onehot.html
from category_encoders import OrdinalEncoder # see: https://contrib.scikit-learn.org/categorical-encoding/ordinal.html
from sklearn.impute import SimpleImputer # see: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
from sklearn.preprocessing import StandardScaler # see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.tree import DecisionTreeClassifier # see: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#from sklearn.ensemble import RandomForestClassifier

#from sklearn.pipeline import make_pipeline # see: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
from sklearn.pipeline import Pipeline # see: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
from sklearn.model_selection import GridSearchCV # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#from sklearn.model_selection import RandomizedSearchCV # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
#from scipy.stats import randint, uniform

from sklearn.metrics import classification_report
from graphviz import Source # see: https://pypi.org/project/graphviz/
from sklearn.tree import export_graphviz # see: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

from app import REPORTS_DIR, SUBMISSIONS_DIR, MODELS_DIR
from app.importer import Importer

TREE_GRAPH_FILEPATH = os.path.join(REPORTS_DIR, "decision_tree.png")
FEATURES_GRAPH_FILEPATH = os.path.join(REPORTS_DIR, "feature_importances.png")
SEARCH_RESULTS_FILEPATH = os.path.join(REPORTS_DIR, "search_results.json")
PREDICTIONS_FILEPATH = os.path.join(SUBMISSIONS_DIR, "predictions.csv")
LATEST_MODEL_FILEPATH = os.path.join(MODELS_DIR, "latest_model.pkl")

#class Trainer():
#    def __init__(self):
#        pass

def filter_features(xdf):
    df = xdf.copy().drop(columns=["passenger_id", "full_name", "ticket_id", "cabin_id",
        "sib_spouse_count", "parent_child_count", "salutation", "embarked_from_port"
    ])
    return df

def train_model():

    print("-----------------")
    print("LOADING DATA...")

    importer = Importer()
    xtrain, ytrain = filter_features(importer.xtrain), importer.ytrain
    xval, yval = filter_features(importer.xval), importer.yval
    xtest, xtest_passenger_ids = filter_features(importer.xtest), importer.xtest_passenger_ids
    print("TRAINING:", xtrain.shape, ytrain.shape)
    print("VALIDATION:", xval.shape, yval.shape)
    print("TEST:", xtest.shape, len(xtest_passenger_ids))

    print("-----------------")
    print("CONFIGURING SEARCH PARAMETERS...")

    pipeline = Pipeline(steps=[
        ("one_hot", OneHotEncoder(use_cat_names=True, cols=["gender"])),
        ("ordinal", OrdinalEncoder(cols=["ticket_class"], mapping=[{"col": "ticket_class", "mapping": {"UPPER":3, "MIDDLE":2, "LOWER":1}}])),
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(random_state=89))
    ])
    print(type(pipeline))

    search = GridSearchCV(estimator=pipeline, cv=5, verbose=10, return_train_score=True, n_jobs=-1, # -1 means using all processors
        scoring="accuracy", #"f1",
        param_grid={
            "imputer__strategy": ["mean", "median"],
            "classifier__min_samples_leaf": (0.02, 0.03, 0.06),
            "classifier__max_depth": (1, 2, 3, 4, 5),
        }
    )
    print(type(search))

    print("-----------------")
    print("TRAINING...")

    search.fit(xtrain, ytrain)

    best_estimator = search.best_estimator_ #> Pipeline
    one_hot = best_estimator.named_steps["one_hot"]
    ordinal = best_estimator.named_steps["ordinal"]
    model = search.best_estimator_.named_steps["classifier"]

    encoded_feature_names = ordinal.transform(one_hot.transform(xval)).columns
    encoded_feature_importances = Series(model.feature_importances_, encoded_feature_names).sort_values(ascending=False)
    encoded_feature_importances.name = "encoded_feature"

    print("-----------------")
    print("BEST SEARCH PARAMS:", search.best_params_)
    print("BEST SEARCH SCORE:", search.best_score_)

    print("-----------------")
    training_score = search.score(xtrain, ytrain)
    print("TRAINING SCORE:", training_score)

    print("-----------------")
    validation_score = search.score(xval, yval)
    print("VALIDATION SCORE:", validation_score)

    print("-----------------")
    print("VALIDATION REPORT:")
    yval_predicted = search.predict(xval)
    validation_report = classification_report(yval, yval_predicted)
    print(validation_report)

    print("-----------------")
    print("FEATURE IMPORTANCES:")
    print(encoded_feature_importances)

    print("-----------------")
    print("FEATURE IMPORTANCES VIZ:")
    print(os.path.abspath(FEATURES_GRAPH_FILEPATH))
    fig = px.bar(orientation="h", # horizontal flips x and y params below, but not labels
        x=encoded_feature_importances.values,
        y=encoded_feature_importances.keys(),
        labels={"x": "Relative Importance", "y": "Feature Name"}
    )
    #fig.show()
    fig.write_image(FEATURES_GRAPH_FILEPATH) # requires "orca" dependency. see: https://plot.ly/python/static-image-export/

    print("-----------------")
    print("DECISION TREE VIZ:")
    print(os.path.abspath(TREE_GRAPH_FILEPATH))
    class_names = [{0:"Not Survive", 1:"Survive"}[class_name] for class_name in search.classes_] # [0, 1]
    dot_data = export_graphviz(model,
        out_file=None,
        max_depth=3,
        feature_names=encoded_feature_names,
        class_names=class_names, # expects class names to be strings
        impurity=False,
        filled=True,
        proportion=True,
        rounded=True
    )
    graph = Source(dot_data)
    png_bytes = graph.pipe(format="png")
    with open(TREE_GRAPH_FILEPATH, "wb") as f:
        f.write(png_bytes)

    print("-----------------")
    print("SAVING RESULTS:")
    print(os.path.abspath(SEARCH_RESULTS_FILEPATH))
    results = {
        "pipeline_steps": [step_name for step_name, obj in best_estimator.steps],
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "training_score": training_score,
        "validation_score": validation_score,
        "encoded_feature_importances": encoded_feature_importances.to_dict(),
    }
    with open(SEARCH_RESULTS_FILEPATH, "w") as f:
        f.write(json.dumps(results))

    print("-----------------")
    print("SAVING MODEL:")
    print(os.path.abspath(LATEST_MODEL_FILEPATH))
    with open(LATEST_MODEL_FILEPATH, "wb") as f:
        pickle.dump(search, f)

    print("-----------------")
    print("GENERATING SUBMISSION FILE:")
    print(os.path.abspath(PREDICTIONS_FILEPATH))

    ytest_predictions = search.predict(xtest)
    ytest_predictions_df = DataFrame({"PassengerId": xtest_passenger_ids, "Survived": ytest_predictions})
    ytest_predictions_df.to_csv(PREDICTIONS_FILEPATH, index=False)

    return search

#def load_model():
#    print("-----------------")
#    print("LOADING SAVED MODEL FROM FILE:")
#    print(os.path.abspath(LATEST_MODEL_FILEPATH))
#
#    with open(LATEST_MODEL_FILEPATH, "rb") as model_file:
#        saved_model = pickle.load(model_file)
#    return saved_model

if __name__ == "__main__":

    train_model()
