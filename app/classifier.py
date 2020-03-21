# adapted from: https://github.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge

import os
import json
from pprint import pprint

from category_encoders import OneHotEncoder # see: https://contrib.scikit-learn.org/categorical-encoding/onehot.html
from category_encoders import OrdinalEncoder # see: https://contrib.scikit-learn.org/categorical-encoding/ordinal.html
from sklearn.impute import SimpleImputer # see: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
from sklearn.preprocessing import StandardScaler # see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.tree import DecisionTreeClassifier # see: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline # see: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
from sklearn.model_selection import GridSearchCV # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import RandomizedSearchCV # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
from scipy.stats import randint, uniform

from graphviz import Source # see: https://pypi.org/project/graphviz/
from sklearn.tree import export_graphviz # see: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

from app import REPORTS_DIR
from app.importer import Importer

TREE_VIEW_FILEPATH = os.path.join(REPORTS_DIR, "decision_tree.png")
SEARCH_RESULTS_FILEPATH = os.path.join(REPORTS_DIR, "search_results.json")

if __name__ == "__main__":

    importer = Importer()
    xtrain, ytrain, xval, yval = importer.training_and_validation_splits()

    pipeline = make_pipeline(
        OneHotEncoder(use_cat_names=True, cols=["gender"]),
        OrdinalEncoder(cols=["ticket_class"], mapping=[{"col": "ticket_class", "mapping": {"UPPER":3, "MIDDLE":2, "LOWER":1}}]),
        SimpleImputer(),
        DecisionTreeClassifier(random_state=89, min_samples_leaf=.05)
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "simpleimputer__strategy": ["mean", "median"],
            "decisiontreeclassifier__min_samples_leaf": (0.02, 0.2),
            "decisiontreeclassifier__max_depth": (1, 20),
        },
        scoring="accuracy", # "neg_mean_absolute_error",
        n_jobs=-1, # -1 means using all processors
        cv=5,
        verbose=10,
        return_train_score=True,
    )
    #search = RandomizedSearchCV(
    #    estimator=pipeline,
    #    param_distributions={
    #        "simpleimputer__strategy": ["mean", "median"],
    #        "decisiontreeclassifier__min_samples_leaf": uniform(0.02, 0.2),
    #        "decisiontreeclassifier__max_depth": uniform(1, 20),
    #    },
    #    scoring="accuracy", # "neg_mean_absolute_error",
    #    n_jobs=-1, # -1 means using all processors
    #    cv=5,
    #    verbose=10,
    #    return_train_score=True,
    #)

    print("-----------------")
    print("TRAINING...")
    print("-----------------")
    #pipeline.fit(xtrain, ytrain)
    #print("MODEL CLASSES:", pipeline.classes_)
    #print("ACCURACY (TRAINING):", pipeline.score(xtrain, ytrain))
    #print("ACCURACY (VALIDATION):", pipeline.score(xval, yval))
    search.fit(xtrain, ytrain)
    print("-----------------")
    print("MODEL CLASSES:", search.classes_)
    training_accy = search.score(xtrain, ytrain)
    print("ACCURACY (TRAINING):", training_accy)
    val_accy = search.score(xval, yval)
    print("ACCURACY (VALIDATION):", val_accy)
    print("BEST PARAMS:")
    pprint(search.best_params_)
    print("BEST SCORE:", search.best_score_)
    with open(SEARCH_RESULTS_FILEPATH, "w") as f:
        f.write(json.dumps({
            "accy_training": training_accy,
            "accy_validation": val_accy,
            "best_score": search.best_score_,
            "best_params": search.best_params_
        }))

    #
    # INSPECTION...
    #

    print("-----------------")
    print("GRAPHING DECISION TREE:")
    print(os.path.abspath(TREE_VIEW_FILEPATH))
    #model = pipeline.named_steps["decisiontreeclassifier"]
    #one_hot = pipeline.named_steps["onehotencoder"]
    #ordinal = pipeline.named_steps["ordinalencoder"]
    model = search.best_estimator_.named_steps["decisiontreeclassifier"]
    one_hot = search.best_estimator_.named_steps["onehotencoder"]
    ordinal = search.best_estimator_.named_steps["ordinalencoder"]
    xval_encoded = one_hot.transform(xval)
    xval_encoded = ordinal.transform(xval_encoded)

    encoded_columns = xval_encoded.columns
    class_names = [str(class_name) for class_name in search.classes_]
    dot_data = export_graphviz(model,
        out_file=None,
        max_depth=3,
        feature_names=encoded_columns,
        class_names=class_names,
        impurity=False,
        filled=True,
        proportion=True,
        rounded=True
    )
    graph = Source(dot_data)
    png_bytes = graph.pipe(format="png")
    with open(TREE_VIEW_FILEPATH, "wb") as f:
        f.write(png_bytes)
