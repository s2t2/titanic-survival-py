# adapted from: https://github.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge/blob/master/module1-decision-trees/LS_DS_221.ipynb

import os

from category_encoders import OneHotEncoder # see: https://contrib.scikit-learn.org/categorical-encoding/onehot.html
from category_encoders import OrdinalEncoder # see: https://contrib.scikit-learn.org/categorical-encoding/ordinal.html
from sklearn.impute import SimpleImputer # see: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
from sklearn.preprocessing import StandardScaler # see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from graphviz import Source # see: https://pypi.org/project/graphviz/
from sklearn.tree import export_graphviz # see: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

from app import REPORTS_DIR
from app.importer import Importer

TREE_VIEW_FILEPATH = os.path.join(REPORTS_DIR, "decision_tree.png")

if __name__ == "__main__":

    importer = Importer()
    xtrain, ytrain, xval, yval = importer.training_and_validation_splits()

    #one_hot_encoder = OneHotEncoder(use_cat_names=True, cols=["gender",
    #    #"embarked_from_port", "salutation"
    #])
    #ordinal_encoder = OrdinalEncoder(cols=["ticket_class"], mapping=[{"col": "ticket_class", "mapping": {"UPPER":3, "MIDDLE":2, "LOWER":1}}])
    #imputer = SimpleImputer() # strategy="median"
    ##scaler = StandardScaler()
    #model = DecisionTreeClassifier(random_state=89, min_samples_leaf=.05) # max_depth=5 OR min_samples_leaf=.05

    pipeline = make_pipeline(
        OneHotEncoder(use_cat_names=True, cols=["gender"]),
        OrdinalEncoder(cols=["ticket_class"], mapping=[{"col": "ticket_class", "mapping": {"UPPER":3, "MIDDLE":2, "LOWER":1}}]),
        SimpleImputer(),
        DecisionTreeClassifier(random_state=89, min_samples_leaf=.05)
    )

    print("-----------------")
    print("TRAINING...")
    print("-----------------")
    #xtrain_one_hot_encoded = one_hot_encoder.fit_transform(xtrain)
    #xtrain_ordinal_encoded = ordinal_encoder.fit_transform(xtrain_one_hot_encoded)
    #xtrain_imputed = imputer.fit_transform(xtrain_ordinal_encoded)
    #model.fit(xtrain_imputed, ytrain)
    #print("MODEL CLASSES:", model.classes_)
    #print("ACCURACY (TRAINING):", model.score(xtrain_imputed, ytrain))
    pipeline.fit(xtrain, ytrain)
    print("MODEL CLASSES:", pipeline.classes_)
    print("ACCURACY (TRAINING):", pipeline.score(xtrain, ytrain))

    #xval_one_hot_encoded = one_hot_encoder.transform(xval)
    #xval_ordinal_encoded = ordinal_encoder.transform(xval_one_hot_encoded)
    #xval_imputed = imputer.transform(xval_ordinal_encoded)
    #print("ACCURACY (VALIDATION):", model.score(xval_imputed, yval))
    print("ACCURACY (VALIDATION):", pipeline.score(xval, yval))

    #
    # INSPECTION...
    #

    #encoded_columns = xval_ordinal_encoded.columns
    #class_names = [str(class_name) for class_name in model.classes_]
    model = pipeline.named_steps["decisiontreeclassifier"]
    one_hot = pipeline.named_steps["onehotencoder"]
    ordinal = pipeline.named_steps["ordinalencoder"]
    xval_encoded = one_hot.transform(xval)
    xval_encoded = ordinal.transform(xval_encoded)
    encoded_columns = xval_encoded.columns
    class_names = [str(class_name) for class_name in model.classes_]
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
