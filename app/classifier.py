
# see:
# https://contrib.scikit-learn.org/categorical-encoding/onehot.html
# https://contrib.scikit-learn.org/categorical-encoding/ordinal.html
from category_encoders import OneHotEncoder, OrdinalEncoder

# see: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
from sklearn.impute import SimpleImputer

#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#from sklearn.pipeline import make_pipeline

from app.importer import Importer

if __name__ == "__main__":

    importer = Importer()
    xtrain, ytrain, xval, yval = importer.training_and_validation_splits()

    # pipeline:
    one_hot_encoder = OneHotEncoder(use_cat_names=True, cols=["gender", "embarked_from_port", "salutation"])
    ordinal_encoder = OrdinalEncoder(cols=["ticket_class"], mapping=[{"col": "ticket_class", "mapping": {"UPPER":3, "MIDDLE":2, "LOWER":1}}])
    imputer = SimpleImputer()
    model = DecisionTreeClassifier(random_state=89)
    #pipeline = make_pipeline(one_hot_encoder, ordinal_encoder, imputer, model)
    #pipeline.fit(xtrain, ytrain)

    xtrain_one_hot_encoded = one_hot_encoder.fit_transform(xtrain)
    xtrain_ordinal_encoded = ordinal_encoder.fit_transform(xtrain_one_hot_encoded)
    xtrain_imputed = imputer.fit_transform(xtrain_ordinal_encoded)
    model.fit(xtrain_imputed, ytrain)
    print("Accuracy (Training)", model.score(xtrain_imputed, ytrain))

    xval_one_hot_encoded = one_hot_encoder.transform(xval)
    xval_ordinal_encoded = ordinal_encoder.transform(xval_one_hot_encoded)
    xval_imputed = imputer.transform(xval_ordinal_encoded)
    print("Accuracy (Validation):", model.score(xval_imputed, yval))
