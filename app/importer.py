
import os
import pandas

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_train.csv")
COLUMNS_MAP = {
    "PassengerId": "passenger_id",
    "Survived": "survived",
    "Pclass": "ticket_class",
    "Name": "full_name",
    "Sex": "gender",
    "Age": "age_yrs",
    "SibSp": "sib_spouse_count",
    "Parch": "parent_child_count",
    "Ticket": "ticket_id",
    "Fare": "fare_usd",
    "Cabin": "cabin_id",
    "Embarked": "embarked_from_port"
}

#class Importer():
#    def __init__(self):
#        self.

def training_and_validation_df():
    df = pandas.read_csv(TRAINING_DATA_FILEPATH)
    df = df.rename(columns=COLUMNS_MAP)
    print("-------------------")
    print("TRAINING / VALIDATION DATA...")
    print("-------------------")
    print(df.head())
    return df

#def training_and_validation_splits():
#    train, val = train_test_split(training_and_validation_df(), random_state=89
#        train_size=0.80, test_size=0.20, #stratify=train[""]
#    )
#    # xtrain
#    # ytrain
#    # xval
#    # yval
#    return xtrain, ytrain, xval, yval


if __name__ == "__main__":

    passengers_df = training_and_validation_df()
