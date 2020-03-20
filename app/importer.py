
import os
import pandas

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_train.csv")

#class Importer():
#    def __init__(self):
#        self.

def training_and_validation_df():
    df = pandas.read_csv(TRAINING_DATA_FILEPATH)
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
    df = df.rename(columns=COLUMNS_MAP)
    print("-------------------")
    print("TRAINING / VALIDATION DATA...")
    print("-------------------")

    df["ticket_class"] = df["ticket_class"].transform(parse_class)
    df["embarked_from_port"] = df["embarked_from_port"].transform(parse_port)
    print(df.head())
    return df

def parse_class(original_val):
    TICKET_CLASS_MAP = {1:"UPPER", 2:"MIDDLE", 3:"LOWER"}
    try:
        new_val = TICKET_CLASS_MAP[original_val]
    except KeyError as err:
        print("CLASS PARSER ERR", err, "ORIGINAL VAL:", type(original_val), original_val)
        new_val = None
    return new_val

def parse_port(original_val):
    PORTS_MAP = {"C":"Cherbourg".upper(), "Q":"Queenstown".upper(), "S":"Southampton".upper()}
    try:
        new_val = PORTS_MAP[original_val]
    except KeyError as err:
        print("PORT PARSER ERR", err, "ORIGINAL VAL:", type(original_val), original_val)
        new_val = None
    return new_val







def training_and_validation_splits():
    train, val = train_test_split(training_and_validation_df(), random_state=89,
        train_size=0.80, test_size=0.20, #stratify=train[""]
    )
    breakpoint()
    # xtrain
    # ytrain
    # xval
    # yval
    return xtrain, ytrain, xval, yval


if __name__ == "__main__":

    passengers_df = training_and_validation_df()
