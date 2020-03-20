
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

    df["ticket_class"] = df["ticket_class"].transform(parse_class)
    df["embarked_from_port"] = df["embarked_from_port"].transform(parse_port)
    df["marital_status"] = df["full_name"].transform(parse_marital_status)
    df["salutation"] = df["full_name"].transform(parse_salutation)

    print("-------------------")
    print("TRAINING / VALIDATION DATA...")
    print("-------------------")
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

def parse_marital_status(original_val):
    return ("Mr." in original_val or "Mrs." in original_val)

def parse_salutation(full_name):
    if "Mr." in full_name:
        new_val = "MISTER"
    elif "Mrs." in full_name:
        new_val = "MRS"
    elif "Master." in full_name:
        new_val = "MASTER"
    elif "Miss." in full_name:
        new_val = "MISS"
    elif "Dr." in full_name:
        new_val = "DOCTOR"
    elif "Rev." in full_name:
        new_val = "REVERAND"
    elif ("Col." in full_name or "Major." in full_name or "Capt." in full_name):
        new_val = "MILITARY"
    elif ("Mme." in full_name):
        new_val = "MADAME"
    elif ("Mlle." in full_name):
        new_val = "MADEMOISELLE"
    elif ("Sir." in full_name):
        new_val = "SIR"
    elif ("Lady." in full_name):
        new_val = "LADY"
    elif ("Countess." in full_name):
        new_val = "COUNTESS"
    elif ("Ms." in full_name):
        new_val = "MS"
    elif ("Don." in full_name):
        new_val = "DON"
    else:
        print("SALUTATION PARSER ERROR", full_name, type(full_name))
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
