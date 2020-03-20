
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
    # overwriting vals:
    df["ticket_class"] = df["ticket_class"].transform(parse_class)
    df["embarked_from_port"] = df["embarked_from_port"].transform(parse_port)
    # engineering new cols:
    df["marital_status"] = df["full_name"].transform(parse_marital_status)
    df["salutation"] = df["full_name"].transform(parse_salutation)

    print("-------------------")
    print("TRAINING / VALIDATION DATA...")
    print("-------------------")
    print(df.head())
    return df

def parse_class(class_num):
    TICKET_CLASS_MAP = {1:"UPPER", 2:"MIDDLE", 3:"LOWER"}
    try:
        class_name = TICKET_CLASS_MAP[class_num]
    except KeyError as err:
        print("CLASS PARSER ERR", err, "ORIGINAL VAL:", type(class_num), class_num)
        class_name = None
    return class_name

def parse_port(port_abbrev):
    PORTS_MAP = {"C":"Cherbourg".upper(), "Q":"Queenstown".upper(), "S":"Southampton".upper()}
    try:
        port_name = PORTS_MAP[port_abbrev]
    except KeyError as err:
        print("PORT PARSER ERR", err, "ORIGINAL VAL:", type(port_abbrev), port_abbrev)
        port_name = None
    return port_name

def parse_marital_status(full_name):
    return ("Mr." in full_name or "Mrs." in full_name)

def parse_salutation(full_name):
    if "Mr." in full_name:
        sal = "MISTER"
    elif "Mrs." in full_name:
        sal = "MRS"
    elif "Master." in full_name:
        sal = "MASTER"
    elif "Miss." in full_name:
        sal = "MISS"
    elif "Dr." in full_name:
        sal = "DOCTOR"
    elif "Rev." in full_name:
        sal = "REVERAND"
    elif ("Col." in full_name or "Major." in full_name or "Capt." in full_name):
        sal = "MILITARY"
    elif "Mme." in full_name:
        sal = "MADAME"
    elif "Mlle." in full_name:
        sal = "MADEMOISELLE"
    elif "Sir." in full_name:
        sal = "SIR"
    elif "Lady." in full_name:
        sal = "LADY"
    elif "Countess." in full_name:
        sal = "COUNTESS"
    elif "Ms." in full_name:
        sal = "MS"
    elif "Don." in full_name:
        sal = "DON"
    else:
        print("SALUTATION PARSER ERROR", full_name, type(full_name))
        sal = None
    return sal

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
