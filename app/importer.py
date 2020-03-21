
import os
import pandas
from pprint import pprint

# see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_train.csv")
TESTING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_test.csv")

class Importer():
    def __init__(self):
        self.training_df_raw = pandas.read_csv(TRAINING_DATA_FILEPATH).copy()
        self.testing_df_raw = pandas.read_csv(TESTING_DATA_FILEPATH).copy()
        self.training_df = self.process(self.training_df_raw)
        self.testing_df = self.process(self.testing_df_raw)

    def training_and_validation_splits(self):
        train, val = train_test_split(self.training_df, random_state=89, train_size=0.80, test_size=0.20,
            #stratify= self.training_df
        )
        #print(type(train), len(train)) #> <class 'pandas.core.frame.DataFrame'> 712
        #print(type(val), len(val)) #> <class 'pandas.core.frame.DataFrame'> 179
        target_col = "survived"
        xtrain = train.drop(columns=[target_col]) # use all other features except the target, inplace is False by default
        ytrain = train[target_col]
        xval = val.drop(columns=[target_col]) # use all other features except the target, inplace is False by default
        yval = val[target_col]
        return xtrain, ytrain, xval, yval

    @classmethod
    def process(cls, passengers_df):
        df = passengers_df.copy()

        # renaming cols:
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
        df["ticket_class"] = df["ticket_class"].transform(cls.parse_class)
        df["embarked_from_port"] = df["embarked_from_port"].transform(cls.parse_port)

        # engineering new cols:
        df["marital_status"] = df["full_name"].transform(cls.parse_marital_status)
        df["salutation"] = df["full_name"].transform(cls.parse_salutation)

        # dropping cols:
        df = df.drop(columns=["ticket_id", "cabin_id", "passenger_id", "full_name", "sib_spouse_count", "parent_child_count"])

        return df

    @staticmethod
    def parse_class(class_num):
        TICKET_CLASS_MAP = {1:"UPPER", 2:"MIDDLE", 3:"LOWER"}
        try:
            class_name = TICKET_CLASS_MAP[class_num]
        except KeyError as err:
            print("CLASS PARSER ERR", err, "ORIGINAL VAL:", type(class_num), class_num)
            class_name = None
        return class_name

    @staticmethod
    def parse_port(port_abbrev):
        PORTS_MAP = {"C":"Cherbourg".upper(), "Q":"Queenstown".upper(), "S":"Southampton".upper()}
        try:
            port_name = PORTS_MAP[port_abbrev]
        except KeyError as err:
            print("PORT PARSER ERR", err, "ORIGINAL VAL:", type(port_abbrev), port_abbrev)
            port_name = None #port_abbrev # None
        return port_name

    @staticmethod
    def parse_marital_status(full_name):
        return ("Mr." in full_name or "Mrs." in full_name)

    @staticmethod
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
            sal = "DOC/REV"
        elif "Rev." in full_name:
            sal = "DOC/REV"
        elif ("Col." in full_name or "Major." in full_name or "Capt." in full_name):
            sal = "MILITARY"
        elif "Mme." in full_name:
            sal = "MRS"
        elif "Mlle." in full_name:
            sal = "MISS" #"MADEMOISELLE"
        elif "Sir." in full_name:
            sal = "NOBILITY" # "SIR"
        elif "Lady." in full_name:
            sal = "NOBILITY" # "LADY"
        elif "Countess." in full_name:
            sal = "NOBILITY" # "LADY"
        elif "Ms." in full_name:
            sal = "MRS"
        elif "Don." in full_name:
            sal = "NOBILITY" # "SIR"
        elif "Dona." in full_name:
            sal = "NOBILITY" # "LADY"
        elif "Jonkheer." in full_name:
            sal = "NOBILITY" # "LADY"
        else:
            print("SALUTATION PARSER ERROR", full_name, type(full_name))
            sal = None
        return sal

if __name__ == "__main__":

    importer = Importer()

    print("-------------------")
    print("TRAINING DATA...")
    print("-------------------")
    print(importer.training_df.head())
    pprint(sorted(list(importer.training_df.columns)))
