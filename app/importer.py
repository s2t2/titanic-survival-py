
import os
import pandas
from pprint import pprint

from sklearn.model_selection import train_test_split # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from app import DATA_DIR

TRAINING_AND_VALIDATION_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_trainval.csv")
TESTING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_test.csv")

class Importer():
    target_col = "survived"

    def __init__(self):
        # process data used for testing and submitting:
        self.testing_df_raw = pandas.read_csv(TESTING_DATA_FILEPATH).copy()
        self.xtest_passenger_ids = self.testing_df_raw["PassengerId"] # preserve now because this feature may be removed during processing
        self.testing_df = self.process(self.testing_df_raw) # assumes processing doesn't change the number or order of rows, but it may remove or add columns and change the values thereof
        self.xtest = self.testing_df # because the test set doesn't include the target column

        # process data used for training and evaluation:
        self.trainval_df_raw = pandas.read_csv(TRAINING_AND_VALIDATION_DATA_FILEPATH).copy()
        self.trainval_df = self.process(self.trainval_df_raw)
        self.training_df, self.validation_df = train_test_split(self.trainval_df, random_state=89,
            train_size=0.80, test_size=0.20,
            #stratify= self.training_df["survived"]
        )
        self.xtrain, self.ytrain = self.split_xy(self.training_df)
        self.xval, self.yval = self.split_xy(self.validation_df)

    @classmethod
    def split_xy(cls, passengers_df):
        df = passengers_df.copy()
        features = df.drop(columns=[cls.target_col]) # use all other features except the target, inplace is False by default
        labels = df[cls.target_col]
        return features, labels # x, y

    @classmethod
    def process(cls, passengers_df):
        """Adds columns and updates values. Does not / should not remove columns or rows."""
        df = passengers_df.copy()

        # renaming cols:
        df = df.rename(columns={
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
        })

        # overwriting vals:
        df["ticket_class"] = df["ticket_class"].transform(cls.parse_class)
        df["embarked_from_port"] = df["embarked_from_port"].transform(cls.parse_port)

        # engineering new cols:
        df["marital_status"] = df["full_name"].transform(cls.parse_marital_status)
        df["salutation"] = df["full_name"].transform(cls.parse_salutation)

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
