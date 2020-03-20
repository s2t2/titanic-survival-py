
import os
import pandas
from pandas_profiling import ProfileReport
import webbrowser

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_train.csv")
TRAINING_PROFILE_FILEPATH = os.path.join(REPORTS_DIR, "passengers_profile.html")
#TESTING_DATA_FILEPATH = os.path.join(DATA_DIR, "passengers_test.csv")

if __name__ == "__main__":
    print("-------------------")
    print("TRAINING DATA...")
    print("-------------------")
    passengers_df = pandas.read_csv(TRAINING_DATA_FILEPATH)
    print(passengers_df.head())

    passengers_profile = ProfileReport(passengers_df, title="Passengers DataFrame (Training)", html={"style":{"full_width":True}})
    print(passengers_profile)
    passengers_profile.to_file(output_file=TRAINING_PROFILE_FILEPATH)
    webbrowser.open(TRAINING_PROFILE_FILEPATH)

    exit()

    # xtrain
    # ytrain
    # xval
    # yval

    #train, val = train_test_split(train,
    #    train_size=0.80,
    #    test_size=0.20,
    #    #stratify=train['status_group'], )
    #    random_state=89
    #)
    #test = pandas.read_csv(TEST_CSV_FILEPATH)
    #print(train.shape, val.shape, test.shape)
