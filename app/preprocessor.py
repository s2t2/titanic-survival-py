
import os
import pandas

CSV_FILEPATH = os.path.join(os.path.dirname(__file__), "..", "data", "titanic_train.csv")

if __name__ == "__main__":
    df = pandas.read_csv(CSV_FILEPATH)
    print(df.head())
