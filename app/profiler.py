
import os
import pandas
from pandas_profiling import ProfileReport
#import webbrowser

from app.importer import training_and_validation_df

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TRAINING_PROFILE_FILEPATH = os.path.join(REPORTS_DIR, "passengers_profile.html")

if __name__ == "__main__":

    df = training_and_validation_df()

    profile = ProfileReport(df, title="Passengers DataFrame (Training)", html={"style":{"full_width":True}})
    print(profile)
    profile.to_file(output_file=TRAINING_PROFILE_FILEPATH)
    #webbrowser.open(os.path.abspath(TRAINING_PROFILE_FILEPATH))

    #> RESULTS ...

    #> Name has a high cardinality: 891 distinct values
    #> Ticket has a high cardinality: 681 distinct values
    #> Cabin has a high cardinality: 147 distinct values

    #> Age has 177 (19.9%) missing values
    #> Cabin has 687 (77.1%) missing values

    #> SibSp has 608 (68.2%) zeros
    #> Parch has 678 (76.1%) zeros
    #> Fare has 15 (1.7%) zeros
