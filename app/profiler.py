
import os
from pprint import pprint
#import webbrowser

import pandas
from pandas_profiling import ProfileReport

from app.importer import Importer
from app import REPORTS_DIR

TO_HTML = (os.getenv("TO_HTML", "True").title() == "True")

if __name__ == "__main__":

    importer = Importer()

    datasets = {
        "raw": importer.training_df_raw,
        "processed": importer.training_df
    }

    for k, df in datasets.items():
        print("------------")
        print(f"TRAINING DATASET ({k.upper()})")
        print("------------")

        profile = ProfileReport(df, title=f"Passengers Training Data ({k.title()})", html={"style":{"full_width":True}})
        print(type(profile))
        if TO_HTML:
            profile_path = os.path.join(REPORTS_DIR, f"passengers_profile_{k}.html")
            profile.to_file(output_file=profile_path)
            #webbrowser.open(os.path.abspath(TRAINING_PROFILE_FILEPATH))

        #> RESULTS ...

        desc = profile.get_description()
        #print(desc.keys()) #> ['table', 'variables', 'scatter', 'correlations', 'missing', 'messages', 'package']
        print("------------")
        print("MESSAGES:")
        print("------------")
        for message in desc["messages"]:
            print(message)

        #print("------------")
        #print("VARS:")
        #print("------------")
        #for k,v in desc["variables"].items():
        #    print(k.upper())
        #    print(v.keys())
        #    #pprint(v)
        #    print("---")
        #> dict_keys(['value_counts', 'value_counts_with_nan', 'value_counts_without_nan',
        #   'distinct_count_with_nan', 'distinct_count_without_nan', 'type', 'n', 'count', 'distinct_count',
        #   'n_unique', 'p_missing', 'n_missing', 'p_infinite', 'n_infinite',
        #   'is_unique', 'mode', 'p_unique', 'memory_size', 'mean', 'std', 'variance', 'min', 'max',
        #   'kurtosis', 'skewness', 'sum', 'mad', 'n_zeros',
        #   'histogram_data', 'scatter_data', 'chi_squared', 'range', '5%', '25%', '50%', '75%', '95%', 'iqr',
        #   'cv', 'p_zeros', 'histogram_bins', 'histogram_bins_bayesian_blocks'])

        #print("------------")
        #print("CORRELATIONS:")
        #print("------------")
        #for k,v in desc["correlations"].items():
        #    print(f"{k.upper()} CORRELATION:")
        #    print(v)
        #    print("---")

        #> Name has a high cardinality: 891 distinct values
        #> Ticket has a high cardinality: 681 distinct values
        #> Cabin has a high cardinality: 147 distinct values

        #> Age has 177 (19.9%) missing values
        #> Cabin has 687 (77.1%) missing values

        #> SibSp has 608 (68.2%) zeros
        #> Parch has 678 (76.1%) zeros
        #> Fare has 15 (1.7%) zeros

        #print(profile.__dict__.keys()) #> ['date_start', 'sample', 'title', 'description_set', 'date_end', 'report']


        #variables = results["variables"]
        #print("VARS", variables.keys())
        #for k,v in variables.items():
        #    print(k)
