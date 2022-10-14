import joblib 
import argparse
# Import required libraries
import pandas as pd
import cms_procedures

def predict_(procedure_id):
    # importing mode
    clf = joblib.load("./Models/GradientBoostingClassifier()_accuracy_0.9244384676891764.pkl")    
    
    # getting data from library
    data_attributes = pd.DataFrame(cms_procedures.get_procedure_attributes(procedure_id))
    data_outcomes = pd.DataFrame(cms_procedures.get_procedure_outcomes(procedure_id))
    
    #combinng the data
    data_pred = pd.concat([data_attributes, data_outcomes], axis = 1)

    # predicting for sucess or failure
    predictions = clf.predict(data_pred)
    if predictions == 0:
        print("Success")
    else:
        print("Failure")
    return predictions
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # taking argument for procedure id
    parser.add_argument(
    "--p_id",
    type=int,
    default= 71599
    )

    args = parser.parse_args()

    predict_(
        procedure_id = args.p_id
    )