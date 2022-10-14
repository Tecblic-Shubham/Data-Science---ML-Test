# Importing Required Libraries
import pandas as pd

# Read CSV
def get_data():
    df = pd.read_csv("./dataset/dataset.csv")
    return df

# Function to get attributes of the procedure
def get_procedure_attributes(procedure_id = None):
    df = get_data()
    column_list = df.columns[:-4]
    if procedure_id == None:
        # if id is not given, it will return random value
        return df[column_list].sample(n=1).to_dict()
    else:
        ind = df.index[df['encounter_id'] == procedure_id].tolist()
        return df[column_list].iloc[ind].to_dict()

# Function to identify Success/Failure
def get_procedure_success(procedure_id):
    df = get_data()
    ind = df.index[df['encounter_id'] == procedure_id].tolist()
    if df["hospital_death"][ind].values[0] == 0:
        # Success
        return True
    else:
        # Failure
        return False

# Return Procedure Outcomes
def get_procedure_outcomes(procedure_id):
    df = get_data()
    column_list = df.columns[-4:-2]
    ind = df.index[df['encounter_id'] == procedure_id].tolist()
    return df[column_list].iloc[ind].to_dict()
