import pandas as pd
import os

# 🔧 File paths to valid
raw_dir = "D:/Bruno_Entropy_Project/data/raw"
files_to_check = [
    "IceCube_Gold_Bronze_Recent_event.csv",
    "IceCube_Gold_Bronze_Events.csv"
]

# ✅ Required normalized header
expected_columns = [
    "Run_Event_ID", "Revision", "Date_UTC", "Time_UTC", "Alert_Type",
    "RA_deg", "Dec_deg", "Error90_arcmin", "Error50_arcmin",
    "Energy_TeV", "Signalness", "FAR_per_year", "Notes"
]

# 🛠️ Validation routine
def validate_csv_columns(file_path):
    print(f"🔍 Validating: {file_path}")
    try:
        df = pd.read_csv(file_path)
        cols = df.columns.tolist()

        # ✔ Check for column match
        if cols == expected_columns:
            print("✅ Header is valid and normalized.")
        else:
            print("❌ Header mismatch!")
            print("→ Found:   ", cols)
            print("→ Expected:", expected_columns)

        # 🧪 Check for nulls in critical fields
        critical = ["RA_deg", "Dec_deg", "Date_UTC", "Energy_TeV"]
        missing = df[critical].isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values in critical fields.\n")
        else:
            print("⚠️ Missing values found:")
            print(missing, "\n")

    except Exception as e:
        print(f"❌ Error reading file: {e}\n")

# 🚀 Run validation on each file
for filename in files_to_check:
    validate_csv_columns(os.path.join(raw_dir, filename))
