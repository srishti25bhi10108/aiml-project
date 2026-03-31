import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def load_data(filepath="city_day.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["AQI_Bucket"])
    features = ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3", "Benzene"]
    df = df[features + ["AQI_Bucket"]].dropna()
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df["AQI_Bucket"])
    return X, y, le

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    print(f"  Model trained — Accuracy: {acc:.1f}%")
    return model, scaler

def ask_num(prompt, min_val, max_val):
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a number.")

def get_input():
    print("\n--- Enter Air Quality Readings ---")
    print("  (You can find these on apps like IQAir or AQI India)\n")
    pm25    = ask_num("  PM2.5  (0-500)  : ", 0, 500)
    pm10    = ask_num("  PM10   (0-600)  : ", 0, 600)
    no      = ask_num("  NO     (0-100)  : ", 0, 100)
    no2     = ask_num("  NO2    (0-200)  : ", 0, 200)
    co      = ask_num("  CO     (0-50)   : ", 0, 50)
    so2     = ask_num("  SO2    (0-100)  : ", 0, 100)
    o3      = ask_num("  O3     (0-300)  : ", 0, 300)
    benzene = ask_num("  Benzene(0-50)   : ", 0, 50)
    return {
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2,
        "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene
    }

def health_effects(bucket):
    effects = {
        "Good": {
            "risk"    : "Very Low",
            "diseases": "No significant health risks.",
            "lungs"   : "Air is clean. Lungs are safe.",
            "advice"  : "Enjoy outdoor activities freely."
        },
        "Satisfactory": {
            "risk"    : "Low",
            "diseases": "Minor breathing discomfort for very sensitive people.",
            "lungs"   : "Minimal lung irritation possible.",
            "advice"  : "Generally safe. Sensitive individuals may limit prolonged outdoor exposure."
        },
        "Moderate": {
            "risk"    : "Moderate",
            "diseases": "Asthma, allergies, eye and throat irritation.",
            "lungs"   : "Mild lung irritation. Can worsen existing respiratory conditions.",
            "advice"  : "Sensitive groups (children, elderly, asthma patients) should reduce outdoor time."
        },
        "Poor": {
            "risk"    : "High",
            "diseases": "Asthma attacks, Bronchitis, respiratory infections.",
            "lungs"   : "Significant lung irritation. Prolonged exposure causes lung damage.",
            "advice"  : "Wear a mask outdoors. Avoid exercise outside. Keep windows closed."
        },
        "Very Poor": {
            "risk"    : "Very High",
            "diseases": "Chronic Bronchitis, Lung disease, heart complications.",
            "lungs"   : "Serious lung damage with prolonged exposure. Reduced lung capacity.",
            "advice"  : "Stay indoors. Use air purifier. Wear N95 mask if going out."
        },
        "Severe": {
            "risk"    : "Hazardous",
            "diseases": "Lung Cancer, COPD, Cardiovascular disease, stroke risk.",
            "lungs"   : "Severe lung damage. Can cause permanent respiratory problems.",
            "advice"  : "Do NOT go outside. Seal windows. Seek medical help if breathing issues occur."
        }
    }
    return effects.get(bucket, effects["Moderate"])

def show_results(bucket, effects):
    print("\n========================================")
    print("       AIR QUALITY RISK REPORT")
    print("========================================")
    print(f"  AQI Category     : {bucket}")
    print(f"  Health Risk      : {effects['risk']}")
    print(f"\n  Lung Impact      : {effects['lungs']}")
    print(f"\n  Disease Risks    : {effects['diseases']}")
    print(f"\n  Advice           : {effects['advice']}")
    print("========================================")
    print("  For educational purposes only.")
    print("========================================\n")


print("\n========================================")
print("  AIR QUALITY & POLLUTION RISK ANALYZER")
print("  Powered by Machine Learning")
print("========================================")

print("\nLoading dataset...")
try:
    X, y, le = load_data("city_day.csv")
except FileNotFoundError:
    print("\nERROR: city_day.csv not found.")
    print("Make sure city_day.csv is in the same folder as this script.")
    exit()

print("Training model...")
model, scaler = train(X, y)

while True:
    data = get_input()
    df = pd.DataFrame([data])
    pred = model.predict(scaler.transform(df))[0]
    bucket = le.inverse_transform([pred])[0]
    effects = health_effects(bucket)
    show_results(bucket, effects)

    again = input("  Check another reading? (yes/no): ").strip().lower()
    if again not in ["yes", "y"]:
        print("\n  Goodbye!\n")
        break