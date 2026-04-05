import pandas as pd
import sys
from pymongo import MongoClient, errors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_predictive_model():
    print("=" * 65)
    print("   MACHINE LEARNING MODEL TRAINING - PREDICTIVE MAINTENANCE   ")
    print("=" * 65)

    # 1. Database Connection
    print("\n[1/5] Initializing connection to MongoDB Atlas...")
    mongo_uri = "mongodb+srv://databasesensor:1234mongo@navdeep.p32wk6a.mongodb.net/?appName=navdeep"
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test the connection
        db = client["printer_maintenance"]
        collection = db["sensor_data_ml"]
        print("[SUCCESS] Database connection established.")
    except errors.ConnectionFailure as e:
        print(f"[ERROR] Failed to connect to MongoDB: {e}")
        sys.exit(1)

    # 2. Data Extraction
    print("\n[2/5] Fetching dataset from cloud storage...")
    data = list(collection.find())
    
    if len(data) < 50:
        print("[ERROR] Insufficient data. Minimum 50 records required for training.")
        print("        Please run the data ingestion stream to populate the database.")
        sys.exit(1)

    df = pd.DataFrame(data)
    print(f"[SUCCESS] Loaded {len(df)} operational records.")

    # 3. Data Preprocessing
    print("\n[3/5] Preprocessing features and target variables...")
    required_features = ['air_temp', 'proc_temp', 'rpm', 'torque', 'wear', 'power_w']
    
    # Validate columns
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required features in dataset: {missing_cols}")
        sys.exit(1)

    X = df[required_features]
    y = df['label']  # 0: Normal, 1: Failure

    # Split dataset: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 4. Model Training & Evaluation
    print("\n[4/5] Training Random Forest Classifier...")
    # Utilizing class_weight='balanced' to handle potential class imbalances
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Performance Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n[SUCCESS] Model Training Completed!")
    print("-" * 65)
    print(f"Model Accuracy Score : {accuracy * 100:.2f}%")
    print("-" * 65)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal Operation (0)', 'System Failure (1)']))
    print("-" * 65)

    # 5. Export Model
    print("\n[5/5] Exporting trained predictive model...")
    model_filename = 'printer_model.pkl'
    joblib.dump(model, model_filename)
    print(f"[SUCCESS] Model successfully serialized and saved as '{model_filename}'")
    print("=" * 65)

if __name__ == "__main__":
    train_predictive_model()
