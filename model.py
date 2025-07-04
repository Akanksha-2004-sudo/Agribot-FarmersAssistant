import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('encoders', exist_ok=True)
    
    # Load dataset
    try:
        data = pd.read_csv('data/Crop_recommendation.csv')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found. Please ensure the file exists at data/Crop_recommendation.csv")

    # Verify column names
    print("Dataset columns:", data.columns.tolist())

    # Data preprocessing
    data.rename(columns={
        'State': 'State_Name',
        'Nitrogen': 'N',
        'Phosphorus': 'P',
        'Potassium': 'K',
        'pH_value': 'pH',
        'Rain_mm': 'rainfall',
        'Temp_C': 'temperature'
    }, inplace=True)

    # Encode categorical columns
    label_encoders = {
        'state': LabelEncoder().fit(data['State_Name']),
        'crop_type': LabelEncoder().fit(data['Crop_Type']),
        'crop': LabelEncoder().fit(data['Crop'])
    }
    data.drop(['Unnamed: 0', 'Area_in_hectares', 'Production_in_tons', 'Yield_ton_per_hec'], 
          axis=1, inplace=True)
    data['State_Name'] = label_encoders['state'].transform(data['State_Name'])
    data['Crop_Type'] = label_encoders['crop_type'].transform(data['Crop_Type'])
    data['Crop'] = label_encoders['crop'].transform(data['Crop'])

    # Features and target
    X = data[['State_Name', 'Crop_Type', 'N', 'P', 'K', 'pH', 'rainfall', 'temperature']]
    y = data['Crop']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoders['crop'].classes_))

    # Save artifacts
    joblib.dump(model, 'model/crop_recommendation_model.pkl')
    for name, encoder in label_encoders.items():
        joblib.dump(encoder, f'encoders/{name}_encoder.pkl')
    
    print("Model training complete and artifacts saved!")
    print(data.isnull().sum())

if __name__ == "__main__":
    train_model()