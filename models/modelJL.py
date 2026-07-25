import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

CSV_path = os.path.join("data", "process","landmarks.csv")
MODEL_dir = "models"
MODEL_path = os.path.join(MODEL_dir, "randomForest_model.pkl")

#Uses simple randomforest model to train I would say try to use a different model if you can to get the best accuracy 
#if i have the time i'll work on a xgboost model
def main():
    if not os.path.exists(CSV_path):
        print(f"CSV file not found at {CSV_path}")
        return
    df = pd.read_csv(CSV_path)
    x = df.drop(columns=["label"])
    y = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify = y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
    os.makedirs(MODEL_dir, exist_ok=True)
    joblib.dump(model, MODEL_path)
    print(f"Model saved at {MODEL_path}")

if __name__ == "__main__":
    main()