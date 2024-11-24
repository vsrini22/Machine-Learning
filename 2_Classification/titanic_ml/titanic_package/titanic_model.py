# titanic_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import config  # Import the configuration file

class TitanicModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """Loads the Titanic dataset."""
        self.data = pd.read_csv(config.DATA_PATH)
        print("Data loaded successfully")

    def preprocess_data(self):
        """Preprocesses the data by filling missing values and encoding categorical features."""
        # Fill missing values
        self.data['Age'].fillna(self.data['Age'].mean(), inplace=True)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        self.data['Fare'].fillna(self.data['Fare'].mean(), inplace=True)

        # Encode categorical variables
        self.data = pd.get_dummies(self.data, columns=['Sex', 'Embarked'], drop_first=True)
        print("Data preprocessing complete")

    def feature_engineering(self):
        """Extracts features and the target variable."""
        features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']
        X = self.data[features]
        y = self.data['Survived']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config.DATA_SPLIT["test_size"], random_state=config.DATA_SPLIT["random_state"]
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Feature engineering complete")

    def train_model(self):
        """Trains a Random Forest classifier."""
        self.model = RandomForestClassifier(**config.MODEL_PARAMS)
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete")

    def evaluate_model(self):
        """Evaluates the model using accuracy, precision, recall, and F1 score."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"Model Evaluation:\n Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1 Score: {f1}")
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
