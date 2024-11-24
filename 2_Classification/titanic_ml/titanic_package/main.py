# main.py

from titanic_model import TitanicModel

def run_pipeline():
    # Initialize the TitanicModel class
    titanic_model = TitanicModel()
    
    # Step-by-step pipeline
    titanic_model.load_data()
    titanic_model.preprocess_data()
    titanic_model.feature_engineering()
    titanic_model.train_model()
    metrics = titanic_model.evaluate_model()
    
    print("Pipeline execution completed successfully.")
    return metrics

if __name__ == "__main__":
    run_pipeline()
