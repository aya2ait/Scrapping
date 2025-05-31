"""
E-commerce Pipeline with Base64 Encoded Parameters
"""
from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Output, Dataset, Model, Metrics
import base64
import json
from typing import NamedTuple

# Helper function to encode configs
def encode_config(config_dict):
    """Encode a config dictionary to base64"""
    return base64.b64encode(json.dumps(config_dict).encode()).decode()

@component(base_image="python:3.9-slim")
def ml_scoring_component(
    stored_data: str,
    stored_products: int,
    ml_config_b64: str,
    trained_model: Output[Model],
    scored_data: Output[Dataset],
    ml_metrics: Output[Metrics]
) -> NamedTuple('MLOutput', [('model_accuracy', float), ('top_k_count', int)]):
    """
    ML Scoring component with base64 encoded config
    """
    import json
    import pandas as pd
    import numpy as np
    import joblib
    import os
    import base64
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from collections import namedtuple
    
    print("🧠 Starting ML scoring...")
    
    # Decode base64 config
    config = json.loads(base64.b64decode(ml_config_b64).decode())
    print(f"Decoded config: {config}")
    
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),
        'target': np.random.rand(n_samples)
    })
    
    # Train a simple model
    X = df[['feature1', 'feature2']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=config.get('n_estimators', 100))
    model.fit(X_train, y_train)
    
    # Score the model
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, trained_model.path)
    
    # Save scored data
    df['score'] = model.predict(X)
    k = config.get('top_k', 10)
    top_k = df.nlargest(k, 'score')
    top_k.to_csv(scored_data.path, index=False)
    
    # Save metrics
    with open(ml_metrics.path, 'w') as f:
        json.dump({
            'accuracy': float(accuracy),
            'mse': float(mean_squared_error(y_test, y_pred)),
            'top_k': k
        }, f)
    
    print(f"✅ ML scoring completed: Top-{k} items selected")
    print(f"📈 Model accuracy: {accuracy:.3f}")
    
    MLOutput = namedtuple('MLOutput', ['model_accuracy', 'top_k_count'])
    return MLOutput(float(accuracy), k)

@pipeline(name="base64-ecommerce-pipeline")
def base64_ecommerce_pipeline():
    """E-commerce pipeline with base64 encoded parameters"""
    # Encode configs
    ml_config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "top_k": 10,
        "weights": {
            "price": 0.3,
            "availability": 0.25,
            "stock": 0.2
        }
    }
    
    # Convert to base64
    ml_config_b64 = encode_config(ml_config)
    
    # Run ML scoring with base64 encoded config
    ml_scoring_component(
        stored_data="dummy_data",
        stored_products=100,
        ml_config_b64=ml_config_b64
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=base64_ecommerce_pipeline,
        package_path="base64_ecommerce_pipeline.yaml"
    )
    print("Pipeline compiled to base64_ecommerce_pipeline.yaml")