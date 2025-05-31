"""
Kubeflow Pipeline pour l'orchestration ML du système E-commerce - VERSION KFP 2.0.0
Compatible avec Kubeflow Pipelines 2.x (API v2)
"""

from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics, Artifact
from typing import NamedTuple, List
import json

# =============================================================================
# COMPOSANTS KUBEFLOW V2.0.0 - NOUVELLE API
# =============================================================================

@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3", "numpy==1.24.3"]
)
def data_extraction_component(
    stores_config: str,
    scraping_config: str,
    output_data: Output[Dataset]
) -> NamedTuple("ExtractionOutputs", [("total_products", int), ("stores_processed", int)]):
    """
    Composant d'extraction de données A2A
    Utilise votre système UnifiedExtractionPipeline existant
    """
    import json
    import pandas as pd
    from collections import namedtuple
    import os
    
    # Simulation de votre pipeline d'extraction existant
    print("🚀 Démarrage de l'extraction A2A...")
    
    # Parsing des configurations
    stores = json.loads(stores_config)
    scraping_params = json.loads(scraping_config)
    
    # Simulation de l'extraction (remplacer par votre code réel)
    extracted_data = []
    total_products = 0
    
    for store in stores:
        print(f"📦 Extraction depuis {store['domain']} ({store['platform']})...")
        
        # Ici vous appelleriez votre agent A2A approprié
        if store['platform'] == 'shopify':
            products_count = 150  # Simulation
        else:
            products_count = 80   # Simulation
            
        total_products += products_count
        
        # Données simulées pour le pipeline
        for i in range(products_count):
            extracted_data.append({
                'store_domain': store['domain'],
                'platform': store['platform'],
                'title': f"Product {i} from {store['domain']}",
                'price': 29.99 + (i % 100),
                'available': i % 4 != 0,
                'stock_quantity': i % 50,
                'vendor': f"Vendor_{i % 10}"
            })
    
    # Sauvegarde des données extraites
    df = pd.DataFrame(extracted_data)
    df.to_csv(output_data.path, index=False)
    
    print(f"✅ Extraction terminée: {total_products} produits de {len(stores)} magasins")
    
    ExtractionOutputs = namedtuple('ExtractionOutputs', ['total_products', 'stores_processed'])
    return ExtractionOutputs(total_products, len(stores))

@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3"]
)
def data_storage_component(
    extraction_data: Input[Dataset],
    mongodb_config: str
) -> NamedTuple("StorageOutputs", [("stored_products", int), ("quality_ratio", float)]):
    """
    Composant de stockage MongoDB
    Utilise votre système ProductsDB existant
    """
    import pandas as pd
    import json
    from collections import namedtuple
    
    print("💾 Démarrage du stockage MongoDB...")
    
    # Chargement des données extraites
    df = pd.read_csv(extraction_data.path)
    config = json.loads(mongodb_config)
    
    print(f"📊 Traitement de {len(df)} produits...")
    
    # Simulation du stockage (remplacer par votre ProductsDB)
    # db = ProductsDB(config['connection_string'])
    # db.insert_products_batch(df.to_dict('records'))
    
    # Nettoyage et validation des données
    df_clean = df.dropna(subset=['title', 'price'])
    df_clean = df_clean[df_clean['price'] > 0]
    
    stored_products = len(df_clean)
    quality_ratio = stored_products / len(df) if len(df) > 0 else 0
    
    print(f"✅ Stockage terminé: {stored_products} produits stockés")
    
    StorageOutputs = namedtuple('StorageOutputs', ['stored_products', 'quality_ratio'])
    return StorageOutputs(stored_products, quality_ratio)

@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3", "numpy==1.24.3", "scikit-learn==1.2.2"]
)
def ml_scoring_component(
    stored_products: int,
    ml_config: str,
    scored_data: Output[Dataset],
    model: Output[Model]
) -> NamedTuple("MLOutputs", [("model_accuracy", float), ("top_k_count", int)]):
    """
    Composant de scoring ML
    Utilise votre ProductAnalyzer existant
    """
    import json
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pickle
    from collections import namedtuple
    
    print("🧠 Démarrage du scoring ML...")
    
    # Configuration ML
    config = json.loads(ml_config)
    
    # Simulation de données depuis MongoDB (remplacer par votre ProductAnalyzer)
    np.random.seed(42)
    n_products = min(int(stored_products), 1000)  # Conversion explicite en int
    
    # Génération de données simulées
    data = {
        'price': np.random.uniform(10, 500, n_products),
        'stock_quantity': np.random.randint(0, 100, n_products),
        'available': np.random.choice([True, False], n_products, p=[0.8, 0.2]),
        'vendor_popularity': np.random.uniform(0, 1, n_products),
        'platform_score': np.random.uniform(0.5, 1, n_products)
    }
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['price_score'] = 1 / (1 + df['price'] / 100)
    df['availability_score'] = df['available'].astype(float)
    df['stock_score'] = np.minimum(df['stock_quantity'] / 50, 1)
    
    # Calcul du score synthétique (cible)
    weights = config.get('weights', {
        'price': 0.3, 'availability': 0.25, 'stock': 0.2, 
        'vendor_popularity': 0.15, 'platform': 0.1
    })
    
    df['synthetic_score'] = (
        weights['price'] * df['price_score'] +
        weights['availability'] * df['availability_score'] +
        weights['stock'] * df['stock_score'] +
        weights['vendor_popularity'] * df['vendor_popularity'] +
        weights['platform'] * df['platform_score']
    )
    
    # Préparation des features pour ML
    X = df[['price', 'stock_quantity', 'vendor_popularity', 'platform_score']].copy()
    X['available'] = df['available'].astype(int)
    y = df['synthetic_score']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    model_type = config.get('model_type', 'random_forest')
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    print(f"🎯 Entraînement du modèle {model_type}...")
    rf_model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = max(0, 1 - mse)
    
    # Prédiction sur l'ensemble complet
    df['ml_score'] = rf_model.predict(X)
    df['final_score'] = (df['synthetic_score'] + df['ml_score']) / 2
    
    # Sélection des Top-K
    k = config.get('top_k', 50)
    top_products = df.nlargest(k, 'final_score')
    
    # Sauvegarde des résultats
    top_products.to_csv(scored_data.path, index=False)
    
    # Sauvegarde du modèle
    with open(model.path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    print(f"✅ Scoring ML terminé: Top-{k} produits sélectionnés")
    print(f"📈 Précision du modèle: {accuracy:.3f}")
    
    MLOutputs = namedtuple('MLOutputs', ['model_accuracy', 'top_k_count'])
    return MLOutputs(float(accuracy), k)

@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3"]
)
def validation_component(
    scored_data: Input[Dataset],
    validation_config: str,
    validation_report: Output[Metrics]
) -> NamedTuple("ValidationOutputs", [("quality_score", float), ("recommendations_count", int)]):
    """
    Composant de validation et contrôle qualité
    """
    import pandas as pd
    import json
    from collections import namedtuple
    
    print("🔍 Démarrage de la validation...")
    
    # Chargement des données scorées
    df = pd.read_csv(scored_data.path)
    config = json.loads(validation_config)
    
    # Métriques de qualité
    quality_checks = {
        'data_completeness': (df.isnull().sum().sum() == 0),
        'price_validity': (df['price'] > 0).all(),
        'score_distribution': df['final_score'].std() > 0.1,
        'top_products_available': (df.head(10)['available'].astype(bool)).mean() > 0.7
    }
    
    quality_score = sum(quality_checks.values()) / len(quality_checks)
    
    # Génération du rapport
    report = {
        'quality_score': quality_score,
        'total_products_analyzed': len(df),
        'top_products_count': config.get('top_k', 50),
        'average_score': float(df['final_score'].mean()),
        'quality_checks': quality_checks,
        'recommendations': []
    }
    
    # Recommandations basées sur l'analyse
    if quality_score < 0.8:
        report['recommendations'].append("Améliorer la qualité des données d'entrée")
    if df['final_score'].std() < 0.1:
        report['recommendations'].append("Diversifier les critères de scoring")
    
    # Sauvegarde du rapport comme métriques
    with open(validation_report.path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Validation terminée - Score qualité: {quality_score:.2f}")
    
    ValidationOutputs = namedtuple('ValidationOutputs', ['quality_score', 'recommendations_count'])
    return ValidationOutputs(quality_score, len(report['recommendations']))

# =============================================================================
# PIPELINE PRINCIPAL - VERSION KFP 2.0.0
# =============================================================================

@pipeline(
    name="ecommerce-ml-pipeline-v2",
    description="Pipeline ML e-commerce - Version KFP 2.0.0"
)
def ecommerce_ml_pipeline_v2(
    stores_config: str = '[]',
    scraping_config: str = '{}',
    mongodb_config: str = '{}',
    ml_config: str = '{}',
    validation_config: str = '{}'
):
    """
    Pipeline principal - Version KFP 2.0.0
    Utilise la nouvelle API Component et Pipeline
    """
    
    # Étape 1: Extraction des données
    extraction_task = data_extraction_component(
        stores_config=stores_config,
        scraping_config=scraping_config
    )
    
    # Étape 2: Stockage MongoDB
    storage_task = data_storage_component(
        extraction_data=extraction_task.outputs["output_data"],
        mongodb_config=mongodb_config
    )
    
    # Étape 3: Scoring ML et sélection Top-K
    ml_task = ml_scoring_component(
        stored_products=storage_task.outputs["stored_products"],
        ml_config=ml_config
    )
    
    # Étape 4: Validation et contrôle qualité
    validation_task = validation_component(
        scored_data=ml_task.outputs["scored_data"],
        validation_config=validation_config
    )

# =============================================================================
# UTILITAIRES DE CONFIGURATION
# =============================================================================

def create_default_configs():
    """
    Crée les configurations par défaut pour le pipeline
    """
    
    stores_config = [
        {
            "domain": "allbirds.com",
            "name": "Allbirds",
            "platform": "shopify",
            "region": "US",
            "currency": "USD",
            "priority": 1
        },
        {
            "domain": "gymshark.com", 
            "name": "Gymshark",
            "platform": "shopify",
            "region": "UK",
            "currency": "GBP",
            "priority": 2
        }
    ]
    
    scraping_config = {
        "max_retries": 3,
        "timeout_seconds": 30,
        "delay_between_requests": 1.5,
        "selenium_options": {
            "headless": True,
            "window_size": "1920,1080"
        },
        "max_products_per_store": 1000
    }
    
    mongodb_config = {
        "connection_string": "mongodb://localhost:27017",
        "database": "ecommerce_products",
        "collection": "products"
    }
    
    ml_config = {
        "model_type": "random_forest",
        "top_k": 50,
        "weights": {
            "price": 0.3,
            "availability": 0.25,
            "stock": 0.2,
            "vendor_popularity": 0.15,
            "platform": 0.1
        }
    }
    
    validation_config = {
        "top_k": 50,
        "quality_threshold": 0.8,
        "enable_recommendations": True
    }
    
    return {
        'stores_config': json.dumps(stores_config),
        'scraping_config': json.dumps(scraping_config),
        'mongodb_config': json.dumps(mongodb_config),
        'ml_config': json.dumps(ml_config),
        'validation_config': json.dumps(validation_config)
    }

# =============================================================================
# COMPILATION ET EXÉCUTION
# =============================================================================

def compile_pipeline():
    """
    Compilation du pipeline pour KFP v2.0.0
    """
    pipeline_package_path = 'ecommerce_ml_pipeline_v2.yaml'
    
    try:
        print(f"📋 Compilation avec KFP 2.0.0...")
        
        # Nouvelle méthode de compilation KFP v2
        compiler.Compiler().compile(
            pipeline_func=ecommerce_ml_pipeline_v2,
            package_path=pipeline_package_path
        )
        
        print(f"✅ Pipeline compilé: {pipeline_package_path}")
        return pipeline_package_path
        
    except Exception as e:
        print(f"❌ Erreur de compilation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compile-only':
        # Compilation uniquement
        result = compile_pipeline()
        if result:
            print(f"\n📁 Fichier généré: {result}")
            print("💡 Vous pouvez maintenant l'uploader via l'interface web Kubeflow")
    else:
        # Compilation par défaut
        compile_pipeline()