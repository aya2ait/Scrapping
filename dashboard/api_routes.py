from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
from datetime import datetime
import uuid
from fastapi import Query
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
import io
from pymongo.errors import BulkWriteError

# Modèles Pydantic pour la validation
from pydantic import BaseModel


from typing import Union
import pandas as pd
import sys

# Ajoute le dossier parent dans les chemins de recherche
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Import de vos classes existantes
from pipeline import (
    ScrapingConfig, StoreConfig, ProductData, 
    UnifiedExtractionPipeline, AgentFactory,
    DataValidator, ExportManager, ConfigManager
)
from paste import ProductAnalyzer  # ou le nom réel de votre fichier


app = FastAPI(title="E-commerce Scraping API", version="1.0.0")
# === NOUVEAUX MODELS PYDANTIC ===

class DatabaseConfig(BaseModel):
    connection_string: str
    database_name: str
    collection_name: str

class ProductSearchRequest(BaseModel):
    search_term: str
    limit: int = 100

class PriceRangeRequest(BaseModel):
    min_price: float
    max_price: float

class StoreProductsRequest(BaseModel):
    store_domain: str

class ExportRequest(BaseModel):
    output_filename: Optional[str] = None
    limit: Optional[int] = None

# === ROUTES MONGODB ===

@app.post("/mongodb/health")
async def mongodb_health_check(db_config: DatabaseConfig):
    """Vérifie la connexion MongoDB"""
    try:
        db = ProductAnalyzer(
            mongo_uri=db_config.connection_string,
            db_name=db_config.database_name,
            collection_name=db_config.collection_name
        )

        if db.connect():
            # Correction: utiliser la méthode correcte selon votre classe ProductAnalyzer
            
            return {
                "status": "connected",
                "database": db_config.database_name,
                "collection": db_config.collection_name,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "disconnected",
                "error": "Failed to connect to MongoDB"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB health check failed: {str(e)}")


@app.post("/mongodb/import-csv")
async def import_csv_to_mongodb(
    file: UploadFile = File(...),
    connection_string: str = Form(...),
    database_name: str = Form(...),
    collection_name: str = Form(...),
    clear_collection: bool = Form(False),
    batch_size: int = Form(1000)
):
    """
    Importe un fichier CSV vers MongoDB
    
    Args:
        file: Fichier CSV à importer
        connection_string: URI de connexion MongoDB
        database_name: Nom de la base de données
        collection_name: Nom de la collection
        clear_collection: Vider la collection avant l'import (défaut: False)
        batch_size: Taille du lot pour l'insertion (défaut: 1000)
    """
    
    # Validation du fichier
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
    
    try:
        # Lire le contenu du fichier CSV
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        
        # Créer un DataFrame à partir du CSV
        df = pd.read_csv(io.StringIO(csv_data))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier CSV est vide")
        
        # Initialiser l'analyseur de produits
        db_config = DatabaseConfig(
            connection_string=connection_string,
            database_name=database_name,
            collection_name=collection_name
        )
        
        analyzer = ProductAnalyzer(
            mongo_uri=db_config.connection_string,
            db_name=db_config.database_name,
            collection_name=db_config.collection_name
        )
        
        # Vérifier la connexion
        if not analyzer.connect():
            raise HTTPException(status_code=500, detail="Impossible de se connecter à MongoDB")
        
        # Nettoyer la collection si demandé
        if clear_collection:
            analyzer.collection.delete_many({})
        
        # Préparer les données pour l'insertion
        records = []
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Convertir la ligne en dictionnaire et nettoyer les valeurs NaN
                record = row.to_dict()
                
                # Remplacer les valeurs NaN par None
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (int, float)) and pd.isna(value):
                        record[key] = None
                
                # Ajouter des métadonnées d'import
                record['_import_timestamp'] = datetime.utcnow()
                record['_import_source'] = file.filename
                record['_import_row'] = index + 1
                
                # Traitement spécifique selon les colonnes détectées
                if 'price' in record and record['price'] is not None:
                    try:
                        record['price'] = float(record['price'])
                    except (ValueError, TypeError):
                        record['price'] = 0.0
                
                if 'compare_at_price' in record and record['compare_at_price'] is not None:
                    try:
                        record['compare_at_price'] = float(record['compare_at_price'])
                    except (ValueError, TypeError):
                        record['compare_at_price'] = None
                
                if 'stock_quantity' in record and record['stock_quantity'] is not None:
                    try:
                        record['stock_quantity'] = int(record['stock_quantity'])
                    except (ValueError, TypeError):
                        record['stock_quantity'] = 0
                
                if 'available' in record and record['available'] is not None:
                    # Convertir en booléen
                    if isinstance(record['available'], str):
                        record['available'] = record['available'].lower() in ['true', '1', 'yes', 'oui']
                    else:
                        record['available'] = bool(record['available'])
                
                # Traiter les tags s'ils sont sous forme de chaîne
                if 'tags' in record and record['tags'] is not None:
                    if isinstance(record['tags'], str):
                        # Assumer que les tags sont séparés par des virgules
                        record['tags'] = [tag.strip() for tag in record['tags'].split(',') if tag.strip()]
                
                # Traiter les dates
                date_fields = ['created_at', 'updated_at', 'published_at']
                for date_field in date_fields:
                    if date_field in record and record[date_field] is not None:
                        try:
                            record[date_field] = pd.to_datetime(record[date_field]).to_pydatetime()
                        except:
                            record[date_field] = None
                
                records.append(record)
                
            except Exception as e:
                errors.append(f"Erreur ligne {index + 1}: {str(e)}")
        
        # Insertion par lots
        inserted_count = 0
        failed_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                result = analyzer.collection.insert_many(batch, ordered=False)
                inserted_count += len(result.inserted_ids)
            except BulkWriteError as bwe:
                # Compter les insertions réussies même en cas d'erreur partielle
                inserted_count += bwe.details.get('nInserted', 0)
                failed_count += len(batch) - bwe.details.get('nInserted', 0)
                
                # Ajouter les erreurs détaillées
                for error in bwe.details.get('writeErrors', []):
                    errors.append(f"Erreur insertion: {error.get('errmsg', 'Erreur inconnue')}")
            except Exception as e:
                failed_count += len(batch)
                errors.append(f"Erreur lot {i//batch_size + 1}: {str(e)}")
        
        # Préparer les données d'exemple pour la réponse
        sample_data = []
        if inserted_count > 0:
            # Récupérer quelques enregistrements récemment insérés
            sample_cursor = analyzer.collection.find({
                '_import_source': file.filename
            }).limit(3)
            
            for doc in sample_cursor:
                # Convertir ObjectId en string pour la sérialisation JSON
                doc['_id'] = str(doc['_id'])
                # Convertir les dates en string
                for key, value in doc.items():
                    if isinstance(value, datetime):
                        doc[key] = value.isoformat()
                sample_data.append(doc)
        
        # Préparer la réponse
        response = CSVImportResponse(
            status="success" if failed_count == 0 else "partial_success" if inserted_count > 0 else "failed",
            total_records=len(df),
            inserted_records=inserted_count,
            failed_records=failed_count,
            errors=errors[:10],  # Limiter à 10 erreurs pour éviter une réponse trop longue
            timestamp=datetime.utcnow().isoformat(),
            sample_data=sample_data
        )
        
        return response
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide ou mal formaté")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'import: {str(e)}")

@app.post("/mongodb/validate-csv")
async def validate_csv_structure(file: UploadFile = File(...)):
    """
    Valide la structure d'un fichier CSV avant l'import
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
    
    try:
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        
        # Lire seulement les premières lignes pour la validation
        df = pd.read_csv(io.StringIO(csv_data), nrows=5)
        
        # Analyser la structure
        validation_info = {
            "filename": file.filename,
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "sample_rows": len(df),
            "column_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records'),
            "recommendations": []
        }
        
        # Ajouter des recommandations basées sur les colonnes détectées
        expected_columns = ['title', 'vendor', 'price', 'available', 'stock_quantity', 'store_domain', 'platform']
        missing_important = [col for col in expected_columns if col not in df.columns]
        
        if missing_important:
            validation_info["recommendations"].append(
                f"Colonnes importantes manquantes: {', '.join(missing_important)}"
            )
        
        if 'price' in df.columns:
            non_numeric_prices = df['price'].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)) if pd.notna(x) else False).sum()
            if non_numeric_prices > 0:
                validation_info["recommendations"].append("Certains prix ne sont pas numériques")
        
        return validation_info
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de validation: {str(e)}")
# === MODELS PYDANTIC POUR L'API ===

class ScrapingConfigAPI(BaseModel):
    max_retries: int = 3
    timeout: int = 30
    delay_between_requests: float = 1.5
    delay_between_domains: float = 2.0
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    max_products_per_store: int = 10000
    use_selenium: bool = False
    headless: bool = True
    chrome_driver_path: Optional[str] = None

class StoreConfigAPI(BaseModel):
    domain: str
    name: str
    platform: str  # 'shopify', 'woocommerce', 'generic'
    region: str = "Unknown"
    currency: str = "USD"
    priority: int = 1
    custom_headers: Optional[Dict[str, str]] = None
    api_credentials: Optional[Dict[str, str]] = None
    custom_selectors: Optional[Dict[str, str]] = None

class ExtractionRequest(BaseModel):
    stores: List[StoreConfigAPI]
    scraping_config: Optional[ScrapingConfigAPI] = None
    output_format: str = "csv"  # csv, json, excel
    output_filename: Optional[str] = None

class SingleStoreRequest(BaseModel):
    store: StoreConfigAPI
    scraping_config: Optional[ScrapingConfigAPI] = None
    output_format: str = "csv"

class DatabaseConfig(BaseModel):
    connection_string: str
    database_name: str
    collection_name: str

class CSVImportResponse(BaseModel):
    status: str
    total_records: int
    inserted_records: int
    failed_records: int
    errors: List[str]
    timestamp: str
    sample_data: List[Dict]
# === STOCKAGE DES TÂCHES ===
extraction_tasks = {}

# === UTILITAIRES ===

def convert_api_models_to_dataclasses(stores_api: List[StoreConfigAPI], 
                                    config_api: Optional[ScrapingConfigAPI] = None):
    """Convertit les modèles Pydantic en dataclasses"""
    
    # Configuration par défaut
    if config_api is None:
        config_api = ScrapingConfigAPI()
    
    scraping_config = ScrapingConfig(
        max_retries=config_api.max_retries,
        timeout=config_api.timeout,
        delay_between_requests=config_api.delay_between_requests,
        delay_between_domains=config_api.delay_between_domains,
        user_agent=config_api.user_agent,
        max_products_per_store=config_api.max_products_per_store,
        use_selenium=config_api.use_selenium,
        headless=config_api.headless,
        chrome_driver_path=config_api.chrome_driver_path
    )
    
    stores = []
    for store_api in stores_api:
        store = StoreConfig(
            domain=store_api.domain,
            name=store_api.name,
            platform=store_api.platform,
            region=store_api.region,
            currency=store_api.currency,
            priority=store_api.priority,
            custom_headers=store_api.custom_headers,
            api_credentials=store_api.api_credentials,
            custom_selectors=store_api.custom_selectors
        )
        stores.append(store)
    
    return stores, scraping_config

async def run_extraction_task(task_id: str, stores: List[StoreConfig], 
                            scraping_config: ScrapingConfig, output_filename: str):
    """Exécute une tâche d'extraction en arrière-plan"""
    try:
        extraction_tasks[task_id]["status"] = "running"
        extraction_tasks[task_id]["started_at"] = datetime.now().isoformat()
        
        # Exécution du pipeline
        pipeline = UnifiedExtractionPipeline(scraping_config)
        products = pipeline.extract_all_stores(stores, output_filename)
        
        # Validation des données
        validation_stats = DataValidator.validate_products(products)
        
        # Mise à jour du statut
        extraction_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "total_products": len(products),
            "validation_stats": validation_stats,
            "output_file": output_filename
        })
        
    except Exception as e:
        extraction_tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })

# === ROUTES API ===

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "E-commerce Scraping API",
        "version": "1.0.0",
        "endpoints": {
            "extract_all": "/extract/all",
            "extract_shopify": "/extract/shopify",
            "extract_woocommerce": "/extract/woocommerce",
            "extract_single": "/extract/single",
            "tasks": "/tasks",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len([t for t in extraction_tasks.values() if t["status"] == "running"])
    }

@app.post("/extract/all")
async def extract_all_stores(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Lance l'extraction pour tous les stores (toutes plateformes)"""
    
    try:
        # Génération d'un ID de tâche unique
        task_id = str(uuid.uuid4())
        
        # Conversion des modèles
        stores, scraping_config = convert_api_models_to_dataclasses(
            request.stores, request.scraping_config
        )
        
        # Nom de fichier
        output_filename = request.output_filename or f"extraction_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialisation de la tâche
        extraction_tasks[task_id] = {
            "task_id": task_id,
            "type": "all_platforms",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "stores_count": len(stores),
            "platforms": list(set(store.platform for store in stores))
        }
        
        # Lancement en arrière-plan
        background_tasks.add_task(
            run_extraction_task, task_id, stores, scraping_config, output_filename
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Extraction started for {len(stores)} stores",
            "check_status_url": f"/tasks/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start extraction: {str(e)}")

@app.post("/extract/shopify")
async def extract_shopify_stores(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Lance l'extraction uniquement pour les stores Shopify"""
    
    try:
        # Filtrer seulement les stores Shopify
        shopify_stores = [store for store in request.stores if store.platform.lower() == "shopify"]
        
        if not shopify_stores:
            raise HTTPException(status_code=400, detail="No Shopify stores found in request")
        
        task_id = str(uuid.uuid4())
        
        stores, scraping_config = convert_api_models_to_dataclasses(
            shopify_stores, request.scraping_config
        )
        
        output_filename = request.output_filename or f"shopify_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        extraction_tasks[task_id] = {
            "task_id": task_id,
            "type": "shopify_only",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "stores_count": len(stores),
            "platforms": ["shopify"]
        }
        
        background_tasks.add_task(
            run_extraction_task, task_id, stores, scraping_config, output_filename
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Shopify extraction started for {len(stores)} stores",
            "check_status_url": f"/tasks/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Shopify extraction: {str(e)}")

@app.post("/extract/woocommerce")
async def extract_woocommerce_stores(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Lance l'extraction uniquement pour les stores WooCommerce"""
    
    try:
        # Filtrer seulement les stores WooCommerce
        woo_stores = [store for store in request.stores if store.platform.lower() == "woocommerce"]
        
        if not woo_stores:
            raise HTTPException(status_code=400, detail="No WooCommerce stores found in request")
        
        task_id = str(uuid.uuid4())
        
        stores, scraping_config = convert_api_models_to_dataclasses(
            woo_stores, request.scraping_config
        )
        
        output_filename = request.output_filename or f"woocommerce_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        extraction_tasks[task_id] = {
            "task_id": task_id,
            "type": "woocommerce_only",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "stores_count": len(stores),
            "platforms": ["woocommerce"]
        }
        
        background_tasks.add_task(
            run_extraction_task, task_id, stores, scraping_config, output_filename
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"WooCommerce extraction started for {len(stores)} stores",
            "check_status_url": f"/tasks/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start WooCommerce extraction: {str(e)}")

@app.post("/extract/single")
async def extract_single_store(request: SingleStoreRequest, background_tasks: BackgroundTasks):
    """Lance l'extraction pour un seul store"""
    
    try:
        task_id = str(uuid.uuid4())
        
        stores, scraping_config = convert_api_models_to_dataclasses(
            [request.store], request.scraping_config
        )
        
        output_filename = f"single_{request.store.platform}_{request.store.domain.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        extraction_tasks[task_id] = {
            "task_id": task_id,
            "type": "single_store",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "stores_count": 1,
            "store_domain": request.store.domain,
            "platform": request.store.platform
        }
        
        background_tasks.add_task(
            run_extraction_task, task_id, stores, scraping_config, output_filename
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Single store extraction started for {request.store.domain}",
            "check_status_url": f"/tasks/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start single store extraction: {str(e)}")

@app.get("/tasks")
async def list_tasks():
    """Liste toutes les tâches d'extraction"""
    return {
        "total_tasks": len(extraction_tasks),
        "tasks": list(extraction_tasks.values())
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Récupère le statut d'une tâche spécifique"""
    
    if task_id not in extraction_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return extraction_tasks[task_id]

@app.get("/tasks/{task_id}/download")
async def download_task_result(task_id: str):
    """Télécharge le fichier résultat d'une tâche"""
    
    if task_id not in extraction_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = extraction_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    output_file = task.get("output_file")
    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_file,
        filename=os.path.basename(output_file),
        media_type='application/octet-stream'
    )

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Supprime une tâche de la liste"""
    
    if task_id not in extraction_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Supprimer le fichier de sortie si il existe
    task = extraction_tasks[task_id]
    output_file = task.get("output_file")
    if output_file and os.path.exists(output_file):
        try:
            os.remove(output_file)
        except:
            pass
    
    # Supprimer la tâche
    del extraction_tasks[task_id]
    
    return {"message": "Task deleted successfully"}

# === ROUTES DE CONFIGURATION ===

@app.get("/config/stores/sample")
async def get_sample_store_config():
    """Retourne un exemple de configuration de stores"""
    return {
        "sample_config": {
            "stores": [
                {
                    "domain": "example-shopify.com",
                    "name": "Example Shopify Store",
                    "platform": "shopify",
                    "region": "US", 
                    "currency": "USD",
                    "priority": 1,
                    "custom_headers": {
                        "Accept": "application/json"
                    }
                },
                {
                    "domain": "example-woo.com",
                    "name": "Example WooCommerce Store", 
                    "platform": "woocommerce",
                    "region": "EU",
                    "currency": "EUR",
                    "priority": 2,
                    "api_credentials": {
                        "consumer_key": "ck_your_key",
                        "consumer_secret": "cs_your_secret"
                    }
                }
            ]
        }
    }

@app.get("/config/scraping/sample")
async def get_sample_scraping_config():
    """Retourne un exemple de configuration de scraping"""
    return {
        "sample_config": {
            "max_retries": 3,
            "timeout": 30,
            "delay_between_requests": 1.5,
            "delay_between_domains": 2.0,
            "max_products_per_store": 10000,
            "use_selenium": False,
            "headless": True
        }
    }

# === ROUTES D'EXPORT ===

@app.post("/export/json/{task_id}")
async def export_to_json(task_id: str):
    """Exporte les résultats d'une tâche en JSON"""
    
    if task_id not in extraction_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = extraction_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    try:
        # Charger les données CSV et convertir en JSON
        csv_file = task.get("output_file")
        if not csv_file or not os.path.exists(csv_file):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        json_file = csv_file.replace('.csv', '.json')
        df.to_json(json_file, orient='records', indent=2)
        
        return FileResponse(
            path=json_file,
            filename=os.path.basename(json_file),
            media_type='application/json'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)