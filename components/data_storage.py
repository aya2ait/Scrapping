# For KFP v1.x compatibility
try:
    # Try KFP v2.x imports first
    from kfp.dsl import component, Input, Output, Dataset, Metrics
except ImportError:
    # Fall back to KFP v1.x imports
    from kfp.components import create_component_from_func
    from typing import NamedTuple
    
    # Define a function that will be converted to a component
    def data_storage_func(
        input_data_path: str,
        mongodb_config: str
    ) -> NamedTuple('Outputs', [('stored_products', int), ('storage_success_rate', float)]):
        """
        Component that stores product data in MongoDB.
        
        Args:
            input_data_path: Path to the extracted products data
            mongodb_config: JSON string with MongoDB connection details
        Returns:
            stored_products: Number of products stored
            storage_success_rate: Success rate of storage operation
        """
        import json
        import pandas as pd
        import logging
        from pymongo import MongoClient
        from collections import namedtuple
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('data-storage')
        
        try:
            # Parse configuration
            config = json.loads(mongodb_config)
            
            # Load data
            df = pd.read_csv(input_data_path)
            products = df.to_dict('records')
            
            logger.info(f"Loaded {len(products)} products for storage")
            
            # Connect to MongoDB
            client = MongoClient(config['connection_string'])
            db = client[config['database']]
            collection = db[config['collection']]
            
            # Store data with upsert
            stored_count = 0
            for product in products:
                result = collection.update_one(
                    {"store_domain": product["store_domain"], "title": product["title"]},
                    {"$set": product},
                    upsert=True
                )
                if result.modified_count or result.upserted_id:
                    stored_count += 1
            
            # Log results
            logger.info(f"Successfully stored {stored_count} products in MongoDB")
            
            # Calculate metrics
            storage_success_rate = stored_count / len(products) if products else 0
            
            Outputs = namedtuple('Outputs', ['stored_products', 'storage_success_rate'])
            return Outputs(stored_count, storage_success_rate)
            
        except Exception as e:
            logger.error(f"Error in data storage: {str(e)}")
            raise
    
    # Create the component
    data_storage = create_component_from_func(
        func=data_storage_func,
        base_image="python:3.9-slim",
        packages_to_install=["pandas", "pymongo"],
    )
