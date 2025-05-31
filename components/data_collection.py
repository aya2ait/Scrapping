# For KFP v1.x compatibility
try:
    # Try KFP v2.x imports first
    from kfp.dsl import component, Output, Dataset
except ImportError:
    # Fall back to KFP v1.x imports
    from kfp.components import create_component_from_func
    from typing import NamedTuple
    
    # Define a function that will be converted to a component
    def data_collection_func(
        stores_config: str,
        scraping_config: str
    ) -> NamedTuple('Outputs', [('output_data', str)]):
        """
        Component that scrapes product data from eCommerce stores.
        
        Args:
            stores_config: JSON string with store configurations
            scraping_config: JSON string with scraping parameters
        Returns:
            output_data: Path to the extracted products
        """
        import json
        import pandas as pd
        import sys
        import logging
        import os
        from collections import namedtuple
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('data-collection')
        
        try:
            # Parse configurations
            stores = json.loads(stores_config)
            config = json.loads(scraping_config)
            
            logger.info(f"Starting data collection for {len(stores)} stores")
            
            # Import the scraping logic from pipeline.py
            # This would normally be a proper import, but for this example:
            from pipeline import DataExtractor
            
            # Initialize extractor
            extractor = DataExtractor(stores, config)
            
            # Extract data
            products = extractor.extract_all_stores()
            
            # Save to output
            output_path = '/tmp/extracted_products.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df = pd.DataFrame(products)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Extracted {len(products)} products and saved to {output_path}")
            
            Outputs = namedtuple('Outputs', ['output_data'])
            return Outputs(output_path)
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            raise
    
    # Create the component
    data_collection = create_component_from_func(
        func=data_collection_func,
        base_image="python:3.9-slim",
        packages_to_install=["pandas", "selenium", "requests"],
    )
