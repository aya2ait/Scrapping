# For KFP v1.x compatibility
try:
    # Try KFP v2.x imports first
    from kfp.dsl import component, Output, Dataset, Metrics
except ImportError:
    # Fall back to KFP v1.x imports
    from kfp.components import create_component_from_func
    from typing import NamedTuple
    
    # Define a function that will be converted to a component
    def analysis_func(
        mongodb_config: str,
        analysis_config: str
    ) -> NamedTuple('Outputs', [('output_data', str), ('analyzed_products', int), ('top_k_count', int)]):
        """
        Component that analyzes products and finds the top-K best ones.
        
        Args:
            mongodb_config: JSON string with MongoDB connection details
            analysis_config: JSON string with analysis parameters
        Returns:
            output_data: Path to the top products
            analyzed_products: Number of products analyzed
            top_k_count: Number of top products selected
        """
        import json
        import pandas as pd
        import logging
        from pymongo import MongoClient
        import os
        from collections import namedtuple
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('analysis')
        
        try:
            # Parse configurations
            mongo_config = json.loads(mongodb_config)
            config = json.loads(analysis_config)
            
            # Connect to MongoDB
            client = MongoClient(mongo_config['connection_string'])
            db = client[mongo_config['database']]
            collection = db[mongo_config['collection']]
            
            # Retrieve products
            products = list(collection.find({}))
            logger.info(f"Retrieved {len(products)} products for analysis")
            
            # Import the analysis logic from topk_analyzer.py
            # This would normally be a proper import, but for this example:
            from topk_analyzer import ProductAnalyzer
            
            # Initialize analyzer
            analyzer = ProductAnalyzer(config)
            
            # Analyze and get top products
            top_products = analyzer.find_top_k(products)
            
            # Save results
            output_path = '/tmp/top_products.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df = pd.DataFrame(top_products)
            df.to_csv(output_path, index=False)
            
            # Log metrics
            logger.info(f"Identified top {len(top_products)} products")
            
            Outputs = namedtuple('Outputs', ['output_data', 'analyzed_products', 'top_k_count'])
            return Outputs(output_path, len(products), len(top_products))
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise
    
    # Create the component
    analysis = create_component_from_func(
        func=analysis_func,
        base_image="python:3.9-slim",
        packages_to_install=["pandas", "pymongo", "scikit-learn"],
    )
