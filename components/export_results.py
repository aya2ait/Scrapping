# For KFP v1.x compatibility
try:
    # Try KFP v2.x imports first
    from kfp.dsl import component, Input, Output, Dataset, Artifact
except ImportError:
    # Fall back to KFP v1.x imports
    from kfp.components import create_component_from_func
    from typing import NamedTuple
    
    # Define a function that will be converted to a component
    def export_results_func(
        input_data_path: str,
        export_config: str
    ) -> NamedTuple('Outputs', [('json_output_path', str), ('csv_output_path', str)]):
        """
        Component that exports the top products to various formats.
        
        Args:
            input_data_path: Path to the top products data
            export_config: JSON string with export parameters
        Returns:
            json_output_path: Path to the JSON output
            csv_output_path: Path to the CSV output
        """
        import json
        import pandas as pd
        import logging
        import os
        from collections import namedtuple
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('export-results')
        
        try:
            # Parse configuration
            config = json.loads(export_config)
            
            # Load data
            df = pd.read_csv(input_data_path)
            
            # Apply any transformations specified in config
            if config.get('include_columns'):
                df = df[config['include_columns']]
            
            # Create output directory
            os.makedirs('/tmp/outputs', exist_ok=True)
            
            # Export to JSON
            json_output_path = '/tmp/outputs/top_products.json'
            df.to_json(json_output_path, orient='records', indent=2)
            logger.info(f"Exported JSON results to {json_output_path}")
            
            # Export to CSV
            csv_output_path = '/tmp/outputs/top_products.csv'
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Exported CSV results to {csv_output_path}")
            
            Outputs = namedtuple('Outputs', ['json_output_path', 'csv_output_path'])
            return Outputs(json_output_path, csv_output_path)
            
        except Exception as e:
            logger.error(f"Error in export: {str(e)}")
            raise
    
    # Create the component
    export_results = create_component_from_func(
        func=export_results_func,
        base_image="python:3.9-slim",
        packages_to_install=["pandas"],
    )