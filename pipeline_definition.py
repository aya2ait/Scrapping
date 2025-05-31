# Import necessary libraries
import kfp
import json

# Import components
from components.data_collection import data_collection
from components.data_storage import data_storage
from components.analysis import analysis
from components.export_results import export_results

# Check KFP version
import pkg_resources
kfp_version = pkg_resources.get_distribution("kfp").version
is_v2 = kfp_version.startswith('2.')

if is_v2:
    # KFP v2.x pipeline definition
    from kfp import dsl, compiler
    
    @dsl.pipeline(
        name="ecommerce-intelligence-pipeline",
        description="Pipeline for eCommerce product intelligence"
    )
    def ecommerce_pipeline(
        stores_config: str = '[]',
        scraping_config: str = '{}',
        mongodb_config: str = '{"connection_string": "mongodb://localhost:27017", "database": "ecommerce_products", "collection": "products"}',
        analysis_config: str = '{"k": 10, "criteria": {"weights": {"price": 0.3, "availability": 0.25, "stock": 0.2}}}',
        export_config: str = '{"include_columns": ["store_domain", "title", "price", "score"]}'
    ):
        """
        Main pipeline that orchestrates the eCommerce intelligence workflow.
        
        Args:
            stores_config: JSON string with store configurations
            scraping_config: JSON string with scraping parameters
            mongodb_config: JSON string with MongoDB connection details
            analysis_config: JSON string with analysis parameters
            export_config: JSON string with export parameters
        """
        
        # Data Collection
        collection_task = data_collection(
            stores_config=stores_config,
            scraping_config=scraping_config
        ).set_display_name("Data Collection")
        
        # Add retry and resource limits
        collection_task.set_retry(3)
        collection_task.set_cpu_limit('1')
        collection_task.set_memory_limit('2G')
        
        # Data Storage
        storage_task = data_storage(
            input_data=collection_task.outputs["output_data"],
            mongodb_config=mongodb_config
        ).set_display_name("Data Storage")
        
        storage_task.set_retry(3)
        storage_task.set_cpu_limit('0.5')
        storage_task.set_memory_limit('1G')
        
        # Analysis
        analysis_task = analysis(
            mongodb_config=mongodb_config,
            analysis_config=analysis_config
        ).set_display_name("Product Analysis")
        
        analysis_task.set_retry(2)
        analysis_task.set_cpu_limit('2')
        analysis_task.set_memory_limit('4G')
        
        # Export Results
        export_task = export_results(
            input_data=analysis_task.outputs["output_data"],
            export_config=export_config
        ).set_display_name("Export Results")
        
        export_task.set_cpu_limit('0.5')
        export_task.set_memory_limit('1G')
        
        # Add dependencies
        storage_task.after(collection_task)
        analysis_task.after(storage_task)
        export_task.after(analysis_task)
    
    # Compile the pipeline
    if __name__ == "__main__":
        compiler.Compiler().compile(
            pipeline_func=ecommerce_pipeline,
            package_path="ecommerce_pipeline.yaml"
        )
        print("Pipeline compiled successfully to ecommerce_pipeline.yaml")

else:
    # KFP v1.x pipeline definition
    import kfp.dsl as dsl
    from kfp.compiler import Compiler
    
    @dsl.pipeline(
        name="ecommerce-intelligence-pipeline-v1",
        description="Pipeline for eCommerce product intelligence (KFP v1.x)"
    )
    def ecommerce_pipeline_v1(
        stores_config: str = '[]',
        scraping_config: str = '{}',
        mongodb_config: str = '{"connection_string": "mongodb://localhost:27017", "database": "ecommerce_products", "collection": "products"}',
        analysis_config: str = '{"k": 10, "criteria": {"weights": {"price": 0.3, "availability": 0.25, "stock": 0.2}}}',
        export_config: str = '{"include_columns": ["store_domain", "title", "price", "score"]}'
    ):
        """
        Main pipeline that orchestrates the eCommerce intelligence workflow.
        
        Args:
            stores_config: JSON string with store configurations
            scraping_config: JSON string with scraping parameters
            mongodb_config: JSON string with MongoDB connection details
            analysis_config: JSON string with analysis parameters
            export_config: JSON string with export parameters
        """
        
        # Data Collection
        collection_task = data_collection(
            stores_config=stores_config,
            scraping_config=scraping_config
        ).set_display_name("Data Collection")
        
        # Data Storage
        storage_task = data_storage(
            input_data_path=collection_task.outputs['output_data'],
            mongodb_config=mongodb_config
        ).set_display_name("Data Storage")
        
        # Analysis
        analysis_task = analysis(
            mongodb_config=mongodb_config,
            analysis_config=analysis_config
        ).set_display_name("Product Analysis")
        
        # Export Results
        export_task = export_results(
            input_data_path=analysis_task.outputs['output_data'],
            export_config=export_config
        ).set_display_name("Export Results")
        
        # Add dependencies
        storage_task.after(collection_task)
        analysis_task.after(storage_task)
        export_task.after(analysis_task)
    
    # Compile the pipeline
    if __name__ == "__main__":
        Compiler().compile(
            pipeline_func=ecommerce_pipeline_v1,
            package_path="ecommerce_pipeline_v1.yaml"
        )
        print("Pipeline compiled successfully to ecommerce_pipeline_v1.yaml")
