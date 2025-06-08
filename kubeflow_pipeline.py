"""
Kubeflow Pipeline for E-commerce ML Pipeline
Simple beginner-friendly pipeline definition
"""
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

def scraping_component():
    """Component for data scraping"""
    def scrape_data():
        import subprocess
        import sys
        
        # Run the scraping script
        result = subprocess.run([sys.executable, "pipeline.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Scraping completed successfully")
            return "data/unified_extracted_products.csv"
        else:
            print(f"❌ Scraping failed: {result.stderr}")
            raise Exception("Scraping failed")
    
    return scrape_data

def storage_component():
    """Component for MongoDB storage"""
    def store_data(input_file: str):
        import subprocess
        import sys
        
        # Run the storage script
        result = subprocess.run([sys.executable, "mongo.py", input_file], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Storage completed successfully")
            return "stored"
        else:
            print(f"❌ Storage failed: {result.stderr}")
            raise Exception("Storage failed")
    
    return store_data

def analysis_component():
    """Component for data analysis"""
    def analyze_data():
        import subprocess
        import sys
        import time
        
        # Start the analysis API
        process = subprocess.Popen([sys.executable, "topk_analyzer.py"])
        time.sleep(5)  # Give it time to start
        
        if process.poll() is None:
            print("✅ Analysis API started successfully")
            return "analysis_complete"
        else:
            print("❌ Analysis API failed to start")
            raise Exception("Analysis failed")
    
    return analyze_data

# Create Kubeflow components
scraping_op = create_component_from_func(
    scraping_component(),
    base_image='python:3.9'
)

storage_op = create_component_from_func(
    storage_component(),
    base_image='python:3.9'
)

analysis_op = create_component_from_func(
    analysis_component(),
    base_image='python:3.9'
)

@dsl.pipeline(
    name='E-commerce ML Pipeline',
    description='Complete pipeline for e-commerce data processing'
)
def ecommerce_pipeline():
    """Define the pipeline workflow"""
    
    # Step 1: Scraping
    scraping_task = scraping_op()
    
    # Step 2: Storage (depends on scraping)
    storage_task = storage_op(scraping_task.output)
    
    # Step 3: Analysis (depends on storage)
    analysis_task = analysis_op()
    analysis_task.after(storage_task)

if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(ecommerce_pipeline, 'ecommerce_pipeline.yaml')
    print("✅ Pipeline compiled to ecommerce_pipeline.yaml")