"""
Simple ML Pipeline Orchestration Script - LINUX COMPATIBLE VERSION

This script orchestrates the execution of three main components:
1. Data extraction (pipeline.py) - Scrapes product data from e-commerce stores
2. Data storage (mongo.py) - Stores the extracted data in MongoDB
3. API server (topk_analyzer.py) - Starts the analysis API server

Usage:
    python orchestration.py [--skip-extraction] [--skip-storage] [--skip-api]
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available, continuing without ML tracking")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Orchestration")

# Define paths to scripts
EXTRACTION_SCRIPT = "pipeline.py"
STORAGE_SCRIPT = "mongo.py"
API_SCRIPT = "topk_analyzer.py"

# Define output paths - LINUX COMPATIBLE
OUTPUT_DIR = "data"
EXTRACTION_OUTPUT = f"{OUTPUT_DIR}/unified_extracted_products.csv"  # ← FIXED: No os.path.join

def setup_environment():
    """Set up the environment for the pipeline run"""
    logger.info("Setting up environment...")
    
    # Create output directory if it doesn't exist - LINUX COMPATIBLE
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")
    
    # Set up MLflow (optional)
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("e-commerce-ml-pipeline")
            logger.info("MLflow experiment set: e-commerce-ml-pipeline")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    else:
        logger.info("MLflow not available, skipping ML tracking setup")

def run_extraction():
    """Run the data extraction script"""
    logger.info("Starting data extraction...")
    
    try:
        # Check if extraction script exists
        if not os.path.exists(EXTRACTION_SCRIPT):
            logger.error(f"Extraction script not found: {EXTRACTION_SCRIPT}")
            return False
        
        # MLflow tracking (optional)
        mlflow_context = None
        if MLFLOW_AVAILABLE:
            try:
                mlflow_context = mlflow.start_run(run_name="data_extraction", nested=True)
                mlflow_context.__enter__()
            except Exception as e:
                logger.warning(f"MLflow run creation failed: {e}")
                mlflow_context = None

        start_time = time.time()
        
        # Run the extraction script - LINUX COMPATIBLE
        result = subprocess.run(
            [sys.executable, EXTRACTION_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),  # ← ADDED: Explicit working directory
            env=os.environ.copy()  # ← ADDED: Copy current environment
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log metrics and artifacts (optional MLflow)
        if mlflow_context and MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric("duration_seconds", duration)
                mlflow.log_param("script_path", EXTRACTION_SCRIPT)
                
                if os.path.exists(EXTRACTION_OUTPUT):
                    mlflow.log_artifact(EXTRACTION_OUTPUT)
                    file_size = os.path.getsize(EXTRACTION_OUTPUT)
                    mlflow.log_metric("output_file_size_bytes", file_size)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
        # Close MLflow run
        if mlflow_context:
            try:
                mlflow_context.__exit__(None, None, None)
            except:
                pass
        
        if os.path.exists(EXTRACTION_OUTPUT):
            logger.info(f"Extraction completed successfully in {duration:.2f} seconds")
            return True
        else:
            logger.error(f"Extraction output file not found: {EXTRACTION_OUTPUT}")
            return False
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False

def run_storage():
    """Run the MongoDB storage script"""
    logger.info("Starting data storage in MongoDB...")
    
    try:
        # Check if storage script exists
        if not os.path.exists(STORAGE_SCRIPT):
            logger.error(f"Storage script not found: {STORAGE_SCRIPT}")
            return False
        
        # MLflow tracking (optional)
        mlflow_context = None
        if MLFLOW_AVAILABLE:
            try:
                mlflow_context = mlflow.start_run(run_name="data_storage", nested=True)
                mlflow_context.__enter__()
            except Exception as e:
                logger.warning(f"MLflow run creation failed: {e}")
                mlflow_context = None

        start_time = time.time()
        
        # Run the storage script - LINUX COMPATIBLE
        result = subprocess.run(
            [sys.executable, STORAGE_SCRIPT, EXTRACTION_OUTPUT],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),  # ← ADDED: Explicit working directory
            env=os.environ.copy()  # ← ADDED: Copy current environment
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log metrics (optional MLflow)
        if mlflow_context and MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric("duration_seconds", duration)
                mlflow.log_param("script_path", STORAGE_SCRIPT)
                mlflow.log_param("input_file", EXTRACTION_OUTPUT)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
        # Close MLflow run
        if mlflow_context:
            try:
                mlflow_context.__exit__(None, None, None)
            except:
                pass
        
        logger.info(f"Storage completed successfully in {duration:.2f} seconds")
        return True
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Storage failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during storage: {e}")
        return False

def run_api_server():
    """Run the API server for product analysis"""
    logger.info("Starting API server for product analysis...")
    
    try:
        # Check if API script exists
        if not os.path.exists(API_SCRIPT):
            logger.error(f"API script not found: {API_SCRIPT}")
            return False, None
        
        # MLflow tracking (optional)
        mlflow_context = None
        if MLFLOW_AVAILABLE:
            try:
                mlflow_context = mlflow.start_run(run_name="api_server", nested=True)
                mlflow_context.__enter__()
                mlflow.log_param("script_path", API_SCRIPT)
            except Exception as e:
                logger.warning(f"MLflow run creation failed: {e}")
                mlflow_context = None

        # Start the API server process - LINUX COMPATIBLE
        process = subprocess.Popen(
            [sys.executable, API_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),  # ← ADDED: Explicit working directory
            env=os.environ.copy()  # ← ADDED: Copy current environment
        )
        
        # Wait a bit to see if it starts successfully
        time.sleep(5)
        
        if process.poll() is None:
            # Process is still running, which is good
            logger.info(f"API server started successfully (PID: {process.pid})")
            
            if mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("api_server_pid", process.pid)
                except:
                    pass
            
            # Close MLflow run
            if mlflow_context:
                try:
                    mlflow_context.__exit__(None, None, None)
                except:
                    pass
            
            return True, process
        else:
            # Process exited early
            stdout, stderr = process.communicate()
            logger.error(f"API server failed to start: {stderr}")
            
            if mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("error", "API server exited prematurely")
                except:
                    pass
            
            # Close MLflow run
            if mlflow_context:
                try:
                    mlflow_context.__exit__(None, None, None)
                except:
                    pass
            
            return False, None
                
    except Exception as e:
        logger.error(f"Unexpected error starting API server: {e}")
        return False, None

def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(description="E-commerce ML Pipeline Orchestration")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip the data extraction step")
    parser.add_argument("--skip-storage", action="store_true", help="Skip the MongoDB storage step")
    parser.add_argument("--skip-api", action="store_true", help="Skip starting the API server")
    args = parser.parse_args()
    
    # Setup environment and MLflow
    setup_environment()
    
    # Start the main MLflow run (optional)
    main_mlflow_context = None
    if MLFLOW_AVAILABLE:
        try:
            main_mlflow_context = mlflow.start_run(run_name=f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            main_mlflow_context.__enter__()
        except Exception as e:
            logger.warning(f"Main MLflow run creation failed: {e}")
            main_mlflow_context = None
    
    pipeline_success = True
    
    try:
        # Step 1: Data Extraction
        if not args.skip_extraction:
            extraction_success = run_extraction()
            
            if main_mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("extraction_success", extraction_success)
                except:
                    pass
            
            pipeline_success = pipeline_success and extraction_success
            
            if not extraction_success:
                logger.error("Extraction failed, stopping pipeline")
                return False
        else:
            logger.info("Skipping extraction step")
            if main_mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("extraction_skipped", True)
                except:
                    pass
        
        # Step 2: Data Storage
        if not args.skip_storage:
            if not os.path.exists(EXTRACTION_OUTPUT) and not args.skip_extraction:
                logger.error(f"Extraction output file not found: {EXTRACTION_OUTPUT}")
                
                if main_mlflow_context and MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_param("storage_success", False)
                    except:
                        pass
                
                pipeline_success = False
                return False
            
            storage_success = run_storage()
            
            if main_mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("storage_success", storage_success)
                except:
                    pass
            
            pipeline_success = pipeline_success and storage_success
            
            if not storage_success:
                logger.error("Storage failed, stopping pipeline")
                return False
        else:
            logger.info("Skipping storage step")
            if main_mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("storage_skipped", True)
                except:
                    pass
        
        # Step 3: API Server
        if not args.skip_api:
            api_success, api_process = run_api_server()
            
            if main_mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("api_server_success", api_success)
                except:
                    pass
            
            pipeline_success = pipeline_success and api_success
            
            if api_success:
                logger.info("API server is running. Press Ctrl+C to stop.")
                try:
                    # Keep the script running while the API server is active
                    while api_process.poll() is None:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping API server...")
                    api_process.terminate()
                    api_process.wait()
                    logger.info("API server stopped")
        else:
            logger.info("Skipping API server step")
            if main_mlflow_context and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("api_server_skipped", True)
                except:
                    pass
        
        # Log overall success
        if main_mlflow_context and MLFLOW_AVAILABLE:
            try:
                mlflow.log_param("pipeline_success", pipeline_success)
            except:
                pass
        
        if pipeline_success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline completed with errors")
        
        return pipeline_success
        
    finally:
        # Close main MLflow run
        if main_mlflow_context:
            try:
                main_mlflow_context.__exit__(None, None, None)
            except:
                pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)