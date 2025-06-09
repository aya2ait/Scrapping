# main.py - Version mise à jour avec API

import uvicorn
from fastapi import FastAPI
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Maintenant tu peux importer ton code
from app.pipeline import (
    ScrapingConfig, StoreConfig, UnifiedExtractionPipeline,
    DataValidator, ExportManager
)

# Import des routes API
from api_routes import app as api_app

def run_cli_extraction():
    """Mode ligne de commande original"""
    
    # Configuration des stores (votre configuration existante)
    STORES = [
        StoreConfig(
            domain="allbirds.com",
            name="Allbirds",
            platform="shopify",
            region="US",
            currency="USD",
            priority=1
        ),
        StoreConfig(
            domain="gymshark.com",
            name="Gymshark",
            platform="shopify",
            region="UK",
            currency="GBP",
            priority=2
        ),
        StoreConfig(
            domain="barefootbuttons.com",
            name="Studio McGee",
            platform="woocommerce",
            region="US",
            currency="USD",
            priority=3
        ),
    ]
    
    # Configuration du scraping
    scraping_config = ScrapingConfig(
        delay_between_requests=2.0,
        delay_between_domains=3.0,
        max_products_per_store=5000,
        use_selenium=True,
        headless=True,
        timeout=45
    )
    
    try:
        print("🚀 Starting Unified A2A Extraction Pipeline...")
        print("=" * 60)
        
        # Exécution du pipeline unifié
        pipeline = UnifiedExtractionPipeline(scraping_config)
        extracted_products = pipeline.extract_all_stores(
            stores=STORES,
            output_file="unified_extracted_products.csv"
        )
        
        print("\n" + "=" * 60)
        print("🎉 EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Total products extracted: {len(extracted_products)}")
        print(f"📁 Main results file: 'unified_extracted_products.csv'")
        print("=" * 60)
        
        # Validation des données
        if extracted_products:
            validation_stats = DataValidator.validate_products(extracted_products)
            DataValidator.print_validation_report(validation_stats)
            
            # Exports multiples
            ExportManager.export_to_json(extracted_products, "products.json")
            ExportManager.export_to_excel(extracted_products, "products.xlsx")
        
    except KeyboardInterrupt:
        print("\n⚠️ Extraction interrupted by user")
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Démarre le serveur API"""
    
    print("🚀 Starting E-commerce Scraping API Server...")
    print("=" * 60)
    print(f"🌐 Server will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 ReDoc Documentation: http://{host}:{port}/redoc")
    print("=" * 60)
    print("\n📋 Available API Endpoints:")
    print("  • GET  /                    - API Information")
    print("  • GET  /health              - Health Check")
    print("  • POST /extract/all         - Extract All Platforms")
    print("  • POST /extract/shopify     - Extract Shopify Only")
    print("  • POST /extract/woocommerce - Extract WooCommerce Only")
    print("  • POST /extract/single      - Extract Single Store")
    print("  • GET  /tasks               - List All Tasks")
    print("  • GET  /tasks/{task_id}     - Get Task Status")
    print("  • GET  /tasks/{task_id}/download - Download Results")
    print("  • DELETE /tasks/{task_id}   - Delete Task")
    print("=" * 60)
    
    uvicorn.run(api_app, host=host, port=port, reload=False)

def main():
    """Point d'entrée principal avec choix du mode"""
    
    parser = argparse.ArgumentParser(
        description="E-commerce Unified Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli                    # Run CLI extraction
  python main.py api                    # Start API server  
  python main.py api --port 8080        # Start API on port 8080
  python main.py api --host 127.0.0.1   # Start API on localhost only
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['cli', 'api'],
        help='Execution mode: cli for command line, api for REST API server'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development (API mode only)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        print("🔧 Running in CLI mode...")
        run_cli_extraction()
        
    elif args.mode == 'api':
        print("🌐 Running in API mode...")
        
        if args.reload:
            print("⚠️ Auto-reload enabled (development mode)")
            uvicorn.run(
                "api_routes:app", 
                host=args.host, 
                port=args.port, 
                reload=True
            )
        else:
            run_api_server(args.host, args.port)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()