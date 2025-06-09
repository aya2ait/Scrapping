import requests
import json
import csv
import time
import logging
import os
import pandas as pd
from urllib.parse import urlparse, urlunparse, urljoin
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import base64

# --- Configuration Classes ---
@dataclass
class ScrapingConfig:
    """Configuration pour le scraping"""
    max_retries: int = 3
    timeout: int = 30
    delay_between_requests: float = 1.5
    delay_between_domains: float = 2.0
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    max_products_per_store: int = 10000
    use_selenium: bool = False  # Pour les sites nécessitant JavaScript
    headless: bool = True
    chrome_driver_path: Optional[str] = None
    skip_woocommerce: bool = True  # NEW: Skip WooCommerce for now

@dataclass
class StoreConfig:
    """Configuration d'un store"""
    domain: str
    name: str
    platform: str  # 'shopify', 'woocommerce', 'generic'
    region: str = "Unknown"
    currency: str = "USD"
    priority: int = 1
    custom_headers: Dict[str, str] = None
    api_credentials: Dict[str, str] = None  # Pour WooCommerce API
    custom_selectors: Dict[str, str] = None  # Pour scraping générique

# --- Data Models ---
@dataclass
class ProductData:
    """Modèle de données unifié pour tous les produits"""
    store_domain: str
    store_region: str
    platform: str
    product_id: str
    title: str
    handle: str
    vendor: str
    product_type: str
    created_at: str
    updated_at: str
    published_at: str
    tags: List[str]
    body_html: str
    variant_id: str
    variant_title: str
    sku: str
    price: float
    compare_at_price: Optional[float]
    available: bool
    stock_quantity: Optional[int]
    variant_created_at: str
    variant_updated_at: str
    image_src: Optional[str]
    all_image_srcs: List[str]
    rating: Optional[float] = None
    review_count: Optional[int] = None
    sales_rank: Optional[int] = None
    traffic_estimate: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour CSV"""
        data = asdict(self)
        data['tags'] = ', '.join(self.tags) if isinstance(self.tags, list) else str(self.tags)
        data['all_image_srcs'] = '|'.join(self.all_image_srcs) if isinstance(self.all_image_srcs, list) else str(self.all_image_srcs)
        return data

# --- Agent A2A Base Class ---
class A2AAgent(ABC):
    """Agent A2A de base pour l'extraction de données"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        self.driver = None
        
    def __enter__(self):
        if self.config.use_selenium:
            self._init_selenium()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()
    
    def _init_selenium(self):
        """Initialize Selenium WebDriver"""
        try:
            chrome_options = Options()
            if self.config.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"--user-agent={self.config.user_agent}")
            
            if self.config.chrome_driver_path:
                self.driver = webdriver.Chrome(
                    executable_path=self.config.chrome_driver_path,
                    options=chrome_options
                )
            else:
                self.driver = webdriver.Chrome(options=chrome_options)
                
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
            raise
    
    @abstractmethod
    def extract_data(self, store_config: StoreConfig) -> List[ProductData]:
        """Extrait les données d'un store"""
        pass
    
    def _make_request(self, url: str, headers: Dict = None) -> Optional[Dict]:
        """Effectue une requête HTTP avec retry et gestion d'erreurs"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.config.timeout,
                    headers=headers or {}
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"All attempts failed for {url}")
                    return None
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error for {url}: {e}")
                return None
        return None

# --- Shopify A2A Agent ---
class ShopifyA2AAgent(A2AAgent):
    """Agent A2A spécialisé pour Shopify"""
    
    def extract_data(self, store_config: StoreConfig) -> List[ProductData]:
        """Extrait les données produits d'un store Shopify"""
        self.logger.info(f"Starting Shopify extraction for {store_config.domain}")
        
        url = self._construct_url(store_config.domain)
        if not url:
            return []
            
        all_products = self._fetch_all_products(url, store_config)
        processed_products = self._process_products(all_products, store_config)
        
        self.logger.info(f"Shopify extraction completed for {store_config.domain}: {len(processed_products)} products")
        return processed_products
    
    def _construct_url(self, domain: str) -> Optional[str]:
        """Construit l'URL de l'API Shopify"""
        try:
            if not domain.startswith(('http://', 'https://')):
                domain = f"https://{domain}"
            
            parsed = urlparse(domain)
            return f"{parsed.scheme}://{parsed.netloc}/products.json"
        except Exception as e:
            self.logger.error(f"Error constructing URL for {domain}: {e}")
            return None
    
    def _fetch_all_products(self, base_url: str, store_config: StoreConfig) -> List[Dict]:
        """Récupère tous les produits avec pagination"""
        products = []
        page = 1
        limit = 250
        max_products = self.config.max_products_per_store
        
        while len(products) < max_products:
            url = f"{base_url}?limit={limit}&page={page}"
            self.logger.info(f"Fetching Shopify page {page} from {store_config.domain}")
            
            headers = store_config.custom_headers or {}
            data = self._make_request(url, headers)
            
            if not data or "products" not in data or not data["products"]:
                break
                
            page_products = data["products"]
            products.extend(page_products)
            
            if len(page_products) < limit:
                break
                
            page += 1
            time.sleep(self.config.delay_between_requests)
            
        return products[:max_products]
    
    def _process_products(self, products: List[Dict], store_config: StoreConfig) -> List[ProductData]:
        """Traite les données produits Shopify"""
        processed_products = []
        
        for product in products:
            try:
                images = product.get('images', [])
                first_image = images[0].get('src') if images else None
                all_images = [img.get('src') for img in images if img.get('src')]
                
                variants = product.get('variants', [{}])
                
                for variant in variants:
                    try:
                        price = float(variant.get('price', 0))
                        compare_at_price = float(variant.get('compare_at_price')) if variant.get('compare_at_price') else None
                        
                        tags = product.get('tags', [])
                        if isinstance(tags, str):
                            tags = [tag.strip() for tag in tags.split(',')]
                        
                        product_data = ProductData(
                            store_domain=store_config.domain,
                            store_region=store_config.region,
                            platform="shopify",
                            product_id=str(product.get('id', '')),
                            title=product.get('title', ''),
                            handle=product.get('handle', ''),
                            vendor=product.get('vendor', ''),
                            product_type=product.get('product_type', ''),
                            created_at=product.get('created_at', ''),
                            updated_at=product.get('updated_at', ''),
                            published_at=product.get('published_at', ''),
                            tags=tags,
                            body_html=product.get('body_html', ''),
                            variant_id=str(variant.get('id', '')),
                            variant_title=variant.get('title', ''),
                            sku=variant.get('sku', ''),
                            price=price,
                            compare_at_price=compare_at_price,
                            available=bool(variant.get('available', False)),
                            stock_quantity=variant.get('inventory_quantity'),
                            variant_created_at=variant.get('created_at', ''),
                            variant_updated_at=variant.get('updated_at', ''),
                            image_src=first_image,
                            all_image_srcs=all_images
                        )
                        processed_products.append(product_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing Shopify variant: {e}")
                        continue
                    
            except Exception as e:
                self.logger.error(f"Error processing Shopify product: {e}")
                continue
                
        return processed_products

# --- WooCommerce A2A Agent ---
class WooCommerceA2AAgent(A2AAgent):
    """Agent A2A spécialisé pour WooCommerce - TEMPORARILY DISABLED"""
    
    def extract_data(self, store_config: StoreConfig) -> List[ProductData]:
        """Extrait les données produits d'un store WooCommerce"""
        
        # SKIP WOOCOMMERCE FOR NOW
        if self.config.skip_woocommerce:
            self.logger.warning(f"SKIPPING WooCommerce extraction for {store_config.domain} (temporarily disabled)")
            return []
        
        self.logger.info(f"Starting WooCommerce extraction for {store_config.domain}")
        
        if store_config.api_credentials:
            # Utilisation de l'API REST WooCommerce si disponible
            return self._extract_via_api(store_config)
        else:
            # Fallback vers scraping HTML
            return self._extract_via_scraping(store_config)
    
    def _extract_via_api(self, store_config: StoreConfig) -> List[ProductData]:
        """Extraction via API REST WooCommerce"""
        try:
            consumer_key = store_config.api_credentials.get('consumer_key')
            consumer_secret = store_config.api_credentials.get('consumer_secret')
            
            if not consumer_key or not consumer_secret:
                self.logger.error("Missing WooCommerce API credentials")
                return []
            
            base_url = f"https://{store_config.domain}/wp-json/wc/v3/products"
            auth = (consumer_key, consumer_secret)
            
            all_products = []
            page = 1
            per_page = 100
            
            while len(all_products) < self.config.max_products_per_store:
                params = {
                    'page': page,
                    'per_page': per_page,
                    'status': 'publish'
                }
                
                response = self.session.get(base_url, auth=auth, params=params, timeout=self.config.timeout)
                
                if response.status_code != 200:
                    self.logger.error(f"WooCommerce API error: {response.status_code}")
                    break
                
                products = response.json()
                if not products:
                    break
                
                all_products.extend(products)
                self.logger.info(f"Fetched WooCommerce page {page}: {len(products)} products")
                
                if len(products) < per_page:
                    break
                
                page += 1
                time.sleep(self.config.delay_between_requests)
            
            return self._process_woocommerce_products(all_products[:self.config.max_products_per_store], store_config)
            
        except Exception as e:
            self.logger.error(f"WooCommerce API extraction failed: {e}")
            return []
    
    def _extract_via_scraping(self, store_config: StoreConfig) -> List[ProductData]:
        """Extraction via scraping HTML pour WooCommerce"""
        self.logger.info(f"Using HTML scraping for WooCommerce store: {store_config.domain}")
        
        if not self.driver:
            self.logger.error("Selenium driver required for WooCommerce scraping")
            return []
        
        try:
            # Navigation vers la page produits
            shop_url = f"https://{store_config.domain}/shop"
            self.driver.get(shop_url)
            
            products = []
            page = 1
            
            while len(products) < self.config.max_products_per_store:
                self.logger.info(f"Scraping WooCommerce page {page}")
                
                # Attendre le chargement des produits
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".product, .woocommerce-product"))
                    )
                except TimeoutException:
                    self.logger.warning("No products found on page")
                    break
                
                # Extraire les liens produits
                product_links = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    ".product a, .woocommerce-product a, .product-title a"
                )
                
                page_products = []
                for link in product_links[:50]:  # Limite par page
                    try:
                        product_url = link.get_attribute('href')
                        if product_url and '/product/' in product_url:
                            product_data = self._scrape_product_details(product_url, store_config)
                            if product_data:
                                page_products.append(product_data)
                    except Exception as e:
                        self.logger.error(f"Error scraping product link: {e}")
                        continue
                
                products.extend(page_products)
                
                # Navigation vers page suivante
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, ".next, .page-numbers.next")
                    if next_button and next_button.is_enabled():
                        next_button.click()
                        time.sleep(self.config.delay_between_requests)
                        page += 1
                    else:
                        break
                except:
                    break
            
            return products[:self.config.max_products_per_store]
            
        except Exception as e:
            self.logger.error(f"WooCommerce scraping failed: {e}")
            return []
    
    def _scrape_product_details(self, product_url: str, store_config: StoreConfig) -> Optional[ProductData]:
        """Scrape les détails d'un produit WooCommerce"""
        try:
            self.driver.get(product_url)
            
            # Attendre le chargement de la page produit
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".product, .single-product"))
            )
            
            # Extraction des données de base
            title = self._safe_get_text(".product_title, .entry-title, h1")
            price_text = self._safe_get_text(".price .amount, .price, .woocommerce-Price-amount")
            price = self._extract_price(price_text)
            
            # Images
            image_elements = self.driver.find_elements(By.CSS_SELECTOR, ".product-image img, .woocommerce-product-gallery img")
            images = [img.get_attribute('src') for img in image_elements if img.get_attribute('src')]
            
            # Disponibilité
            stock_element = self.driver.find_element(By.CSS_SELECTOR, ".stock")
            available = "in-stock" in stock_element.get_attribute('class') if stock_element else True
            
            # Description
            description = self._safe_get_text(".product-description, .woocommerce-product-details__short-description")
            
            product_data = ProductData(
                store_domain=store_config.domain,
                store_region=store_config.region,
                platform="woocommerce",
                product_id=self._extract_product_id_from_url(product_url),
                title=title,
                handle=self._extract_handle_from_url(product_url),
                vendor=store_config.name,
                product_type="",
                created_at="",
                updated_at="",
                published_at="",
                tags=[],
                body_html=description,
                variant_id="",
                variant_title="",
                sku="",
                price=price,
                compare_at_price=None,
                available=available,
                stock_quantity=None,
                variant_created_at="",
                variant_updated_at="",
                image_src=images[0] if images else None,
                all_image_srcs=images
            )
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error scraping product details from {product_url}: {e}")
            return None
    
    def _safe_get_text(self, selector: str) -> str:
        """Extraction sécurisée de texte"""
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except:
            return ""
    
    def _extract_price(self, price_text: str) -> float:
        """Extrait le prix numérique du texte"""
        try:
            # Supprime les caractères non numériques sauf point et virgule
            price_clean = re.sub(r'[^\d.,]', '', price_text)
            price_clean = price_clean.replace(',', '.')
            return float(price_clean)
        except:
            return 0.0
    
    def _extract_product_id_from_url(self, url: str) -> str:
        """Extrait l'ID produit de l'URL"""
        try:
            # Cherche un pattern comme ?product_id=123 ou /product/123
            match = re.search(r'(?:product_id=|/product/.*?-|/product/)(\d+)', url)
            return match.group(1) if match else url.split('/')[-1]
        except:
            return ""
    
    def _extract_handle_from_url(self, url: str) -> str:
        """Extrait le handle du produit de l'URL"""
        try:
            return url.split('/')[-1].split('?')[0]
        except:
            return ""
    
    def _process_woocommerce_products(self, products: List[Dict], store_config: StoreConfig) -> List[ProductData]:
        """Traite les données produits WooCommerce depuis l'API"""
        processed_products = []
        
        for product in products:
            try:
                # Images
                images = product.get('images', [])
                first_image = images[0].get('src') if images else None
                all_images = [img.get('src') for img in images if img.get('src')]
                
                # Prix
                price = float(product.get('price', 0))
                regular_price = float(product.get('regular_price', 0)) if product.get('regular_price') else None
                
                # Tags et catégories
                tags = [tag.get('name', '') for tag in product.get('tags', [])]
                categories = [cat.get('name', '') for cat in product.get('categories', [])]
                tags.extend(categories)
                
                product_data = ProductData(
                    store_domain=store_config.domain,
                    store_region=store_config.region,
                    platform="woocommerce",
                    product_id=str(product.get('id', '')),
                    title=product.get('name', ''),
                    handle=product.get('slug', ''),
                    vendor=store_config.name,
                    product_type=categories[0] if categories else '',
                    created_at=product.get('date_created', ''),
                    updated_at=product.get('date_modified', ''),
                    published_at=product.get('date_created', ''),
                    tags=tags,
                    body_html=product.get('description', ''),
                    variant_id=str(product.get('id', '')),
                    variant_title=product.get('name', ''),
                    sku=product.get('sku', ''),
                    price=price,
                    compare_at_price=regular_price if regular_price != price else None,
                    available=product.get('in_stock', False),
                    stock_quantity=product.get('stock_quantity'),
                    variant_created_at=product.get('date_created', ''),
                    variant_updated_at=product.get('date_modified', ''),
                    image_src=first_image,
                    all_image_srcs=all_images,
                    rating=float(product.get('average_rating', 0)) if product.get('average_rating') else None,
                    review_count=product.get('rating_count', 0)
                )
                processed_products.append(product_data)
                
            except Exception as e:
                self.logger.error(f"Error processing WooCommerce product: {e}")
                continue
                
        return processed_products

# --- Agent Factory ---
class AgentFactory:
    """Factory pour créer les agents selon la plateforme"""
    
    @staticmethod
    def create_agent(platform: str, config: ScrapingConfig) -> A2AAgent:
        """Crée l'agent approprié selon la plateforme"""
        if platform.lower() == 'shopify':
            return ShopifyA2AAgent(config)
        elif platform.lower() == 'woocommerce':
            return WooCommerceA2AAgent(config)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

# --- Pipeline d'Extraction Unifié ---
class UnifiedExtractionPipeline:
    """Pipeline unifié pour l'extraction multi-plateformes"""
    
    def __init__(self, scraping_config: ScrapingConfig):
        self.scraping_config = scraping_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def extract_all_stores(self, stores: List[StoreConfig], 
                          output_file: str = "extracted_products.csv") -> List[ProductData]:
        """Extrait les données de tous les stores"""
        
        self.logger.info(f"Starting unified extraction pipeline for {len(stores)} stores")
        
        try:
            all_products = []
            
            # Grouper par plateforme pour optimiser
            stores_by_platform = {}
            for store in stores:
                platform = store.platform.lower()
                if platform not in stores_by_platform:
                    stores_by_platform[platform] = []
                stores_by_platform[platform].append(store)
            
            # SKIP WOOCOMMERCE STORES IF CONFIGURED
            if self.scraping_config.skip_woocommerce and 'woocommerce' in stores_by_platform:
                woocommerce_stores = stores_by_platform.pop('woocommerce')
                self.logger.warning(f"SKIPPING {len(woocommerce_stores)} WooCommerce stores (temporarily disabled)")
                for store in woocommerce_stores:
                    self.logger.warning(f"SKIPPED: {store.domain} (WooCommerce)")
            
            # Extraction par plateforme
            for platform, platform_stores in stores_by_platform.items():
                self.logger.info(f"=== Processing {platform.upper()} stores ===")
                
                try:
                    agent = AgentFactory.create_agent(platform, self.scraping_config)
                    
                    with agent:  # Context manager pour Selenium
                        for store in platform_stores:
                            try:
                                self.logger.info(f"Extracting from {store.domain} ({platform})")
                                products = agent.extract_data(store)
                                all_products.extend(products)
                                self.logger.info(f"SUCCESS: {store.domain}: {len(products)} products extracted")
                                
                                time.sleep(self.scraping_config.delay_between_domains)
                                
                            except Exception as e:
                                self.logger.error(f"ERROR: Error extracting from {store.domain}: {e}")
                                continue
                                
                except Exception as e:
                    self.logger.error(f"ERROR: Error with {platform} agent: {e}")
                    continue
            
            # Sauvegarde et statistiques
            if all_products:
                self._save_results(all_products, output_file)
                self._print_extraction_stats(all_products)
            else:
                self.logger.warning("No products extracted")
            
            return all_products
            
        except Exception as e:
            self.logger.error(f"Unified extraction pipeline failed: {e}")
            raise
    
    def _save_results(self, products: List[ProductData], filename: str):
        """Sauvegarde les résultats avec métadonnées enrichies"""
        try:
            df = pd.DataFrame([product.to_dict() for product in products])
            
            # Métadonnées
            df['extraction_timestamp'] = datetime.now().isoformat()
            df['pipeline_version'] = "unified_v1.0"
            df['total_products'] = len(products)
            
            # Sauvegarde principale
            df.to_csv(filename, index=False, encoding='utf-8')
            self.logger.info(f"SUCCESS: Results saved to {filename}")
            
            # Sauvegarde par plateforme
            for platform in df['platform'].unique():
                platform_df = df[df['platform'] == platform]
                platform_file = f"{platform}_{filename}"
                platform_df.to_csv(platform_file, index=False, encoding='utf-8')
                self.logger.info(f"SUCCESS: {platform.title()} results saved to {platform_file}")
            
            # Backup avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_{timestamp}_{filename}"
            df.to_csv(backup_file, index=False, encoding='utf-8')
            
        except Exception as e:
            self.logger.error(f"ERROR: Error saving results: {e}")
    
    def _print_extraction_stats(self, products: List[ProductData]):
        """Statistiques détaillées d'extraction"""
        if not products:
            return
        
        total = len(products)
        platforms = {}
        stores = {}
        available_count = 0
        
        for product in products:
            # Stats par plateforme
            if product.platform not in platforms:
                platforms[product.platform] = {'total': 0, 'available': 0}
            platforms[product.platform]['total'] += 1
            
            # Stats par store
            if product.store_domain not in stores:
                stores[product.store_domain] = {'total': 0, 'available': 0, 'platform': product.platform}
            stores[product.store_domain]['total'] += 1
            
            if product.available:
                available_count += 1
                platforms[product.platform]['available'] += 1
                stores[product.store_domain]['available'] += 1
        
        self.logger.info("COMPLETED === UNIFIED EXTRACTION STATISTICS ===")
        self.logger.info(f"STATS: Total products extracted: {total}")
        self.logger.info(f"STORES: Number of stores processed: {len(stores)}")
        self.logger.info(f"AVAILABLE: Available products: {available_count}/{total} ({available_count/total*100:.1f}%)")
        
        self.logger.info("\nPLATFORM === BY PLATFORM ===")
        for platform, stats in platforms.items():
            availability_rate = stats['available'] / stats['total'] * 100 if stats['total'] > 0 else 0
            self.logger.info(f"{platform.upper()}: {stats['total']} products ({stats['available']} available - {availability_rate:.1f}%)")
        
        self.logger.info("\nSTORE === BY STORE ===")
        for store, stats in stores.items():
            availability_rate = stats['available'] / stats['total'] * 100 if stats['total'] > 0 else 0
            self.logger.info(f"{store} ({stats['platform']}): {stats['total']} products ({stats['available']} available - {availability_rate:.1f}%)")

# --- Configuration et Execution ---
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/unified_extraction_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configuration des stores multi-plateformes - FOCUS ON SHOPIFY ONLY
    STORES = [
    # Boutiques Shopify (working perfectly)
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
     # ADD MORE SHOPIFY STORES INSTEAD OF WOOCOMMERCE
     StoreConfig(
        domain="casper.com",
        name="Casper",
        platform="shopify",
        region="US",
        currency="USD",
        priority=3
     ),
     
     # WooCommerce stores (will be skipped for now)
     StoreConfig(
        domain="barefootbuttons.com",
        name="Studio McGee",
        platform="woocommerce",  # This will be automatically skipped
        region="US",
        currency="USD",
        priority=10  # Low priority since it's skipped
     ),
    ]

    
    # Configuration du scraping avancée
    scraping_config = ScrapingConfig(
        delay_between_requests=2.0,
        delay_between_domains=3.0,
        max_products_per_store=5000,
        use_selenium=False,  # Disabled since we're skipping WooCommerce
        headless=True,
        timeout=45,
        skip_woocommerce=True  # NEW: Skip WooCommerce for now
    )
    
    try:
        print("==> Starting Unified A2A Extraction Pipeline...")
        print("=" * 60)
        print("NOTE: WooCommerce scraping is temporarily DISABLED")
        print("      Focus on Shopify stores which work perfectly!")
        print("=" * 60)
        
        # Exécution du pipeline unifié
        pipeline = UnifiedExtractionPipeline(scraping_config)
        extracted_products = pipeline.extract_all_stores(
            stores=STORES,
            output_file="unified_extracted_products.csv"
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS: EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"STATS: Total products extracted: {len(extracted_products)}")
        print(f"FILES: Main results file: 'unified_extracted_products.csv'")
        print(f"FILES: Platform-specific files: 'shopify_unified_extracted_products.csv'")
        print(f"LOGS: Detailed logs: 'unified_extraction_pipeline.log'")
        print("=" * 60)
        
        # Analyse rapide des résultats
        if extracted_products:
            platforms = {}
            for product in extracted_products:
                if product.platform not in platforms:
                    platforms[product.platform] = 0
                platforms[product.platform] += 1
            
            print("\nANALYSIS: QUICK ANALYSIS:")
            for platform, count in platforms.items():
                print(f"  - {platform.upper()}: {count} products")
            
            available_products = sum(1 for p in extracted_products if p.available)
            print(f"  - Available products: {available_products}/{len(extracted_products)} ({available_products/len(extracted_products)*100:.1f}%)")
            
            print("\nNEXT STEPS:")
            print("1. Store data in MongoDB: python mongo.py")
            print("2. Run ML analysis: python topk_analyzer.py")
            print("3. View dashboard: streamlit run dashboard.py")
        
    except KeyboardInterrupt:
        print("\nWARNING: Extraction interrupted by user")
    except Exception as e:
        print(f"\nERROR: Extraction failed: {e}")
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)

# === ADDITIONAL UTILITIES ===

class DataValidator:
    """Validateur pour les données extraites"""
    
    @staticmethod
    def validate_products(products: List[ProductData]) -> Dict[str, Any]:
        """Valide la qualité des données extraites"""
        stats = {
            'total_products': len(products),
            'valid_products': 0,
            'missing_title': 0,
            'missing_price': 0,
            'missing_images': 0,
            'invalid_price': 0,
            'platforms': {},
            'stores': {}
        }
        
        for product in products:
            # Validation de base
            is_valid = True
            
            if not product.title or product.title.strip() == '':
                stats['missing_title'] += 1
                is_valid = False
            
            if product.price <= 0:
                stats['invalid_price'] += 1
                is_valid = False
            
            if not product.image_src:
                stats['missing_images'] += 1
            
            if is_valid:
                stats['valid_products'] += 1
            
            # Stats par plateforme
            if product.platform not in stats['platforms']:
                stats['platforms'][product.platform] = 0
            stats['platforms'][product.platform] += 1
            
            # Stats par store
            if product.store_domain not in stats['stores']:
                stats['stores'][product.store_domain] = 0
            stats['stores'][product.store_domain] += 1
        
        return stats
    
    @staticmethod
    def print_validation_report(stats: Dict[str, Any]):
        """Affiche un rapport de validation"""
        print("\nVALIDATION === DATA VALIDATION REPORT ===")
        print(f"Total products: {stats['total_products']}")
        print(f"Valid products: {stats['valid_products']} ({stats['valid_products']/stats['total_products']*100:.1f}%)")
        print(f"Missing titles: {stats['missing_title']}")
        print(f"Invalid prices: {stats['invalid_price']}")
        print(f"Missing images: {stats['missing_images']}")
        
        print("\nPlatform distribution:")
        for platform, count in stats['platforms'].items():
            print(f"  - {platform}: {count} products")

class ExportManager:
    """Gestionnaire d'export pour différents formats"""
    
    @staticmethod
    def export_to_json(products: List[ProductData], filename: str):
        """Export en JSON"""
        try:
            data = [product.to_dict() for product in products]
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"SUCCESS: JSON export saved to {filename}")
        except Exception as e:
            print(f"ERROR: JSON export failed: {e}")
    
    @staticmethod
    def export_to_excel(products: List[ProductData], filename: str):
        """Export en Excel avec feuilles par plateforme"""
        try:
            df = pd.DataFrame([product.to_dict() for product in products])
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Feuille principale
                df.to_excel(writer, sheet_name='All Products', index=False)
                
                # Feuilles par plateforme
                for platform in df['platform'].unique():
                    platform_df = df[df['platform'] == platform]
                    sheet_name = f"{platform.title()} Products"
                    platform_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"SUCCESS: Excel export saved to {filename}")
        except Exception as e:
            print(f"ERROR: Excel export failed: {e}")

class ConfigManager:
    """Gestionnaire de configuration"""
    
    @staticmethod
    def load_stores_from_config(config_file: str) -> List[StoreConfig]:
        """Charge la configuration des stores depuis un fichier JSON"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            stores = []
            for store_data in config_data.get('stores', []):
                stores.append(StoreConfig(**store_data))
            
            return stores
        except Exception as e:
            print(f"ERROR: Error loading store config: {e}")
            return []
    
    @staticmethod
    def save_stores_config(stores: List[StoreConfig], config_file: str):
        """Sauvegarde la configuration des stores"""
        try:
            config_data = {
                'stores': [asdict(store) for store in stores],
                'created_at': datetime.now().isoformat()
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"SUCCESS: Store config saved to {config_file}")
        except Exception as e:
            print(f"ERROR: Error saving store config: {e}")

# === EXEMPLE D'UTILISATION AVANCÉE ===

def advanced_extraction_example():
    """Exemple d'utilisation avancée du pipeline - SHOPIFY FOCUSED"""
    
    # Configuration avancée
    advanced_config = ScrapingConfig(
        max_retries=5,
        timeout=60,
        delay_between_requests=1.5,
        delay_between_domains=5.0,
        max_products_per_store=10000,
        use_selenium=False,  # Not needed for Shopify
        headless=True,
        skip_woocommerce=True  # Skip WooCommerce
    )
    
    # Focus on Shopify stores only
    stores = [
        StoreConfig(
            domain="allbirds.com",
            name="Allbirds",
            platform="shopify",
            region="US",
            currency="USD",
            priority=1,
            custom_headers={
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9"
            }
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
            domain="casper.com",
            name="Casper",
            platform="shopify",
            region="US",
            currency="USD",
            priority=3
        )
    ]
    
    try:
        # Pipeline principal
        pipeline = UnifiedExtractionPipeline(advanced_config)
        products = pipeline.extract_all_stores(stores, "advanced_extraction.csv")
        
        # Validation des données
        validation_stats = DataValidator.validate_products(products)
        DataValidator.print_validation_report(validation_stats)
        
        # Exports multiples
        ExportManager.export_to_json(products, "products.json")
        ExportManager.export_to_excel(products, "products.xlsx")
        
        # Sauvegarde de la configuration
        ConfigManager.save_stores_config(stores, "stores_config.json")
        
        return products
        
    except Exception as e:
        print(f"ERROR: Advanced extraction failed: {e}")
        return []

# Pour exécuter l'exemple avancé (Shopify only):
# products = advanced_extraction_example()