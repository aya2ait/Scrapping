from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# MongoDB Configuration
MONGO_URI = 'mongodb://localhost:27017/'
DATABASE_NAME = 'products_db'
COLLECTION_NAME = 'products'

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductAnalyzer:
    def __init__(self, mongo_uri=MONGO_URI, db_name=DATABASE_NAME, collection_name=COLLECTION_NAME):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("Connected to MongoDB")
            return True
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
    
    def get_products_dataframe(self, filters: Dict = None) -> pd.DataFrame:
        """Retrieve products from MongoDB and convert to DataFrame"""
        try:
            # Apply filters if provided
            query = filters if filters else {}
            
            # Get products from MongoDB
            cursor = self.collection.find(query)
            products = list(cursor)
            
            if not products:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(products)
            
            # Convert ObjectId to string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving products: {e}")
            return pd.DataFrame()
    
    def calculate_synthetic_score(self, df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
        """Calculate synthetic score based on multiple criteria"""
        try:
            df_scored = df.copy()
            
            # Initialize score
            df_scored['synthetic_score'] = 0.0
            
            # Extract criteria weights
            weights = criteria.get('weights', {})
            total_weight = sum(weights.values()) if weights else 1
            
            # Normalize weights
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Price scoring (lower is better, but consider compare_at_price for discounts)
            if 'price' in normalized_weights and 'price' in df_scored.columns:
                price_weight = normalized_weights['price']
                
                # Handle price scoring based on criteria preference
                price_preference = criteria.get('price_preference', 'low')  # 'low', 'high', 'discount'
                
                if price_preference == 'low':
                    # Lower prices get higher scores
                    max_price = df_scored['price'].max()
                    df_scored['price_score'] = (max_price - df_scored['price']) / max_price
                elif price_preference == 'high':
                    # Higher prices get higher scores (premium products)
                    max_price = df_scored['price'].max()
                    df_scored['price_score'] = df_scored['price'] / max_price
                elif price_preference == 'discount':
                    # Products with better discounts get higher scores
                    df_scored['discount_pct'] = np.where(
                        df_scored['compare_at_price'].notna() & (df_scored['compare_at_price'] > df_scored['price']),
                        (df_scored['compare_at_price'] - df_scored['price']) / df_scored['compare_at_price'],
                        0
                    )
                    df_scored['price_score'] = df_scored['discount_pct']
                
                df_scored['price_score'] = df_scored['price_score'].fillna(0)
                df_scored['synthetic_score'] += df_scored['price_score'] * price_weight
            
            # Availability scoring
            if 'availability' in normalized_weights and 'available' in df_scored.columns:
                availability_weight = normalized_weights['availability']
                df_scored['availability_score'] = df_scored['available'].astype(int)
                df_scored['synthetic_score'] += df_scored['availability_score'] * availability_weight
            
            # Stock quantity scoring
            if 'stock' in normalized_weights and 'stock_quantity' in df_scored.columns:
                stock_weight = normalized_weights['stock']
                # Normalize stock quantity
                max_stock = df_scored['stock_quantity'].max()
                if max_stock > 0:
                    df_scored['stock_score'] = df_scored['stock_quantity'].fillna(0) / max_stock
                else:
                    df_scored['stock_score'] = 0
                df_scored['synthetic_score'] += df_scored['stock_score'] * stock_weight
            
            # Vendor popularity scoring (based on number of products from same vendor)
            if 'vendor_popularity' in normalized_weights and 'vendor' in df_scored.columns:
                vendor_weight = normalized_weights['vendor_popularity']
                vendor_counts = df_scored['vendor'].value_counts()
                df_scored['vendor_popularity_score'] = df_scored['vendor'].map(vendor_counts)
                max_vendor_count = df_scored['vendor_popularity_score'].max()
                if max_vendor_count > 0:
                    df_scored['vendor_popularity_score'] = df_scored['vendor_popularity_score'] / max_vendor_count
                else:
                    df_scored['vendor_popularity_score'] = 0
                df_scored['synthetic_score'] += df_scored['vendor_popularity_score'] * vendor_weight
            
            # Tags relevance scoring (if specific tags are preferred)
            if 'tags_relevance' in normalized_weights and 'tags' in df_scored.columns:
                tags_weight = normalized_weights['tags_relevance']
                preferred_tags = criteria.get('preferred_tags', [])
                
                if preferred_tags:
                    def calculate_tag_relevance(tags):
                        if not isinstance(tags, list):
                            return 0
                        matches = sum(1 for tag in tags if any(pref.lower() in tag.lower() for pref in preferred_tags))
                        return matches / len(preferred_tags) if preferred_tags else 0
                    
                    df_scored['tags_relevance_score'] = df_scored['tags'].apply(calculate_tag_relevance)
                else:
                    df_scored['tags_relevance_score'] = 0
                
                df_scored['synthetic_score'] += df_scored['tags_relevance_score'] * tags_weight
            
            # Recency scoring (newer products might be more attractive)
            if 'recency' in normalized_weights and 'created_at' in df_scored.columns:
                recency_weight = normalized_weights['recency']
                now = datetime.utcnow()
                
                # Convert to datetime if needed
                df_scored['created_at'] = pd.to_datetime(df_scored['created_at'])
                df_scored['days_since_creation'] = (now - df_scored['created_at']).dt.days
                
                max_days = df_scored['days_since_creation'].max()
                if max_days > 0:
                    df_scored['recency_score'] = 1 - (df_scored['days_since_creation'] / max_days)
                else:
                    df_scored['recency_score'] = 1
                
                df_scored['synthetic_score'] += df_scored['recency_score'] * recency_weight
            
            # Platform preference scoring
            if 'platform_preference' in normalized_weights and 'platform' in df_scored.columns:
                platform_weight = normalized_weights['platform_preference']
                preferred_platforms = criteria.get('preferred_platforms', [])
                
                if preferred_platforms:
                    df_scored['platform_preference_score'] = df_scored['platform'].apply(
                        lambda x: 1 if x in preferred_platforms else 0
                    )
                else:
                    df_scored['platform_preference_score'] = 1
                
                df_scored['synthetic_score'] += df_scored['platform_preference_score'] * platform_weight
            
            return df_scored
            
        except Exception as e:
            logger.error(f"Error calculating synthetic score: {e}")
            return df
    
    def apply_ml_scoring(self, df: pd.DataFrame, method: str = 'random_forest') -> pd.DataFrame:
        """Apply ML-based scoring for product success prediction"""
        try:
            df_ml = df.copy()
            
            # Prepare features for ML
            features = []
            feature_names = []
            
            # Numerical features
            if 'price' in df_ml.columns:
                features.append(df_ml['price'].fillna(df_ml['price'].median()))
                feature_names.append('price')
            
            if 'stock_quantity' in df_ml.columns:
                features.append(df_ml['stock_quantity'].fillna(0))
                feature_names.append('stock_quantity')
            
            if 'available' in df_ml.columns:
                features.append(df_ml['available'].astype(int))
                feature_names.append('available')
            
            # Categorical features (encoded)
            if 'platform' in df_ml.columns:
                platform_encoded = pd.get_dummies(df_ml['platform'], prefix='platform')
                for col in platform_encoded.columns:
                    features.append(platform_encoded[col])
                    feature_names.append(col)
            
            if len(features) < 2:
                logger.warning("Not enough features for ML scoring")
                return df_ml
            
            # Combine features
            X = np.column_stack(features)
            
            # Create synthetic target (combination of price attractiveness and availability)
            y = df_ml['synthetic_score'] if 'synthetic_score' in df_ml.columns else np.random.random(len(df_ml))
            
            # Apply ML method
            if method == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                df_ml['ml_score'] = model.predict(X)
                
                # Feature importance
                importance = dict(zip(feature_names, model.feature_importances_))
                df_ml['feature_importance'] = [importance] * len(df_ml)
                
            elif method == 'xgboost':
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                df_ml['ml_score'] = model.predict(X)
                
            elif method == 'lightgbm':
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                model.fit(X, y)
                df_ml['ml_score'] = model.predict(X)
            
            return df_ml
            
        except Exception as e:
            logger.error(f"Error in ML scoring: {e}")
            return df
    
    def cluster_products(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster products using KMeans"""
        try:
            df_cluster = df.copy()
            
            # Prepare features for clustering
            features = []
            
            if 'price' in df_cluster.columns:
                features.append(df_cluster['price'].fillna(df_cluster['price'].median()))
            
            if 'stock_quantity' in df_cluster.columns:
                features.append(df_cluster['stock_quantity'].fillna(0))
            
            if 'synthetic_score' in df_cluster.columns:
                features.append(df_cluster['synthetic_score'])
            
            if len(features) < 2:
                logger.warning("Not enough features for clustering")
                return df_cluster
            
            # Standardize features
            X = np.column_stack(features)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = df_cluster.groupby('cluster').agg({
                'price': ['mean', 'std'],
                'synthetic_score': ['mean', 'std'],
                'available': 'sum'
            }).round(2)
            
            df_cluster['cluster_stats'] = df_cluster['cluster'].apply(
                lambda x: cluster_stats.loc[x].to_dict()
            )
            
            return df_cluster
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return df
    
    def get_top_k_products(self, df: pd.DataFrame, k: int, score_column: str = 'synthetic_score') -> pd.DataFrame:
        """Get top K products based on score"""
        try:
            if score_column not in df.columns:
                logger.error(f"Score column '{score_column}' not found")
                return df.head(k)
            
            # Sort by score (descending) and get top K
            top_k = df.nlargest(k, score_column)
            
            return top_k
            
        except Exception as e:
            logger.error(f"Error getting top K products: {e}")
            return df.head(k)
    
    def analyze_by_geography(self, df: pd.DataFrame) -> Dict:
        """Analyze products by geographical regions"""
        try:
            geo_analysis = {}
            
            if 'store_region' in df.columns:
                region_stats = df.groupby('store_region').agg({
                    'price': ['mean', 'median', 'std'],
                    'available': 'sum',
                    'synthetic_score': 'mean',
                    'store_domain': 'nunique'
                }).round(2)
                
                geo_analysis['by_region'] = region_stats.to_dict()
                
                # Top products by region
                geo_analysis['top_by_region'] = {}
                for region in df['store_region'].unique():
                    region_products = df[df['store_region'] == region]
                    if not region_products.empty:
                        top_product = region_products.nlargest(1, 'synthetic_score').iloc[0]
                        geo_analysis['top_by_region'][region] = {
                            'title': top_product.get('title', 'Unknown'),
                            'price': top_product.get('price', 0),
                            'score': top_product.get('synthetic_score', 0),
                            'vendor': top_product.get('vendor', 'Unknown')
                        }
            
            return geo_analysis
            
        except Exception as e:
            logger.error(f"Error in geographical analysis: {e}")
            return {}
    
    def analyze_shops_ranking(self, df: pd.DataFrame) -> Dict:
        """Rank shops based on their flagship products"""
        try:
            shop_analysis = {}
            
            if 'store_domain' in df.columns:
                # Shop statistics
                shop_stats = df.groupby('store_domain').agg({
                    'synthetic_score': ['mean', 'max', 'count'],
                    'price': ['mean', 'median'],
                    'available': 'sum',
                    'vendor': 'nunique'
                }).round(2)
                
                shop_analysis['shop_statistics'] = shop_stats.to_dict()
                
                # Flagship products by shop
                shop_analysis['flagship_products'] = {}
                for shop in df['store_domain'].unique():
                    shop_products = df[df['store_domain'] == shop]
                    if not shop_products.empty:
                        flagship = shop_products.nlargest(1, 'synthetic_score').iloc[0]
                        shop_analysis['flagship_products'][shop] = {
                            'title': flagship.get('title', 'Unknown'),
                            'price': flagship.get('price', 0),
                            'score': flagship.get('synthetic_score', 0),
                            'vendor': flagship.get('vendor', 'Unknown'),
                            'available': flagship.get('available', False)
                        }
                
                # Top shops by average score
                top_shops = df.groupby('store_domain')['synthetic_score'].mean().nlargest(10)
                shop_analysis['top_shops'] = top_shops.to_dict()
            
            return shop_analysis
            
        except Exception as e:
            logger.error(f"Error in shop ranking: {e}")
            return {}

# Initialize analyzer
analyzer = ProductAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database_connected': analyzer.client is not None
    })

@app.route('/api/top-k-products', methods=['POST'])
def get_top_k_products():
    """
    Main endpoint to get Top-K products based on criteria
    
    Expected JSON payload:
    {
        "k": 10,
        "criteria": {
            "weights": {
                "price": 0.3,
                "availability": 0.2,
                "stock": 0.2,
                "vendor_popularity": 0.1,
                "tags_relevance": 0.1,
                "recency": 0.1
            },
            "price_preference": "low",  # "low", "high", "discount"
            "preferred_tags": ["electronics", "fashion"],
            "preferred_platforms": ["shopify", "woocommerce"]
        },
        "filters": {
            "available": true,
            "price": {"$gte": 10, "$lte": 1000},
            "store_region": "US"
        },
        "ml_method": "random_forest",  # "random_forest", "xgboost", "lightgbm"
        "clustering": {
            "enabled": true,
            "n_clusters": 5
        },
        "include_analysis": {
            "geography": true,
            "shops_ranking": true
        }
    }
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        k = data.get('k', 10)
        criteria = data.get('criteria', {})
        filters = data.get('filters', {})
        ml_method = data.get('ml_method', 'random_forest')
        clustering_config = data.get('clustering', {})
        include_analysis = data.get('include_analysis', {})
        
        # Validate k
        if k <= 0 or k > 1000:
            return jsonify({'error': 'k must be between 1 and 1000'}), 400
        
        # Get products from database
        logger.info(f"Retrieving products with filters: {filters}")
        df = analyzer.get_products_dataframe(filters)
        
        if df.empty:
            return jsonify({
                'error': 'No products found matching the criteria',
                'total_products': 0,
                'top_k_products': []
            }), 404
        
        logger.info(f"Found {len(df)} products")
        
        # Calculate synthetic score
        df_scored = analyzer.calculate_synthetic_score(df, criteria)
        
        # Apply ML scoring if requested
        if ml_method and ml_method != 'none':
            df_scored = analyzer.apply_ml_scoring(df_scored, ml_method)
            score_column = 'ml_score'
        else:
            score_column = 'synthetic_score'
        
        # Apply clustering if requested
        if clustering_config.get('enabled', False):
            n_clusters = clustering_config.get('n_clusters', 5)
            df_scored = analyzer.cluster_products(df_scored, n_clusters)
        
        # Get top K products
        top_k_df = analyzer.get_top_k_products(df_scored, k, score_column)
        
        # Prepare response
        response = {
            'total_products': len(df),
            'k': k,
            'score_method': ml_method if ml_method != 'none' else 'synthetic',
            'criteria_used': criteria,
            'top_k_products': []
        }
        
        # Convert top K products to JSON
        for idx, row in top_k_df.iterrows():
            product = {
                'id': str(row.get('_id', '')),
                'title': row.get('title', ''),
                'vendor': row.get('vendor', ''),
                'price': row.get('price', 0),
                'compare_at_price': row.get('compare_at_price'),
                'available': row.get('available', False),
                'stock_quantity': row.get('stock_quantity', 0),
                'store_domain': row.get('store_domain', ''),
                'store_region': row.get('store_region', ''),
                'platform': row.get('platform', ''),
                'tags': row.get('tags', []),
                'synthetic_score': round(row.get('synthetic_score', 0), 4),
                'rank': len(response['top_k_products']) + 1
            }
            
            # Add ML score if available
            if 'ml_score' in row:
                product['ml_score'] = round(row['ml_score'], 4)
            
            # Add cluster info if available
            if 'cluster' in row:
                product['cluster'] = int(row['cluster'])
            
            response['top_k_products'].append(product)
        
        # Add additional analysis if requested
        if include_analysis.get('geography', False):
            response['geographical_analysis'] = analyzer.analyze_by_geography(df_scored)
        
        if include_analysis.get('shops_ranking', False):
            response['shops_analysis'] = analyzer.analyze_shops_ranking(df_scored)
        
        # Add summary statistics
        response['statistics'] = {
            'average_score': round(df_scored[score_column].mean(), 4),
            'score_std': round(df_scored[score_column].std(), 4),
            'price_range': {
                'min': float(df_scored['price'].min()) if 'price' in df_scored.columns else None,
                'max': float(df_scored['price'].max()) if 'price' in df_scored.columns else None,
                'avg': round(float(df_scored['price'].mean()), 2) if 'price' in df_scored.columns else None
            },
            'availability_rate': round(df_scored['available'].mean(), 2) if 'available' in df_scored.columns else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in top-k products endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/criteria-suggestions', methods=['GET'])
def get_criteria_suggestions():
    """Get suggestions for criteria based on available data"""
    try:
        # Sample some products to understand data structure
        sample_df = analyzer.get_products_dataframe({})
        
        if sample_df.empty:
            return jsonify({'error': 'No products found in database'}), 404
        
        suggestions = {
            'available_criteria': {
                'price': 'Product price (numerical)',
                'availability': 'Product availability (boolean)',
                'stock': 'Stock quantity (numerical)',
                'vendor_popularity': 'Vendor reputation based on product count',
                'tags_relevance': 'Relevance to preferred tags',
                'recency': 'How recently the product was added',
                'platform_preference': 'Preference for specific platforms'
            },
            'price_preferences': ['low', 'high', 'discount'],
            'available_platforms': sample_df['platform'].unique().tolist() if 'platform' in sample_df.columns else [],
            'available_regions': sample_df['store_region'].unique().tolist() if 'store_region' in sample_df.columns else [],
            'sample_tags': [],
            'ml_methods': ['random_forest', 'xgboost', 'lightgbm', 'none'],
            'example_request': {
                "k": 10,
                "criteria": {
                    "weights": {
                        "price": 0.3,
                        "availability": 0.25,
                        "stock": 0.2,
                        "vendor_popularity": 0.15,
                        "recency": 0.1
                    },
                    "price_preference": "low"
                },
                "filters": {
                    "available": True,
                    "price": {"$gte": 10, "$lte": 500}
                },
                "ml_method": "random_forest",
                "include_analysis": {
                    "geography": True,
                    "shops_ranking": True
                }
            }
        }
        
        # Extract sample tags
        if 'tags' in sample_df.columns:
            all_tags = []
            for tags in sample_df['tags'].dropna():
                if isinstance(tags, list):
                    all_tags.extend(tags)
            
            # Get most common tags
            from collections import Counter
            tag_counts = Counter(all_tags)
            suggestions['sample_tags'] = [tag for tag, count in tag_counts.most_common(20)]
        
        return jsonify(suggestions)
        
    except Exception as e:
        logger.error(f"Error getting criteria suggestions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/database-stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    try:
        df = analyzer.get_products_dataframe({})
        
        if df.empty:
            return jsonify({'error': 'No products found in database'}), 404
        
        stats = {
            'total_products': len(df),
            'available_products': int(df['available'].sum()) if 'available' in df.columns else 0,
            'platforms': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
            'regions': df['store_region'].value_counts().to_dict() if 'store_region' in df.columns else {},
            'vendors': df['vendor'].nunique() if 'vendor' in df.columns else 0,
            'stores': df['store_domain'].nunique() if 'store_domain' in df.columns else 0,
            'price_stats': {
                'min': float(df['price'].min()) if 'price' in df.columns else None,
                'max': float(df['price'].max()) if 'price' in df.columns else None,
                'mean': round(float(df['price'].mean()), 2) if 'price' in df.columns else None,
                'median': round(float(df['price'].median()), 2) if 'price' in df.columns else None
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Top-K Products Analysis API...")
    print("="*60)
    print("Available endpoints:")
    print("• GET  /health - Health check")
    print("• POST /api/top-k-products - Get top K products")
    print("• GET  /api/criteria-suggestions - Get criteria suggestions")
    print("• GET  /api/database-stats - Get database statistics")
    print("="*60)
    print("Server starting on http://localhost:5000")
    print("Use Postman to test the API endpoints")
    
    app.run(host='0.0.0.0', port=5000, debug=True)