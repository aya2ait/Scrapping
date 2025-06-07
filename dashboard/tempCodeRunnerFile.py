import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
from enum import Enum
import pandas as pd  


# Import du ProductAnalyzer réel (depuis paste.py)
# On l'importe ici car ProductDataMCPServer en aura besoin pour simuler l'accès aux données
try:
    from paste import ProductAnalyzer
except ImportError:
    # Gérer l'erreur si paste.py n'est pas disponible, ou le faire remonter
    # Pour l'intégration, on peut lever une erreur si crucial
    ProductAnalyzer = None # Ou une classe Mock si vous voulez des tests unitaires isolés


# Import du BuyerLLMAnalyzer réel (vous devrez le passer depuis streamlit_app.py)
# Pour l'instant, on n'importe pas directement ici car il vient d'un autre fichier (streamlit_app.py)
# Il sera passé en paramètre.

# MCP Protocol Implementation
class MCPMessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class MCPPermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

@dataclass
class MCPMessage:
    """Message standardisé MCP"""
    id: str
    type: MCPMessageType
    timestamp: datetime
    source: str
    target: str
    payload: Dict[str, Any]
    permissions_required: List[MCPPermissionLevel]

@dataclass
class MCPContext:
    """Contexte d'exécution MCP"""
    session_id: str
    user_id: str
    permissions: List[MCPPermissionLevel]
    usage_limits: Dict[str, int]
    audit_trail: List[Dict[str, Any]]

class MCPServer(ABC):
    """Interface abstraite pour un serveur MCP"""
    
    def __init__(self, server_id: str, name: str, description: str):
        self.server_id = server_id
        self.name = name
        self.description = description
        self.capabilities = []
        self.audit_logger = logging.getLogger(f"mcp.{server_id}")
        
    @abstractmethod
    async def handle_request(self, message: MCPMessage, context: MCPContext) -> Dict[str, Any]:
        """Traite une requête MCP"""
        pass
    
    @abstractmethod
    def get_exposed_tools(self) -> List[Dict[str, Any]]:
        """Retourne la liste des outils exposés"""
        pass
    
    def log_interaction(self, message: MCPMessage, context: MCPContext, result: Any):
        """Journalise les interactions pour audit"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": context.session_id,
            "user_id": context.user_id,
            "message_id": message.id,
            "source": message.source,
            "target": message.target,
            "action": message.payload.get("action", "N/A"), # Ajouter l'action
            "permissions_used": [p.value for p in message.permissions_required],
            "success": result is not None and result.get("status") == "success" # Plus précis
        }
        self.audit_logger.info(json.dumps(log_entry))
        context.audit_trail.append(log_entry)

class ProductDataMCPServer(MCPServer):
    """Serveur MCP pour l'accès aux données produits"""
    
    # MODIFICATION: Ajout de product_analyzer_instance dans le constructeur
    def __init__(self, product_analyzer_instance: Any = None): # Type Any pour éviter les dépendances circulaires strictes
        super().__init__(
            server_id="product_data",
            name="Product Data Server",
            description="Serveur d'accès sécurisé aux données produits"
        )
        self.capabilities = ["read_products", "filter_products", "aggregate_data"]
        # Utilisez l'instance de ProductAnalyzer passée en paramètre
        if product_analyzer_instance is None and ProductAnalyzer is not None:
             self.data_source = ProductAnalyzer() # Fallback si non passé mais disponible
        else:
             self.data_source = product_analyzer_instance
        
        if self.data_source is None:
            logging.warning("ProductAnalyzer n'a pas été fourni ou importé dans ProductDataMCPServer.")
            # Vous pourriez vouloir lever une erreur ici si la disponibilité de l'analyseur est critique
            # raise ValueError("ProductAnalyzer instance is required for ProductDataMCPServer")

    async def handle_request(self, message: MCPMessage, context: MCPContext) -> Dict[str, Any]:
        """Traite les requêtes de données produits"""
        
        if MCPPermissionLevel.READ not in context.permissions:
            self.log_interaction(message, context, {"status": "error", "message": "Accès lecture requis"})
            raise PermissionError("Accès lecture requis pour les données produits")
        
        if self.data_source is None:
            self.log_interaction(message, context, {"status": "error", "message": "ProductAnalyzer non initialisé"})
            return {"status": "error", "message": "ProductAnalyzer non initialisé pour la récupération des données."}

        action = message.payload.get("action")
        params = message.payload.get("parameters", {})
        
        try:
            # Récupération des critères pour le scoring si l'action implique un score
            criteria = params.get("criteria", {})
            k_products = params.get("k_products", 10)

            if action == "get_products":
                result = await self._get_products(params, context, criteria)
            elif action == "filter_products":
                result = await self._filter_products(params, context, criteria, k_products)
            elif action == "aggregate_stats":
                result = await self._aggregate_stats(params, context, criteria)
            else:
                raise ValueError(f"Action non supportée: {action}")
                
            self.log_interaction(message, context, {"status": "success", "data": result})
            return {"status": "success", "data": result}
            
        except Exception as e:
            self.log_interaction(message, context, {"status": "error", "message": str(e)})
            return {"status": "error", "message": str(e)}
    
    # MODIFICATION: Ajout de 'criteria' pour les fonctions internes
    async def _get_products(self, params: Dict, context: MCPContext, criteria: Dict) -> List[Dict]:
        """Récupère les produits avec limites de contexte"""
        limit = min(params.get("limit", 100), context.usage_limits.get("max_products_per_request", 100))
        
        # Utiliser l'instance réelle de ProductAnalyzer
        filters = params.get("filters", {})
        df = self.data_source.get_products_dataframe(filters)
        
        if df.empty:
            return []

        # Appliquer le scoring si des critères sont fournis
        if criteria:
            df_scored = self.data_source.calculate_synthetic_score(df, criteria)
        else:
            df_scored = df.copy() # Pas de scoring si pas de critères

        # Convertir en liste de dictionnaires et limiter
        return df_scored.head(limit).to_dict(orient='records')
    
    # MODIFICATION: Ajout de 'criteria' et 'k_products'
    async def _filter_products(self, params: Dict, context: MCPContext, criteria: Dict, k_products: int) -> List[Dict]:
        """Filtre les produits selon des critères"""
        filters = params.get("product_filters", {}) # Utiliser "product_filters" comme dans Streamlit
        
        df = self.data_source.get_products_dataframe(filters)

        if df.empty:
            return []
        
        # Appliquer le scoring
        df_scored = self.data_source.calculate_synthetic_score(df, criteria)
        
        # Appliquer les filtres spécifiques à la méthode _filter_products (si différents des filtres de base)
        if "min_price" in filters:
            df_scored = df_scored[df_scored["price"] >= filters["min_price"]]
        if "max_price" in filters:
            df_scored = df_scored[df_scored["price"] <= filters["max_price"]]
        if "available_only" in filters and filters["available_only"]:
            df_scored = df_scored[df_scored["available"] == True]
            
        # Sélectionner les top K produits après filtrage et scoring
        top_k_df = self.data_source.get_top_k_products(df_scored, k_products, 'synthetic_score')

        # Collecter les analyses supplémentaires pour l'LLM
        geo_analysis = self.data_source.analyze_by_geography(df_scored)
        shops_analysis = self.data_source.analyze_shops_ranking(df_scored)

        # Retourner les données structurées pour l'LLM
        llm_data_for_analysis = {
            'total_products': len(df), # Nombre total de produits avant K
            'k': k_products,
            # 'budget': max_budget, # Si nécessaire, le passer en paramètre du client MCP
            'top_k_products': top_k_df.to_dict(orient='records'),
            'statistics': {
                'average_score': df_scored['synthetic_score'].mean() if 'synthetic_score' in df_scored.columns and not df_scored.empty else 0,
                'score_std': df_scored['synthetic_score'].std() if 'synthetic_score' in df_scored.columns and not df_scored.empty else 0,
                'price_range': {
                    'min': df_scored['price'].min() if 'price' in df_scored.columns and not df_scored.empty else None,
                    'max': df_scored['price'].max() if 'price' in df_scored.columns and not df_scored.empty else None,
                    'avg': df_scored['price'].mean() if 'price' in df_scored.columns and not df_scored.empty else None
                },
                'availability_rate': df_scored['available'].mean() * 100 if 'available' in df_scored.columns and not df_scored.empty else None
            },
            'geographical_analysis': geo_analysis,
            'shops_analysis': shops_analysis
        }
            
        return llm_data_for_analysis
    
    # MODIFICATION: Ajout de 'criteria'
    async def _aggregate_stats(self, params: Dict, context: MCPContext, criteria: Dict) -> Dict:
        """Agrège les statistiques produits"""
        products = await self._get_products(params, context, criteria) # Utilisez _get_products qui applique déjà les filtres et le scoring
        
        if not products:
            return {
                "total_products": 0,
                "avg_price": 0,
                "avg_score": 0,
                "availability_rate": 0
            }
        
        df_products = pd.DataFrame(products)

        return {
            "total_products": len(df_products),
            "avg_price": df_products["price"].mean() if "price" in df_products.columns else 0,
            "avg_score": df_products["synthetic_score"].mean() if "synthetic_score" in df_products.columns else 0,
            "availability_rate": df_products["available"].mean() * 100 if "available" in df_products.columns else 0
        }
    
    def get_exposed_tools(self) -> List[Dict[str, Any]]:
        # ... (aucune modification ici)
        return [
            {
                "name": "get_products",
                "description": "Récupère une liste de produits",
                "parameters": {
                    "limit": {"type": "integer", "description": "Nombre max de produits"},
                    "filters": {"type": "object", "description": "Filtres pour les produits (ex: {'price': {'$gte': 10}})"},
                    "criteria": {"type": "object", "description": "Critères de scoring pour les produits"}
                },
                "permissions_required": ["read"]
            },
            {
                "name": "filter_products", 
                "description": "Filtre les produits selon des critères et retourne les données structurées pour l'analyse LLM",
                "parameters": {
                    "product_filters": {"type": "object", "description": "Filtres spécifiques pour les produits (ex: {'min_price': 50, 'available_only': true})"},
                    "criteria": {"type": "object", "description": "Critères de scoring pour les produits (poids pour prix, qualité, etc.)"},
                    "k_products": {"type": "integer", "description": "Nombre de top K produits à inclure dans les résultats structurés"}
                },
                "permissions_required": ["read"]
            },
            {
                "name": "aggregate_stats",
                "description": "Agrège les statistiques clés des produits",
                "parameters": {
                    "filters": {"type": "object", "description": "Filtres pour les produits à agréger"},
                    "criteria": {"type": "object", "description": "Critères de scoring à appliquer avant l'agrégation"}
                },
                "permissions_required": ["read"]
            }
        ]

class LLMAnalysisMCPServer(MCPServer):
    """Serveur MCP pour l'analyse LLM responsable"""
    
    # MODIFICATION: Ajout de buyer_llm_analyzer_instance dans le constructeur
    def __init__(self, buyer_llm_analyzer_instance: Any): # Type Any pour éviter les dépendances circulaires strictes
        super().__init__(
            server_id="llm_analysis",
            name="LLM Analysis Server", 
            description="Serveur d'analyse LLM avec contrôles éthiques"
        )
        self.llm_analyzer = buyer_llm_analyzer_instance # Utilisez l'instance passée en paramètre
        if self.llm_analyzer is None:
            raise ValueError("BuyerLLMAnalyzer instance is required for LLMAnalysisMCPServer")
        self.capabilities = ["buyer_analysis", "market_insights", "recommendations"]
        
    async def handle_request(self, message: MCPMessage, context: MCPContext) -> Dict[str, Any]:
        """Traite les requêtes d'analyse LLM"""
        
        if MCPPermissionLevel.EXECUTE not in context.permissions:
            self.log_interaction(message, context, {"status": "error", "message": "Permission d'exécution requise pour l'analyse LLM"})
            raise PermissionError("Permission d'exécution requise pour l'analyse LLM")
        
        action = message.payload.get("action")
        params = message.payload.get("parameters", {})
        
        try:
            self._validate_analysis_request(params, context) # Passage du contexte pour les limites d'usage
            
            if action == "buyer_analysis":
                result = await self._perform_buyer_analysis(params, context)
            elif action == "generate_insights":
                result = await self._generate_market_insights(params, context) # Assurez-vous d'implémenter cette méthode si elle est exposée
            else:
                raise ValueError(f"Action non supportée: {action}")
                
            result = self._filter_generated_content(result)
            
            self.log_interaction(message, context, {"status": "success", "data": result})
            return {"status": "success", "data": result}
            
        except Exception as e:
            self.log_interaction(message, context, {"status": "error", "message": str(e)})
            return {"status": "error", "message": str(e)}
    
    # MODIFICATION: Ajout de 'context' pour les limites d'usage
    def _validate_analysis_request(self, params: Dict, context: MCPContext):
        """Valide les paramètres de requête d'analyse"""
        if "prompt_injection" in str(params).lower():
            raise ValueError("Tentative d'injection de prompt détectée")
        
        # Limite de taille des données basée sur le contexte
        max_length = context.usage_limits.get("max_analysis_length", 10000)
        if len(json.dumps(params)) > max_length: # Utiliser json.dumps pour une taille plus réaliste
            raise ValueError(f"Paramètres trop volumineux (max {max_length} caractères)")
        
        # Vous pouvez également implémenter des limites sur les appels LLM ici
        # Par exemple, un compteur dans le contexte ou un système de token bucket
        # context.usage_limits["llm_calls_count"] += 1
        # if context.usage_limits["llm_calls_count"] > context.usage_limits["max_llm_calls_per_hour"]:
        #    raise UsageLimitExceededError("Too many LLM calls")

    async def _perform_buyer_analysis(self, params: Dict, context: MCPContext) -> Dict:
        """Effectue l'analyse d'achat de manière responsable"""
        analysis_type = params.get("analysis_type", "general")
        products_data = params.get("products_data", {}) # Ce sont les données structurées de ProductDataMCPServer
        
        # Limitation du contexte pour éviter les abus, basée sur le contexte MCP
        max_products_for_llm = context.usage_limits.get("max_products_per_llm_analysis", 50)
        if len(products_data.get("top_k_products", [])) > max_products_for_llm:
            products_data["top_k_products"] = products_data["top_k_products"][:max_products_for_llm]
            logging.warning(f"Tronqué la liste des produits pour l'analyse LLM à {max_products_for_llm} en raison des limites d'usage.")
            
        # Appeler l'analyseur LLM réel
        analysis_text = self.llm_analyzer.analyze_for_buyer(products_data, analysis_type)
        
        return {
            "analysis": analysis_text,
            "context_info": {
                "products_analyzed": len(products_data.get("top_k_products", [])),
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    # Nouvelle méthode _generate_market_insights si vous l'exposez
    async def _generate_market_insights(self, params: Dict, context: MCPContext) -> Dict:
        """Génère des insights marché (à implémenter)"""
        # Exemple : utiliser llm_analyzer pour une autre fonction
        # insights_text = self.llm_analyzer.generate_market_insights(params.get("market_data"))
        # return {"insights": insights_text}
        raise NotImplementedError("La fonction generate_market_insights n'est pas encore implémentée.")
        
    def _filter_generated_content(self, content: Dict) -> Dict:
        """Filtre le contenu généré pour éviter les sorties inappropriées"""
        if isinstance(content.get("analysis"), str):
            analysis = content["analysis"]
            sensitive_patterns = ["api_key", "password", "secret", "token", "identifiant"] # Ajout de motifs
            for pattern in sensitive_patterns:
                analysis = analysis.replace(pattern, "[REDACTED]")
            content["analysis"] = analysis
        return content
    
    def get_exposed_tools(self) -> List[Dict[str, Any]]:
        # ... (aucune modification ici)
        return [
            {
                "name": "buyer_analysis",
                "description": "Analyse d'achat intelligente et responsable basée sur les données de produits structurées.",
                "parameters": {
                    "analysis_type": {"type": "string", "enum": ["general", "budget", "quality", "urgency"], "description": "Type d'analyse demandé"},
                    "products_data": {"type": "object", "description": "Données structurées des produits à analyser (générées par le serveur de données produits)."}
                },
                "permissions_required": ["execute"]
            },
            {
                "name": "generate_insights",
                "description": "Génère des insights de marché basés sur des données agrégées.",
                "parameters": {
                    "market_data": {"type": "object", "description": "Données agrégées du marché pour l'analyse."}
                },
                "permissions_required": ["execute"]
            }
        ]

class MCPClient:
    """Client MCP pour orchestrer les interactions entre serveurs"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.active_contexts: Dict[str, MCPContext] = {}
        
    def register_server(self, server: MCPServer):
        """Enregistre un serveur MCP"""
        self.servers[server.server_id] = server
        
    def create_context(self, user_id: str, permissions: List[MCPPermissionLevel]) -> MCPContext:
        """Crée un contexte d'exécution"""
        session_id = str(uuid.uuid4())
        context = MCPContext(
            session_id=session_id,
            user_id=user_id,
            permissions=permissions,
            usage_limits={
                "max_products_per_request": 100, # Limite pour ProductDataMCPServer
                "max_products_per_llm_analysis": 50, # Nouvelle limite pour LLMAnalysisMCPServer
                "max_llm_calls_per_hour": 50,
                "max_analysis_length": 10000 # Caractères pour les paramètres LLM
            },
            audit_trail=[]
        )
        self.active_contexts[session_id] = context
        return context
    
    async def send_request(self, server_id: str, action: str, parameters: Dict, 
                           context: MCPContext, permissions_required: List[MCPPermissionLevel]) -> Dict:
        """Envoie une requête à un serveur MCP"""
        
        if server_id not in self.servers:
            raise ValueError(f"Serveur MCP non trouvé: {server_id}")
        
        for perm in permissions_required:
            if perm not in context.permissions:
                raise PermissionError(f"Permission manquante: {perm.value}")
        
        message = MCPMessage(
            id=str(uuid.uuid4()),
            type=MCPMessageType.REQUEST,
            timestamp=datetime.now(),
            source="mcp_client",
            target=server_id,
            payload={"action": action, "parameters": parameters},
            permissions_required=permissions_required
        )
        
        server = self.servers[server_id]
        return await server.handle_request(message, context)

class ResponsibleBuyerAnalyzer:
    """Analyseur d'achat utilisant l'architecture MCP responsable"""
    
    # MODIFICATION: Ajout des instances réelles pour les serveurs MCP
    def __init__(self, groq_api_key: str, product_analyzer_instance: Any, buyer_llm_analyzer_instance: Any):
        self.mcp_client = MCPClient()
        
        # Enregistrement des serveurs MCP avec les instances réelles
        self.product_server = ProductDataMCPServer(product_analyzer_instance=product_analyzer_instance)
        self.llm_server = LLMAnalysisMCPServer(buyer_llm_analyzer_instance=buyer_llm_analyzer_instance)
        
        self.mcp_client.register_server(self.product_server)
        self.mcp_client.register_server(self.llm_server)
        
    async def perform_responsible_analysis(self, user_id: str, analysis_params: Dict) -> Dict:
        """Effectue une analyse responsable via MCP"""
        
        context = self.mcp_client.create_context(
            user_id=user_id,
            permissions=[MCPPermissionLevel.READ, MCPPermissionLevel.EXECUTE]
        )
        
        try:
            # Étape 1: Récupération et filtrage des données produits
            # Passons les critères de scoring et k_products au ProductDataMCPServer
            product_data_request_params = {
                "product_filters": analysis_params.get("product_filters", {}),
                "criteria": analysis_params.get("criteria", {}), # Critères pour le scoring
                "k_products": analysis_params.get("k_products", 10) # Nombre de produits à récupérer
            }

            products_response = await self.mcp_client.send_request(
                server_id="product_data",
                action="filter_products", # Utilisation de filter_products pour obtenir les données structurées
                parameters=product_data_request_params,
                context=context,
                permissions_required=[MCPPermissionLevel.READ]
            )
            
            if products_response["status"] != "success":
                raise Exception(f"Erreur récupération produits: {products_response.get('message')}")
            
            # Les données de `products_response["data"]` contiennent déjà les `top_k_products` et les statistiques
            llm_input_data = products_response["data"]
            
            # Étape 2: Analyse LLM responsable
            analysis_response = await self.mcp_client.send_request(
                server_id="llm_analysis", 
                action="buyer_analysis",
                parameters={
                    "analysis_type": analysis_params.get("analysis_type", "general"),
                    "products_data": llm_input_data # Passez directement les données structurées pour l'LLM
                },
                context=context,
                permissions_required=[MCPPermissionLevel.EXECUTE]
            )
            
            if analysis_response["status"] != "success":
                raise Exception(f"Erreur analyse LLM: {analysis_response.get('message')}")
            
            # Retour des résultats avec métadonnées de traçabilité
            return {
                "status": "success",
                "analysis": analysis_response["data"].get("analysis"), # Le texte d'analyse de l'IA
                "products": llm_input_data.get("top_k_products", []), # Les produits qui ont été analysés par l'IA
                "context": {
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "audit_trail": context.audit_trail,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse responsable: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "context": {
                    "session_id": context.session_id,
                    "audit_trail": context.audit_trail
                }
            }

# Exemple d'utilisation intégrée avec Streamlit (déjà dans votre fichier Streamlit)
# def integrate_mcp_with_streamlit():
#    ...

if __name__ == "__main__":
    # Test de l'architecture MCP
    async def test_mcp():
        # Pour les tests unitaires de MCP, vous pouvez simuler ProductAnalyzer et BuyerLLMAnalyzer
        class MockProductAnalyzer:
            def get_products_dataframe(self, filters):
                print(f"MockProductAnalyzer: get_products_dataframe avec filtres: {filters}")
                # Données de mock pour les tests
                return pd.DataFrame([
                    {"id": "p1", "title": "Produit Test 1", "price": 100, "synthetic_score": 0.9, "available": True, "vendor": "VendorA", "stock_quantity": 20, "platform": "Web", "store_region": "EU"},
                    {"id": "p2", "title": "Produit Test 2", "price": 150, "synthetic_score": 0.7, "available": True, "vendor": "VendorB", "stock_quantity": 10, "platform": "App", "store_region": "US"},
                    {"id": "p3", "title": "Produit Test 3", "price": 50, "synthetic_score": 0.85, "available": False, "vendor": "VendorA", "stock_quantity": 0, "platform": "Web", "store_region": "EU"},
                    {"id": "p4", "title": "Produit Test 4", "price": 200, "synthetic_score": 0.6, "available": True, "vendor": "VendorC", "stock_quantity": 5, "platform": "App", "store_region": "CA"},
                ])
            def calculate_synthetic_score(self, df, criteria):
                print(f"MockProductAnalyzer: calculate_synthetic_score avec critères: {criteria}")
                # Simple calcul de score synthétique pour le mock
                df['synthetic_score'] = df['price'].apply(lambda x: 1 - (x / 200) * 0.5) # Exemple très simple
                return df
            def get_top_k_products(self, df, k, score_col):
                print(f"MockProductAnalyzer: get_top_k_products k={k}, score_col={score_col}")
                return df.nlargest(k, score_col)
            def analyze_by_geography(self, df): return {"EU": 2, "US": 1, "CA": 1}
            def analyze_shops_ranking(self, df): return {"top_shops": {"VendorA": 0.87, "VendorB": 0.7}}

        class MockBuyerLLMAnalyzer:
            def analyze_for_buyer(self, products_data: Dict, analysis_type: str = "general") -> str:
                print(f"MockBuyerLLMAnalyzer: analyze_for_buyer type={analysis_type}")
                return f"Analyse Mock IA pour le type '{analysis_type}' basée sur {len(products_data.get('top_k_products', []))} produits. C'est un excellent choix !"
            def get_conversational_chain(self):
                # Un mock simple pour la chaîne de conversation
                class MockChain:
                    def invoke(self, input_data):
                        return f"Réponse mock à : {input_data['question']}"
                return MockChain()

        # Instancier les mocks pour le test
        mock_product_analyzer = MockProductAnalyzer()
        mock_buyer_llm_analyzer = MockBuyerLLMAnalyzer()

        # Passez les instances mockées à ResponsibleBuyerAnalyzer
        analyzer = ResponsibleBuyerAnalyzer(
            "your-groq-key", # Clé API Groq (même si mocké, le constructeur l'attend)
            product_analyzer_instance=mock_product_analyzer,
            buyer_llm_analyzer_instance=mock_buyer_llm_analyzer
        )
        result = await analyzer.perform_responsible_analysis(
            user_id="test_user",
            analysis_params={
                "product_filters": {"available_only": True},
                "analysis_type": "general",
                "criteria": { # Exemple de critères
                    'weights': {'price': 0.5, 'availability': 0.5},
                    'price_preference': 'low'
                },
                "k_products": 5
            }
        )
        print(json.dumps(result, indent=2))
    
    # Configurez le logging pour le script principal
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Lancement du test MCP autonome
    asyncio.run(test_mcp())