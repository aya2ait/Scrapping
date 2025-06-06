import os
import json
import requests
from typing import Dict, List, Any, Optional
from groq import Groq
import pandas as pd
from datetime import datetime
import logging

# Configuration Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')

class LLMEnhancer:
    def __init__(self, groq_api_key: str = GROQ_API_KEY):
        """Initialize LLM Enhancer with Groq"""
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama3-8b-8192"  # Ou llama3-70b-8192 pour plus de puissance
        self.logger = logging.getLogger(__name__)
    
    def generate_product_summary(self, top_k_products: List[Dict]) -> str:
        """Génère un résumé intelligent des top-K produits"""
        try:
            # Préparer le contexte pour le LLM
            products_context = []
            for i, product in enumerate(top_k_products[:10], 1):  # Limiter à 10 pour éviter les tokens
                context = f"""
                Produit #{i}:
                - Titre: {product.get('title', 'N/A')}
                - Vendeur: {product.get('vendor', 'N/A')}
                - Prix: {product.get('price', 0)}€
                - Score: {product.get('synthetic_score', 0):.3f}
                - Disponible: {'Oui' if product.get('available') else 'Non'}
                - Plateforme: {product.get('platform', 'N/A')}
                - Région: {product.get('store_region', 'N/A')}
                """
                products_context.append(context)
            
            context_text = "\n".join(products_context)
            
            prompt = f"""
            En tant qu'analyste e-commerce expert, analysez ces {len(top_k_products)} produits top performers et générez un résumé stratégique.

            DONNÉES DES PRODUITS:
            {context_text}

            ANALYSEZ ET FOURNISSEZ:
            1. **Tendances principales** : Quels patterns émergent des top produits ?
            2. **Opportunités business** : Quelles opportunités ces produits révèlent-ils ?
            3. **Recommandations stratégiques** : Que devrait faire un e-commerçant ?
            4. **Insights concurrentiels** : Quels avantages concurrentiels identifier ?
            5. **Points d'attention** : Quels risques ou défis anticiper ?

            Répondez de manière concise et actionnable en français, maximum 300 mots.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en analyse e-commerce et business intelligence."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Erreur génération résumé: {e}")
            return "Impossible de générer le résumé automatiquement."
    
    def analyze_market_opportunities(self, products_data: Dict, geographical_analysis: Dict = None) -> str:
        """Analyse les opportunités de marché basées sur les données"""
        try:
            # Extraire les statistiques clés
            stats = products_data.get('statistics', {})
            top_products = products_data.get('top_k_products', [])
            
            context = f"""
            STATISTIQUES MARCHÉ:
            - Nombre total de produits analysés: {products_data.get('total_products', 0)}
            - Score moyen: {stats.get('average_score', 0):.3f}
            - Fourchette de prix: {stats.get('price_range', {}).get('min', 0)}€ - {stats.get('price_range', {}).get('max', 0)}€
            - Prix moyen: {stats.get('price_range', {}).get('avg', 0)}€
            - Taux de disponibilité: {stats.get('availability_rate', 0)*100:.1f}%
            
            TOP 5 PRODUITS:
            """
            
            for i, product in enumerate(top_products[:5], 1):
                context += f"\n{i}. {product.get('title', 'N/A')} - {product.get('price', 0)}€ (Score: {product.get('synthetic_score', 0):.3f})"
            
            if geographical_analysis:
                context += f"\n\nANALYSE GÉOGRAPHIQUE DISPONIBLE: {len(geographical_analysis)} régions analysées"
            
            prompt = f"""
            En tant que consultant en stratégie e-commerce, analysez ces données de marché et identifiez les opportunités business.

            DONNÉES:
            {context}

            FOURNISSEZ UNE ANALYSE STRUCTURÉE:
            
            🎯 **OPPORTUNITÉS IDENTIFIÉES**
            - 3 opportunités principales de croissance
            
            📊 **SEGMENTS PORTEURS**
            - Segments de prix les plus attractifs
            - Catégories sous-exploitées
            
            🌍 **EXPANSION GÉOGRAPHIQUE**
            - Recommandations par région
            
            ⚡ **ACTIONS PRIORITAIRES**
            - 3 actions concrètes à mettre en œuvre
            
            Soyez précis, chiffré et actionnable. Maximum 400 mots.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Vous êtes un consultant senior spécialisé en stratégie e-commerce et expansion de marché."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.6,
                max_tokens=600
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Erreur analyse opportunités: {e}")
            return "Impossible d'analyser les opportunités automatiquement."
    
    def generate_competitive_insights(self, shops_analysis: Dict) -> str:
        """Génère des insights concurrentiels basés sur l'analyse des boutiques"""
        try:
            if not shops_analysis:
                return "Aucune donnée de boutiques disponible pour l'analyse concurrentielle."
            
            top_shops = shops_analysis.get('top_shops', {})
            flagship_products = shops_analysis.get('flagship_products', {})
            
            context = f"""
            ANALYSE CONCURRENTIELLE DES BOUTIQUES:
            
            TOP BOUTIQUES (par score moyen):
            """
            
            for shop, score in list(top_shops.items())[:10]:
                flagship = flagship_products.get(shop, {})
                context += f"\n- {shop}: Score {score:.3f}"
                if flagship:
                    context += f" | Produit phare: {flagship.get('title', 'N/A')} ({flagship.get('price', 0)}€)"
            
            prompt = f"""
            En tant qu'analyste concurrentiel e-commerce, analysez le paysage concurrentiel et fournissez des insights stratégiques.

            {context}

            FOURNISSEZ:
            
            🏆 **LEADERS DU MARCHÉ**
            - Qui domine et pourquoi ?
            - Leurs stratégies gagnantes
            
            🎯 **POSITIONNEMENT CONCURRENTIEL**
            - Espaces de marché peu contestés
            - Différenciations possibles
            
            📈 **BENCHMARKS CLÉS**
            - KPIs à surveiller chez les concurrents
            - Gaps d'opportunités
            
            ⚡ **RECOMMANDATIONS TACTIQUES**
            - Comment se positionner vs la concurrence
            - Avantages concurrentiels à développer
            
            Maximum 350 mots, soyez stratégique et actionnable.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en intelligence concurrentielle pour l'e-commerce."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Erreur insights concurrentiels: {e}")
            return "Impossible de générer les insights concurrentiels automatiquement."
    
    def generate_pricing_strategy(self, products_data: Dict) -> str:
        """Génère des recommandations de stratégie de prix"""
        try:
            stats = products_data.get('statistics', {})
            top_products = products_data.get('top_k_products', [])
            
            # Analyser la distribution des prix des top produits
            prices = [p.get('price', 0) for p in top_products if p.get('price', 0) > 0]
            
            if not prices:
                return "Données de prix insuffisantes pour l'analyse."
            
            price_analysis = {
                'min_price': min(prices),
                'max_price': max(prices),
                'avg_price': sum(prices) / len(prices),
                'median_price': sorted(prices)[len(prices)//2]
            }
            
            prompt = f"""
            En tant que pricing strategist e-commerce, analysez cette data de prix et recommandez une stratégie.

            ANALYSE DES PRIX TOP PRODUITS:
            - Prix minimum: {price_analysis['min_price']:.2f}€
            - Prix maximum: {price_analysis['max_price']:.2f}€
            - Prix moyen: {price_analysis['avg_price']:.2f}€
            - Prix médian: {price_analysis['median_price']:.2f}€
            
            MARCHÉ GLOBAL:
            - Prix moyen marché: {stats.get('price_range', {}).get('avg', 0):.2f}€
            - Fourchette totale: {stats.get('price_range', {}).get('min', 0):.2f}€ - {stats.get('price_range', {}).get('max', 0):.2f}€

            RECOMMANDATIONS PRICING:
            
            💰 **ZONES DE PRIX OPTIMALES**
            - Sweet spots identifiés
            
            📊 **STRATÉGIES RECOMMANDÉES**
            - Pénétration vs écrémage
            - Positionnement prix/valeur
            
            🎯 **TACTICAL PRICING**
            - Prix psychologiques
            - Bundles et promotions
            
            ⚡ **IMPLÉMENTATION**
            - Étapes concrètes de mise en œuvre
            
            Maximum 300 mots, focus actionnable.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en stratégie de prix pour l'e-commerce."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.4,
                max_tokens=450
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Erreur stratégie pricing: {e}")
            return "Impossible de générer la stratégie de prix automatiquement."
    
    def create_executive_summary(self, complete_analysis: Dict) -> str:
        """Crée un résumé exécutif complet"""
        try:
            prompt = f"""
            En tant que directeur e-commerce, créez un résumé exécutif stratégique basé sur cette analyse complète.

            DONNÉES CLÉS:
            - {complete_analysis.get('total_products', 0)} produits analysés
            - Top {complete_analysis.get('k', 0)} produits sélectionnés
            - Score moyen: {complete_analysis.get('statistics', {}).get('average_score', 0):.3f}
            - Méthode: {complete_analysis.get('score_method', 'synthetic')}

            CRÉEZ UN EXECUTIVE SUMMARY:
            
            📋 **RÉSUMÉ EXÉCUTIF**
            - 2-3 points clés en une phrase chacun
            
            📊 **CHIFFRES CLÉS**
            - KPIs les plus importants
            
            🎯 **RECOMMANDATIONS PRIORITAIRES**
            - 3 actions à impact immédiat
            
            ⏱️ **TIMELINE RECOMMANDÉE**
            - Court terme (1-3 mois)
            - Moyen terme (3-6 mois)
            
            🎯 **ROI ATTENDU**
            - Estimation d'impact business
            
            Format: Executive summary professionnel, 250 mots maximum.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Vous êtes un directeur e-commerce expérimenté rédigeant pour le C-level."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=350
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Erreur résumé exécutif: {e}")
            return "Impossible de générer le résumé exécutif automatiquement."

# Extension de votre API Flask existante
def enhance_api_with_llm():
    """
    Fonction pour intégrer l'enhancer LLM à votre API existante
    """
    
    # Ajouter cette route à votre app Flask existante
    @app.route('/api/llm-enhanced-analysis', methods=['POST'])
    def get_llm_enhanced_analysis():
        """
        Endpoint enrichi avec analyse LLM
        Utilise les mêmes paramètres que /api/top-k-products mais ajoute l'analyse LLM
        """
        try:
            # Récupérer l'analyse standard (réutiliser la logique existante)
            standard_response = get_top_k_products()
            
            if standard_response[1] != 200:  # Si erreur dans l'analyse standard
                return standard_response
            
            analysis_data = standard_response[0].get_json()
            
            # Initialiser le LLM enhancer
            llm_enhancer = LLMEnhancer()
            
            # Générer les enrichissements LLM
            enhancements = {
                'product_summary': llm_enhancer.generate_product_summary(
                    analysis_data.get('top_k_products', [])
                ),
                'market_opportunities': llm_enhancer.analyze_market_opportunities(
                    analysis_data,
                    analysis_data.get('geographical_analysis')
                ),
                'executive_summary': llm_enhancer.create_executive_summary(analysis_data)
            }
            
            # Ajouter l'analyse concurrentielle si disponible
            if analysis_data.get('shops_analysis'):
                enhancements['competitive_insights'] = llm_enhancer.generate_competitive_insights(
                    analysis_data['shops_analysis']
                )
            
            # Ajouter la stratégie de prix
            enhancements['pricing_strategy'] = llm_enhancer.generate_pricing_strategy(analysis_data)
            
            # Combiner les données
            analysis_data['llm_insights'] = enhancements
            analysis_data['enhanced_at'] = datetime.utcnow().isoformat()
            
            return jsonify(analysis_data)
            
        except Exception as e:
            logger.error(f"Erreur dans l'analyse LLM enrichie: {e}")
            return jsonify({'error': str(e)}), 500

# Exemple d'utilisation standalone
if __name__ == "__main__":
    # Test de l'enhancer LLM
    enhancer = LLMEnhancer()
    
    # Exemple de données de test
    sample_products = [
        {
            'title': 'iPhone 15 Pro',
            'vendor': 'Apple Store',
            'price': 1199,
            'synthetic_score': 0.923,
            'available': True,
            'platform': 'shopify',
            'store_region': 'US'
        },
        {
            'title': 'Samsung Galaxy S24',
            'vendor': 'Samsung',
            'price': 899,
            'synthetic_score': 0.887,
            'available': True,
            'platform': 'woocommerce',
            'store_region': 'EU'
        }
    ]
    
    # Test des différentes analyses
    print("🔍 Résumé des produits:")
    print(enhancer.generate_product_summary(sample_products))
    
    print("\n📊 Analyse des opportunités:")
    sample_data = {
        'total_products': 1000,
        'top_k_products': sample_products,
        'statistics': {
            'average_score': 0.756,
            'price_range': {'min': 10, 'max': 2000, 'avg': 245}
        }
    }
    print(enhancer.analyze_market_opportunities(sample_data))