import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add utils to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# Try to import with error handling
try:
    from utils.dashboard_utils import DashboardDataManager
    from components.filters import create_advanced_filters
    from components.charts import create_top_products_chart, create_score_distribution_chart, create_ml_feature_importance_chart
    from components.metrics import display_top_products_metrics
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Erreur d'importation: {e}")
    st.info("Certains composants ne sont pas disponibles. Mode dégradé activé.")
    COMPONENTS_AVAILABLE = False

# Import ProductAnalyzer directly
try:
    # Assuming the ProductAnalyzer is in a file called 'product_analyzer.py'
    from paste import ProductAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        sys.path.append(str(Path(__file__).parent))
        from paste import ProductAnalyzer
        ANALYZER_AVAILABLE = True
    except ImportError:
        st.error("❌ ProductAnalyzer non trouvé. Créez un fichier 'product_analyzer.py' avec votre classe.")
        ANALYZER_AVAILABLE = False

def create_simple_filters():
    """Filtres simples MongoDB-compatible"""
    filters = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Filtres de prix")
        min_price = st.number_input("Prix minimum", min_value=0.0, value=0.0, step=0.1)
        max_price = st.number_input("Prix maximum", min_value=0.0, value=1000.0, step=0.1)
        
        if min_price > 0 or max_price < 1000:
            price_filter = {}
            if min_price > 0:
                price_filter["$gte"] = min_price
            if max_price < 1000:
                price_filter["$lte"] = max_price
            filters['price'] = price_filter
    
    with col2:
        st.markdown("#### 📦 Filtres de disponibilité")
        available_only = st.checkbox("Produits disponibles seulement", value=True)
        if available_only:
            filters['available'] = True
        
        min_stock = st.number_input("Stock minimum", min_value=0, value=0, step=1)
        if min_stock > 0:
            filters['stock_quantity'] = {"$gte": min_stock}
    
    # Filtres supplémentaires (remove the nested expander)
    st.markdown("#### 🌍 Filtres géographiques et plateforme")
    col3, col4 = st.columns(2)
    
    with col3:
        region = st.selectbox("Région", ["Toutes", "US", "EU", "CA", "AU"], index=0)
        if region != "Toutes":
            filters['store_region'] = region
    
    with col4:
        platform = st.selectbox("Plateforme", ["Toutes", "shopify", "woocommerce", "magento"], index=0)
        if platform != "Toutes":
            filters['platform'] = platform
    
    return filters

def create_simple_metrics(top_k_df, df_scored):
    """Métriques simples"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de produits", len(top_k_df))
    
    with col2:
        # Utiliser le score disponible
        score_col = 'ml_score' if 'ml_score' in top_k_df.columns else 'synthetic_score'
        avg_score = top_k_df[score_col].mean() if len(top_k_df) > 0 else 0
        st.metric("Score moyen", f"{avg_score:.3f}")
    
    with col3:
        avg_price = top_k_df['price'].mean() if len(top_k_df) > 0 and 'price' in top_k_df.columns else 0
        st.metric("Prix moyen", f"${avg_price:.2f}")
    
    with col4:
        availability_rate = top_k_df['available'].mean() * 100 if len(top_k_df) > 0 and 'available' in top_k_df.columns else 0
        st.metric("Disponibilité", f"{availability_rate:.1f}%")

def create_simple_charts(top_k_df, score_column):
    """Graphiques simples"""
    if len(top_k_df) == 0:
        st.warning("Aucune donnée à afficher")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart des top produits
        display_df = top_k_df.head(10).copy()
        # Tronquer les titres longs
        display_df['title_short'] = display_df['title'].apply(lambda x: x[:30] + "..." if len(str(x)) > 30 else str(x))
        
        fig_bar = px.bar(
            display_df,
            x=score_column,
            y='title_short',
            orientation='h',
            title="Top 10 Produits par Score",
            labels={score_column: 'Score', 'title_short': 'Produit'},
            color=score_column,
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Scatter plot prix vs score
        if 'price' in top_k_df.columns:
            fig_scatter = px.scatter(
                top_k_df,
                x='price',
                y=score_column,
                color='available' if 'available' in top_k_df.columns else None,
                size='stock_quantity' if 'stock_quantity' in top_k_df.columns else None,
                hover_data=['title', 'vendor'] if 'vendor' in top_k_df.columns else ['title'],
                title="Prix vs Score",
                labels={'price': 'Prix ($)', score_column: 'Score', 'available': 'Disponible'}
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Graphique prix vs score non disponible (colonne 'price' manquante)")

def show_top_products_page():
    """Page principale pour l'analyse des top produits"""
    
    st.title("🏆 Analyse des Top Produits")
    st.markdown("Sélection intelligente des meilleurs produits basée sur des critères personnalisables")
    
    # Vérifier la disponibilité de ProductAnalyzer
    if not ANALYZER_AVAILABLE:
        st.error("❌ ProductAnalyzer non disponible. Impossible de continuer.")
        st.info("Assurez-vous que le fichier 'product_analyzer.py' contient votre classe ProductAnalyzer.")
        return
    
    # Section de débogage
    with st.expander("🐛 Informations de débogage", expanded=False):
        try:
            analyzer = ProductAnalyzer()
            st.success("✅ ProductAnalyzer créé avec succès")
            
            # Test de connexion
            if analyzer.client is not None:
                st.success("✅ Connexion MongoDB établie")
                
                # Test d'obtention des données
                test_df = analyzer.get_products_dataframe({})
                st.info(f"📊 Nombre de produits dans la base: {len(test_df)}")
                
                if not test_df.empty:
                    st.write("**Colonnes disponibles:**")
                    st.write(list(test_df.columns))
                    
                    st.write("**Aperçu des données:**")
                    st.write(test_df.head(2))
                else:
                    st.warning("⚠️ Aucun produit trouvé dans la base de données")
            else:
                st.error("❌ Pas de connexion MongoDB")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du test: {e}")
    
    # Configuration dans la sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration de l'analyse")
        
        # Nombre de produits
        k = st.slider("Nombre de produits à afficher", min_value=5, max_value=100, value=20, step=5)
        
        st.markdown("---")
        
        # Poids des critères
        st.markdown("### 📊 Poids des critères")
        
        price_weight = st.slider("Prix", 0.0, 1.0, 0.3, 0.05)
        availability_weight = st.slider("Disponibilité", 0.0, 1.0, 0.25, 0.05)
        stock_weight = st.slider("Stock", 0.0, 1.0, 0.2, 0.05)
        vendor_weight = st.slider("Popularité vendeur", 0.0, 1.0, 0.15, 0.05)
        recency_weight = st.slider("Nouveauté", 0.0, 1.0, 0.1, 0.05)
        
        # Préférence de prix
        price_preference = st.selectbox(
            "Préférence de prix",
            ["low", "high", "discount"],
            format_func=lambda x: {
                "low": "Prix bas",
                "high": "Prix élevé (premium)",
                "discount": "Meilleure remise"
            }[x]
        )
        
        st.markdown("---")
        
        # Méthode ML
        st.markdown("### 🤖 Méthode d'analyse")
        ml_method = st.selectbox(
            "Algorithme ML",
            ["synthetic", "random_forest", "xgboost", "lightgbm"],
            format_func=lambda x: {
                "synthetic": "Score synthétique",
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
                "lightgbm": "LightGBM"
            }[x]
        )
        
        # Clustering
        enable_clustering = st.checkbox("Activer le clustering", value=True)
        if enable_clustering:
            n_clusters = st.slider("Nombre de clusters", 3, 10, 5)
    
    # Filtres avancés - Fixed: Remove nested expander
    st.markdown("### 🔍 Filtres avancés")
    
    # Add a toggle for showing/hiding filters instead of using expander
    show_filters = st.checkbox("Configurer les filtres", value=False)
    
    if show_filters:
        if COMPONENTS_AVAILABLE:
            try:
                filters = create_advanced_filters()
            except:
                st.info("Utilisation des filtres simplifiés")
                filters = create_simple_filters()
        else:
            filters = create_simple_filters()
    else:
        filters = {}
    
    # Construction du dictionnaire de critères
    criteria = {
        'weights': {
            'price': price_weight,
            'availability': availability_weight,
            'stock': stock_weight,
            'vendor_popularity': vendor_weight,
            'recency': recency_weight
        },
        'price_preference': price_preference
    }
    
    # Bouton d'analyse
    if st.button("🚀 Lancer l'analyse", type="primary"):
        
        with st.spinner("Analyse en cours..."):
            try:
                # Initialiser l'analyseur
                analyzer = ProductAnalyzer()
                
                if analyzer.client is None:
                    st.error("❌ Impossible de se connecter à la base de données")
                    return
                
                # 1. Obtenir les données de base avec filtres
                st.info("📊 Récupération des données...")
                df = analyzer.get_products_dataframe(filters)
                
                if df.empty:
                    st.warning("⚠️ Aucun produit trouvé avec ces critères de filtrage")
                    return
                
                st.success(f"✅ {len(df)} produits récupérés")
                
                # 2. Calculer le score synthétique
                st.info("🧮 Calcul des scores...")
                df_scored = analyzer.calculate_synthetic_score(df, criteria)
                
                # 3. Appliquer le ML si demandé
                score_column = 'synthetic_score'
                if ml_method != 'synthetic':
                    st.info(f"🤖 Application de l'algorithme {ml_method}...")
                    df_scored = analyzer.apply_ml_scoring(df_scored, ml_method)
                    score_column = 'ml_score'
                
                # 4. Appliquer le clustering si demandé
                if enable_clustering:
                    st.info("🎯 Application du clustering...")
                    df_scored = analyzer.cluster_products(df_scored, n_clusters)
                
                # 5. Obtenir les top K produits
                st.info(f"🏆 Sélection des top {k} produits...")
                top_k_df = analyzer.get_top_k_products(df_scored, k, score_column)
                
                # Afficher les résultats
                display_results(top_k_df, df_scored, score_column, ml_method, enable_clustering, analyzer)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                st.exception(e)  # Afficher la stack trace complète pour le debug

def display_results(top_k_df, df_scored, score_column, ml_method, enable_clustering, analyzer):
    """Affiche les résultats de l'analyse"""
    
    # Métriques
    st.markdown("## 📈 Métriques principales")
    if COMPONENTS_AVAILABLE:
        try:
            display_top_products_metrics(top_k_df, df_scored)
        except:
            create_simple_metrics(top_k_df, df_scored)
    else:
        create_simple_metrics(top_k_df, df_scored)
    
    # Tableau des top produits
    st.markdown("## 🏆 Top Produits")
    
    # Préparer le DataFrame d'affichage
    display_df = top_k_df.copy()
    display_df['Rang'] = range(1, len(display_df) + 1)
    display_df['Score'] = display_df[score_column].round(4)
    
    if 'price' in display_df.columns:
        display_df['Prix'] = display_df['price'].apply(lambda x: f"${x:.2f}")
    else:
        display_df['Prix'] = "N/A"
    
    if 'available' in display_df.columns:
        display_df['Disponible'] = display_df['available'].apply(lambda x: "✅" if x else "❌")
    else:
        display_df['Disponible'] = "N/A"
    
    if 'stock_quantity' in display_df.columns:
        display_df['Stock'] = display_df['stock_quantity'].fillna(0).astype(int)
    else:
        display_df['Stock'] = 0
    
    # Colonnes à afficher
    columns_to_show = ['Rang', 'title', 'Prix', 'Disponible', 'Stock', 'Score']
    
    if 'vendor' in display_df.columns:
        columns_to_show.insert(-2, 'vendor')
    
    if 'store_domain' in display_df.columns:
        columns_to_show.append('store_domain')
    
    if enable_clustering and 'cluster' in display_df.columns:
        display_df['Cluster'] = display_df['cluster']
        columns_to_show.append('Cluster')
    
    # Renommer les colonnes
    column_names = {
        'title': 'Produit',
        'vendor': 'Vendeur',
        'store_domain': 'Boutique'
    }
    
    st.dataframe(
        display_df[columns_to_show].rename(columns=column_names),
        use_container_width=True
    )
    
    # Section des graphiques
    st.markdown("## 📊 Visualisations")
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        if COMPONENTS_AVAILABLE:
            try:
                fig_dist = create_score_distribution_chart(df_scored, score_column)
                st.plotly_chart(fig_dist, use_container_width=True)
            except:
                # Graphique de distribution simple
                fig_hist = px.histogram(
                    df_scored, 
                    x=score_column, 
                    title="Distribution des Scores",
                    nbins=30
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            # Graphique de distribution simple
            fig_hist = px.histogram(
                df_scored, 
                x=score_column, 
                title="Distribution des Scores",
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        create_simple_charts(top_k_df, score_column)
    
    # Clustering visualization si activé
    if enable_clustering and 'cluster' in df_scored.columns:
        st.markdown("### 🎯 Analyse des Clusters")
        
        if 'price' in df_scored.columns:
            fig_cluster = px.scatter(
                df_scored,
                x='price',
                y=score_column,
                color='cluster',
                title="Clusters de Produits (Prix vs Score)",
                labels={'price': 'Prix ($)', score_column: 'Score'},
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Stats par cluster
        cluster_stats = df_scored.groupby('cluster').agg({
            score_column: ['mean', 'count'],
            'price': 'mean' if 'price' in df_scored.columns else lambda x: 0,
            'available': 'sum' if 'available' in df_scored.columns else lambda x: 0
        }).round(2)
        
        st.write("**Statistiques par cluster:**")
        st.dataframe(cluster_stats)
    
    # Analyses supplémentaires
    st.markdown("## 🌍 Analyses supplémentaires")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Analyse géographique
        geo_analysis = analyzer.analyze_by_geography(df_scored)
        if geo_analysis:
            st.markdown("### 🗺️ Analyse géographique")
            st.json(geo_analysis)
    
    with col2:
        # Analyse des boutiques
        shop_analysis = analyzer.analyze_shops_ranking(df_scored)
        if shop_analysis:
            st.markdown("### 🏪 Classement des boutiques")
            if 'top_shops' in shop_analysis:
                top_shops_df = pd.DataFrame(list(shop_analysis['top_shops'].items()), 
                                          columns=['Boutique', 'Score Moyen'])
                st.dataframe(top_shops_df)
    
    # Export des données
    st.markdown("## 💾 Export des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export CSV
        csv = top_k_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"top_{len(top_k_df)}_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export JSON
        json_data = top_k_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Télécharger JSON",
            data=json_data,
            file_name=f"top_{len(top_k_df)}_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def show_page(analyzer=None):
    """Page principale pour l'analyse des top produits"""
    show_top_products_page()  # Call your existing function

# Point d'entrée pour les tests
if __name__ == "__main__":
    show_top_products_page()