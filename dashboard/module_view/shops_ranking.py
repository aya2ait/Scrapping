import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import sys
import os

# Ajouter le chemin parent pour importer la classe ProductAnalyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Importer la classe ProductAnalyzer depuis le fichier principal
    from paste import ProductAnalyzer
except ImportError:
    st.error("❌ Erreur: Impossible d'importer ProductAnalyzer. Vérifiez le chemin du fichier.")
    st.stop()

def load_data():
    """Charger les données depuis MongoDB"""
    try:
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = ProductAnalyzer()
        
        analyzer = st.session_state.analyzer
        
        # Vérifier la connexion
        if not analyzer.client:
            st.error("❌ Connexion à MongoDB échouée")
            return pd.DataFrame()
        
        # Charger les données
        df = analyzer.get_products_dataframe()
        return df
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

def calculate_shop_metrics(df, criteria):
    """Calculer les métriques des boutiques"""
    try:
        analyzer = st.session_state.analyzer
        
        # Calculer le score synthétique
        df_scored = analyzer.calculate_synthetic_score(df, criteria)
        
        # Analyser le classement des boutiques
        shops_analysis = analyzer.analyze_shops_ranking(df_scored)
        
        return df_scored, shops_analysis
    
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul des métriques: {str(e)}")
        return df, {}

def create_shops_overview_chart(df_scored):
    """Créer un graphique d'aperçu des boutiques"""
    if 'store_domain' not in df_scored.columns:
        st.warning("⚠️ Pas de données de boutiques disponibles")
        return None
    
    # Statistiques par boutique
    shop_stats = df_scored.groupby('store_domain').agg({
        'synthetic_score': ['mean', 'count'],
        'price': 'mean',
        'available': 'sum'
    }).round(2)
    
    shop_stats.columns = ['avg_score', 'product_count', 'avg_price', 'available_products']
    shop_stats = shop_stats.reset_index()
    
    # Top 15 boutiques par score moyen
    top_shops = shop_stats.nlargest(15, 'avg_score')
    
    # Graphique en barres
    fig = px.bar(
        top_shops,
        x='store_domain',
        y='avg_score',
        title='🏪 Top 15 Boutiques par Score Moyen',
        labels={
            'store_domain': 'Boutique',
            'avg_score': 'Score Moyen'
        },
        color='avg_score',
        color_continuous_scale='viridis',
        text='avg_score'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    return fig

def create_shops_bubble_chart(df_scored):
    """Créer un graphique en bulles pour les boutiques"""
    if 'store_domain' not in df_scored.columns:
        return None
    
    # Statistiques par boutique (sans stock_quantity)
    shop_stats = df_scored.groupby('store_domain').agg({
        'synthetic_score': 'mean',
        'price': 'mean',
        'available': 'sum'
    }).round(2)
    
    shop_stats.columns = ['avg_score', 'avg_price', 'available_products']
    shop_stats = shop_stats.reset_index()
    
    # Ajouter une colonne pour la taille des bulles basée sur le nombre de produits
    product_counts = df_scored.groupby('store_domain').size().reset_index(name='product_count')
    shop_stats = shop_stats.merge(product_counts, on='store_domain')
    
    # Limiter aux 20 meilleures boutiques
    top_shops = shop_stats.nlargest(20, 'avg_score')
    
    fig = px.scatter(
        top_shops,
        x='avg_price',
        y='avg_score',
        size='product_count',
        color='available_products',
        hover_name='store_domain',
        title='💹 Performance des Boutiques (Prix vs Score vs Nombre de Produits)',
        labels={
            'avg_price': 'Prix Moyen (€)',
            'avg_score': 'Score Moyen',
            'product_count': 'Nombre de Produits',
            'available_products': 'Produits Disponibles'
        },
        color_continuous_scale='plasma',
        size_max=60
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_flagship_products_chart(shops_analysis):
    """Créer un graphique des produits phares"""
    if 'flagship_products' not in shops_analysis:
        st.warning("⚠️ Aucun produit phare disponible dans l'analyse")
        return None
    
    flagship_data = []
    
    # Debug: afficher la structure des données
    st.write("Debug - Structure flagship_products:", type(shops_analysis['flagship_products']))
    
    try:
        for shop, product in shops_analysis['flagship_products'].items():
            # Vérifier le type de product
            if isinstance(product, dict):
                flagship_data.append({
                    'shop': shop,
                    'product_title': str(product.get('title', 'Unknown'))[:30] + '...',
                    'price': float(product.get('price', 0)),
                    'score': float(product.get('score', 0)),
                    'available': bool(product.get('available', False))
                })
            elif isinstance(product, (int, float)):
                # Si product est juste un score numérique
                flagship_data.append({
                    'shop': shop,
                    'product_title': 'Score numérique',
                    'price': 0.0,
                    'score': float(product),
                    'available': True
                })
            else:
                st.warning(f"⚠️ Type de données inattendu pour {shop}: {type(product)}")
                continue
    
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement des produits phares: {str(e)}")
        st.write("Debug - Contenu flagship_products:", shops_analysis['flagship_products'])
        return None
    
    if not flagship_data:
        st.warning("⚠️ Aucune donnée de produit phare valide trouvée")
        return None
    
    df_flagship = pd.DataFrame(flagship_data)
    
    # Top 15 produits phares
    top_flagship = df_flagship.nlargest(15, 'score')
    
    # Graphique horizontal
    fig = px.bar(
        top_flagship,
        x='score',
        y='shop',
        title='🏆 Produits Phares par Boutique (Top 15)',
        labels={
            'score': 'Score du Produit Phare',
            'shop': 'Boutique'
        },
        color='price',
        color_continuous_scale='turbo',
        orientation='h',
        text='score'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_shops_comparison_radar(df_scored, selected_shops):
    """Créer un graphique radar pour comparer les boutiques (sans stock)"""
    if not selected_shops or 'store_domain' not in df_scored.columns:
        return None
    
    radar_data = []
    
    for shop in selected_shops:
        shop_data = df_scored[df_scored['store_domain'] == shop]
        if not shop_data.empty:
            metrics = {
                'Boutique': shop,
                'Score Moyen': shop_data['synthetic_score'].mean(),
                'Prix Moyen': min(shop_data['price'].mean() / 50, 100),  # Normalisation ajustée
                'Taux Disponibilité': shop_data['available'].mean() * 100,
                'Diversité Prix': min((shop_data['price'].std() or 0) / 10, 100),  # Écart-type des prix
                'Nb Produits': min(len(shop_data) / 5, 100)  # Normalisation
            }
            radar_data.append(metrics)
    
    if not radar_data:
        return None
    
    df_radar = pd.DataFrame(radar_data)
    
    # Créer le graphique radar
    fig = go.Figure()
    
    categories = ['Score Moyen', 'Prix Moyen', 'Taux Disponibilité', 'Diversité Prix', 'Nb Produits']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, row in df_radar.iterrows():
        values = [row[cat] for cat in categories]
        values.append(values[0])  # Fermer le radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Boutique'],
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="📊 Comparaison Multi-Critères des Boutiques",
        height=500
    )
    
    return fig

def create_shops_performance_timeline(df_scored):
    """Créer une timeline de performance des boutiques"""
    if 'created_at' not in df_scored.columns or 'store_domain' not in df_scored.columns:
        return None
    
    # Convertir les dates
    df_scored['created_at'] = pd.to_datetime(df_scored['created_at'])
    df_scored['month_year'] = df_scored['created_at'].dt.to_period('M').astype(str)
    
    # Performance mensuelle par boutique (top 5)
    top_5_shops = df_scored.groupby('store_domain')['synthetic_score'].mean().nlargest(5).index
    
    monthly_performance = df_scored[df_scored['store_domain'].isin(top_5_shops)].groupby(
        ['month_year', 'store_domain']
    )['synthetic_score'].mean().reset_index()
    
    fig = px.line(
        monthly_performance,
        x='month_year',
        y='synthetic_score',
        color='store_domain',
        title='📈 Évolution de Performance des Top 5 Boutiques',
        labels={
            'month_year': 'Mois',
            'synthetic_score': 'Score Moyen',
            'store_domain': 'Boutique'
        },
        markers=True
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def display_shop_detailed_analysis(df_scored, selected_shop):
    """Afficher l'analyse détaillée d'une boutique (sans stock_quantity)"""
    shop_data = df_scored[df_scored['store_domain'] == selected_shop]
    
    if shop_data.empty:
        st.warning(f"⚠️ Aucune donnée trouvée pour la boutique: {selected_shop}")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Produits Total",
            len(shop_data)
        )
    
    with col2:
        st.metric(
            "✅ Produits Disponibles",
            int(shop_data['available'].sum()),
            f"{shop_data['available'].mean()*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "💰 Prix Moyen",
            f"{shop_data['price'].mean():.2f}€",
            f"Médiane: {shop_data['price'].median():.2f}€"
        )
    
    with col4:
        st.metric(
            "⭐ Score Moyen",
            f"{shop_data['synthetic_score'].mean():.3f}",
            f"Max: {shop_data['synthetic_score'].max():.3f}"
        )
    
    # Top produits de la boutique (sans stock_quantity)
    st.subheader(f"🏆 Top 10 Produits - {selected_shop}")
    
    # Colonnes disponibles à afficher
    columns_to_show = ['title', 'vendor', 'price', 'synthetic_score', 'available']
    existing_columns = [col for col in columns_to_show if col in shop_data.columns]
    
    top_products = shop_data.nlargest(10, 'synthetic_score')[existing_columns].round(3)
    
    column_config = {
        'title': 'Produit',
        'vendor': 'Vendeur',
        'price': st.column_config.NumberColumn('Prix (€)', format="%.2f"),
        'synthetic_score': st.column_config.NumberColumn('Score', format="%.3f"),
        'available': 'Disponible'
    }
    
    st.dataframe(
        top_products,
        use_container_width=True,
        column_config={k: v for k, v in column_config.items() if k in existing_columns}
    )
    
    # Distribution des prix
    fig_price = px.histogram(
        shop_data,
        x='price',
        title=f'Distribution des Prix - {selected_shop}',
        nbins=20,
        labels={'price': 'Prix (€)', 'count': 'Nombre de Produits'}
    )
    fig_price.update_layout(height=300)
    st.plotly_chart(fig_price, use_container_width=True)

def safe_create_flagship_table(shops_analysis):
    """Créer le tableau des produits phares de manière sécurisée"""
    if 'flagship_products' not in shops_analysis:
        return pd.DataFrame()
    
    flagship_list = []
    
    try:
        for shop, product in shops_analysis['flagship_products'].items():
            product_data = {
                'Boutique': str(shop),
                'Produit': 'Non spécifié',
                'Prix (€)': 0.0,
                'Score': 0.0,
                'Vendeur': 'Non spécifié',
                'Disponible': '❌'
            }
            
            if isinstance(product, dict):
                product_data.update({
                    'Produit': str(product.get('title', 'Non spécifié')),
                    'Prix (€)': float(product.get('price', 0)),
                    'Score': float(product.get('score', 0)),
                    'Vendeur': str(product.get('vendor', 'Non spécifié')),
                    'Disponible': '✅' if product.get('available', False) else '❌'
                })
            elif isinstance(product, (int, float)):
                product_data.update({
                    'Produit': 'Score numérique',
                    'Score': float(product)
                })
            
            flagship_list.append(product_data)
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la création du tableau: {str(e)}")
        return pd.DataFrame()
    
    if flagship_list:
        df_flagship = pd.DataFrame(flagship_list)
        return df_flagship.sort_values('Score', ascending=False)
    
    return pd.DataFrame()

def main():
    st.title("🏪 Classement et Analyse des Boutiques")
    st.markdown("---")
    
    # Chargement des données
    with st.spinner("📊 Chargement des données..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Aucune donnée disponible")
        st.stop()
    
    # Sidebar pour les critères
    st.sidebar.header("⚙️ Paramètres d'Analyse")
    
    # Critères de scoring (stock_weight retiré)
    st.sidebar.subheader("🎯 Critères de Scoring")
    
    price_weight = st.sidebar.slider("💰 Poids Prix", 0.0, 1.0, 0.35, 0.05)
    availability_weight = st.sidebar.slider("✅ Poids Disponibilité", 0.0, 1.0, 0.3, 0.05)
    vendor_weight = st.sidebar.slider("🏭 Poids Popularité Vendeur", 0.0, 1.0, 0.2, 0.05)
    recency_weight = st.sidebar.slider("🆕 Poids Récence", 0.0, 1.0, 0.15, 0.05)
    
    price_preference = st.sidebar.selectbox(
        "💲 Préférence Prix",
        ["low", "high", "discount"],
        help="low: prix bas, high: prix élevés, discount: remises importantes"
    )
    
    # Filtres
    st.sidebar.subheader("🔍 Filtres")
    
    available_only = st.sidebar.checkbox("✅ Produits disponibles uniquement", value=True)
    
    if 'price' in df.columns:
        price_range = st.sidebar.slider(
            "💰 Fourchette de Prix (€)",
            float(df['price'].min()),
            float(df['price'].max()),
            (float(df['price'].min()), float(df['price'].max())),
            step=1.0
        )
    else:
        price_range = (0, 1000)
    
    # Régions disponibles
    regions = []
    if 'store_region' in df.columns:
        regions = df['store_region'].unique().tolist()
        selected_regions = st.sidebar.multiselect("🌍 Régions", regions, default=regions[:3])
    else:
        selected_regions = []
    
    # Application des filtres
    df_filtered = df.copy()
    
    if available_only and 'available' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['available'] == True]
    
    if 'price' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['price'] >= price_range[0]) & 
            (df_filtered['price'] <= price_range[1])
        ]
    
    if selected_regions and 'store_region' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['store_region'].isin(selected_regions)]
    
    # Préparation des critères (stock retiré)
    criteria = {
        'weights': {
            'price': price_weight,
            'availability': availability_weight,
            'vendor_popularity': vendor_weight,
            'recency': recency_weight
        },
        'price_preference': price_preference
    }
    
    # Calcul des métriques
    with st.spinner("🔄 Calcul des scores et analyses..."):
        df_scored, shops_analysis = calculate_shop_metrics(df_filtered, criteria)
    
    # Métriques globales
    st.subheader("📊 Aperçu Général")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🏪 Boutiques Total",
            df_scored['store_domain'].nunique() if 'store_domain' in df_scored.columns else 0
        )
    
    with col2:
        st.metric(
            "📦 Produits Analysés",
            len(df_scored)
        )
    
    with col3:
        avg_score = df_scored['synthetic_score'].mean()
        st.metric(
            "⭐ Score Moyen Global",
            f"{avg_score:.3f}"
        )
    
    with col4:
        if 'available' in df_scored.columns:
            availability_rate = df_scored['available'].mean() * 100
            st.metric(
                "✅ Taux Disponibilité",
                f"{availability_rate:.1f}%"
            )
    
    with col5:
        if 'price' in df_scored.columns:
            avg_price = df_scored['price'].mean()
            st.metric(
                "💰 Prix Moyen",
                f"{avg_price:.2f}€"
            )
    
    st.markdown("---")
    
    # Onglets pour différentes vues
    tab1, tab2, tab3, tab4 = st.tabs(["🏪 Classement", "💹 Performance", "🏆 Produits Phares", "🔍 Analyse Détaillée"])
    
    with tab1:
        st.subheader("🏪 Classement des Boutiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique aperçu des boutiques
            fig_overview = create_shops_overview_chart(df_scored)
            if fig_overview:
                st.plotly_chart(fig_overview, use_container_width=True)
        
        with col2:
            # Graphique en bulles
            fig_bubble = create_shops_bubble_chart(df_scored)
            if fig_bubble:
                st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Tableau détaillé
        if 'shop_statistics' in shops_analysis:
            st.subheader("📋 Statistiques Détaillées des Boutiques")
            
            shop_stats_df = pd.DataFrame(shops_analysis['shop_statistics']).T
            shop_stats_df = shop_stats_df.round(3)
            
            # Réorganiser les colonnes si elles existent
            if not shop_stats_df.empty:
                st.dataframe(
                    shop_stats_df.head(20),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader("💹 Analyse de Performance")
        
        # Timeline de performance
        fig_timeline = create_shops_performance_timeline(df_scored)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Comparaison radar
        if 'store_domain' in df_scored.columns:
            available_shops = df_scored['store_domain'].unique().tolist()
            selected_shops_radar = st.multiselect(
                "🎯 Sélectionner des boutiques à comparer (max 5)",
                available_shops,
                default=available_shops[:3],
                max_selections=5
            )
            
            if selected_shops_radar:
                fig_radar = create_shops_comparison_radar(df_scored, selected_shops_radar)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.subheader("🏆 Produits Phares par Boutique")
        
        # Graphique des produits phares
        fig_flagship = create_flagship_products_chart(shops_analysis)
        if fig_flagship:
            st.plotly_chart(fig_flagship, use_container_width=True)
        
        # Tableau des produits phares avec gestion d'erreur améliorée
        st.subheader("📋 Liste des Produits Phares")
        df_flagship = safe_create_flagship_table(shops_analysis)
        
        if not df_flagship.empty:
            st.dataframe(
                df_flagship,
                use_container_width=True,
                column_config={
                    'Prix (€)': st.column_config.NumberColumn(format="%.2f"),
                    'Score': st.column_config.NumberColumn(format="%.3f")
                }
            )
        else:
            st.info("ℹ️ Aucun produit phare disponible ou données invalides")
    
    with tab4:
        st.subheader("🔍 Analyse Détaillée par Boutique")
        
        if 'store_domain' in df_scored.columns:
            available_shops = sorted(df_scored['store_domain'].unique().tolist())
            selected_shop = st.selectbox(
                "🏪 Sélectionner une boutique pour l'analyse détaillée",
                available_shops
            )
            
            if selected_shop:
                display_shop_detailed_analysis(df_scored, selected_shop)
    
    # Informations de debug
    with st.expander("🔧 Informations Techniques"):
        st.write("**Critères utilisés:**")
        st.json(criteria)
        st.write(f"**Nombre de produits après filtrage:** {len(df_scored)}")
        st.write(f"**Nombre de boutiques:** {df_scored['store_domain'].nunique() if 'store_domain' in df_scored.columns else 0}")
        st.write(f"**Colonnes disponibles:** {list(df_scored.columns)}")
        
        # Debug pour flagship_products
        if 'flagship_products' in shops_analysis:
            st.write("**Structure flagship_products:**")
            st.write(f"Type: {type(shops_analysis['flagship_products'])}")
            if shops_analysis['flagship_products']:
                first_key = list(shops_analysis['flagship_products'].keys())[0]
                st.write(f"Premier élément - Clé: {first_key}, Type valeur: {type(shops_analysis['flagship_products'][first_key])}")

def show_page(analyzer):
    """Fonction requise pour l'intégration avec l'application principale"""
    # SUPPRIMÉ: st.set_page_config() - sera défini dans l'application principale
    
    # Stocker l'analyzer dans session_state
    st.session_state.analyzer = analyzer
    
    # Appeler la fonction main
    main()

if __name__ == "__main__":
    # Pour l'exécution directe du script (mode développement)
    # Cette configuration ne sera appliquée que si le script est exécuté directement
    st.set_page_config(page_title="🏪 Classement des Boutiques", layout="wide")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ProductAnalyzer()
    main()