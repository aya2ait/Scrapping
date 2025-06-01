import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import from the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paste import ProductAnalyzer

# Import custom components
from components.charts import create_map_chart, create_regional_comparison_chart, create_price_heatmap
from components.filters import create_geography_filters
from components.metrics import display_geography_metrics
from utils.dashboard_utils import format_currency, format_percentage, get_color_palette

def render_geography_page():
    """Main function to render the geography analysis page"""
    st.title("🌍 Analyse Géographique des Produits")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ProductAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar filters
    st.sidebar.header("🔍 Filtres Géographiques")
    geography_filters = create_geography_filters()
    
    # Load data with filters
    with st.spinner("Chargement des données..."):
        df = load_geography_data(analyzer, geography_filters)
    
    if df.empty:
        st.error("Aucune donnée trouvée avec ces filtres.")
        return
    
    # Display main metrics
    st.subheader("📊 Métriques Globales")
    display_geography_metrics(df)
    
    # Create three columns for main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Répartition par Région")
        render_regional_distribution(df)
    
    with col2:
        st.subheader("💰 Prix Moyens par Région")
        render_price_by_region(df)
    
    # Full width visualizations
    st.subheader("📈 Analyse Comparative des Régions")
    render_regional_comparison(df)
    
    st.subheader("🔥 Heatmap des Performances")
    render_performance_heatmap(df)
    
    # Detailed analysis
    st.subheader("🏆 Top Produits par Région")
    render_top_products_by_region(df)
    
    # Regional insights
    st.subheader("💡 Insights Géographiques")
    render_geography_insights(df)

def load_geography_data(analyzer, filters):
    """Load and prepare geography data"""
    try:
        # Build MongoDB filters
        mongo_filters = {}
        
        if filters.get('regions'):
            mongo_filters['store_region'] = {'$in': filters['regions']}
        
        if filters.get('min_price') or filters.get('max_price'):
            price_filter = {}
            if filters.get('min_price'):
                price_filter['$gte'] = filters['min_price']
            if filters.get('max_price'):
                price_filter['$lte'] = filters['max_price']
            mongo_filters['price'] = price_filter
        
        if filters.get('availability_only'):
            mongo_filters['available'] = True
        
        # Get products
        df = analyzer.get_products_dataframe(mongo_filters)
        
        if df.empty:
            return df
        
        # Calculate synthetic scores
        criteria = {
            'weights': {
                'price': 0.3,
                'availability': 0.3,
                'stock': 0.2,
                'vendor_popularity': 0.1,
                'recency': 0.1
            },
            'price_preference': 'low'
        }
        
        df_scored = analyzer.calculate_synthetic_score(df, criteria)
        
        return df_scored
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

def render_regional_distribution(df):
    """Render regional distribution chart"""
    if 'store_region' not in df.columns:
        st.warning("Données de région non disponibles")
        return
    
    # Count products by region
    region_counts = df['store_region'].value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=region_counts.values,
        names=region_counts.index,
        title="Distribution des Produits par Région",
        color_discrete_sequence=get_color_palette()
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Produits: %{value}<br>Pourcentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_price_by_region(df):
    """Render price analysis by region"""
    if 'store_region' not in df.columns or 'price' not in df.columns:
        st.warning("Données de prix ou région non disponibles")
        return
    
    # Calculate price statistics by region
    price_stats = df.groupby('store_region')['price'].agg(['mean', 'median', 'std']).round(2)
    price_stats = price_stats.reset_index()
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Prix Moyen',
        x=price_stats['store_region'],
        y=price_stats['mean'],
        marker_color='lightblue',
        text=price_stats['mean'].apply(lambda x: format_currency(x)),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Prix Médian',
        x=price_stats['store_region'],
        y=price_stats['median'],
        marker_color='darkblue',
        text=price_stats['median'].apply(lambda x: format_currency(x)),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Prix Moyens et Médians par Région",
        xaxis_title="Région",
        yaxis_title="Prix (€)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_regional_comparison(df):
    """Render comprehensive regional comparison"""
    if 'store_region' not in df.columns:
        st.warning("Données de région non disponibles")
        return
    
    # Build aggregation dictionary based on available columns
    agg_dict = {
        'price': ['mean', 'count'],
        'available': 'sum',
        'synthetic_score': 'mean'
    }
    
    # Only add stock_quantity if it exists
    if 'stock_quantity' in df.columns:
        agg_dict['stock_quantity'] = 'mean'
    
    # Calculate metrics by region
    regional_metrics = df.groupby('store_region').agg(agg_dict).round(2)
    
    # Handle column names based on what was aggregated
    base_columns = ['Prix_Moyen', 'Nb_Produits', 'Produits_Disponibles', 'Score_Moyen']
    if 'stock_quantity' in df.columns:
        regional_metrics.columns = base_columns + ['Stock_Moyen']
    else:
        regional_metrics.columns = base_columns
    
    regional_metrics = regional_metrics.reset_index()
    
    # Create subplot with available metrics
    if 'stock_quantity' in df.columns:
        # Full version with stock data
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Nombre de Produits', 'Prix Moyen', 'Score Moyen', 'Stock Moyen'),
            vertical_spacing=0.1
        )
        
        # Stock data available
        fig.add_trace(
            go.Bar(x=regional_metrics['store_region'], y=regional_metrics['Stock_Moyen'],
                   name='Stock Moyen', marker_color='lightgoldenrodyellow'),
            row=2, col=2
        )
    else:
        # Version without stock data
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Nombre de Produits', 'Prix Moyen', 'Score Moyen', 'Produits Disponibles'),
            vertical_spacing=0.1
        )
        
        # Show available products instead
        fig.add_trace(
            go.Bar(x=regional_metrics['store_region'], y=regional_metrics['Produits_Disponibles'],
                   name='Produits Disponibles', marker_color='lightgreen'),
            row=2, col=2
        )
    
    # Common charts
    # Number of products
    fig.add_trace(
        go.Bar(x=regional_metrics['store_region'], y=regional_metrics['Nb_Produits'],
               name='Nb Produits', marker_color='lightgreen'),
        row=1, col=1
    )
    
    # Average price
    fig.add_trace(
        go.Bar(x=regional_metrics['store_region'], y=regional_metrics['Prix_Moyen'],
               name='Prix Moyen', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Average score
    fig.add_trace(
        go.Bar(x=regional_metrics['store_region'], y=regional_metrics['Score_Moyen'],
               name='Score Moyen', marker_color='lightskyblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Comparaison Multi-Métriques par Région"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_performance_heatmap(df):
    """Render performance heatmap"""
    if 'store_region' not in df.columns:
        st.warning("Données de région non disponibles")
        return
    
    # Calculate performance metrics by region and vendor
    if 'vendor' in df.columns:
        heatmap_data = df.groupby(['store_region', 'vendor']).agg({
            'synthetic_score': 'mean',
            'price': 'count'
        }).reset_index()
        
        # Filter to top vendors by product count
        top_vendors = df['vendor'].value_counts().head(10).index
        heatmap_data = heatmap_data[heatmap_data['vendor'].isin(top_vendors)]
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(index='vendor', columns='store_region', values='synthetic_score')
        
        # Create heatmap
        fig = px.imshow(
            heatmap_pivot,
            title="Heatmap des Scores par Région et Vendeur",
            labels=dict(x="Région", y="Vendeur", color="Score Moyen"),
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données de vendeur non disponibles pour la heatmap")

def render_top_products_by_region(df):
    """Render top products by region"""
    if 'store_region' not in df.columns:
        st.warning("Données de région non disponibles")
        return
    
    # Get top 3 products per region
    top_products = []
    for region in df['store_region'].unique():
        region_df = df[df['store_region'] == region]
        top_3 = region_df.nlargest(3, 'synthetic_score')
        
        for idx, product in top_3.iterrows():
            top_products.append({
                'Région': region,
                'Produit': product.get('title', 'N/A')[:50] + ('...' if len(str(product.get('title', ''))) > 50 else ''),
                'Vendeur': product.get('vendor', 'N/A'),
                'Prix': format_currency(product.get('price', 0)),
                'Score': round(product.get('synthetic_score', 0), 3),
                'Disponible': '✅' if product.get('available', False) else '❌'
            })
    
    if top_products:
        top_products_df = pd.DataFrame(top_products)
        
        # Display as expandable sections by region
        for region in df['store_region'].unique():
            with st.expander(f"🏆 Top 3 - {region}"):
                region_products = top_products_df[top_products_df['Région'] == region]
                st.dataframe(
                    region_products.drop('Région', axis=1),
                    use_container_width=True,
                    hide_index=True
                )

def render_geography_insights(df):
    """Render geographical insights and recommendations"""
    if df.empty:
        return
    
    insights = []
    
    # Regional analysis
    if 'store_region' in df.columns:
        region_stats = df.groupby('store_region').agg({
            'price': 'mean',
            'synthetic_score': 'mean',
            'available': 'mean'
        }).round(3)
        
        # Best performing region
        best_region = region_stats['synthetic_score'].idxmax()
        best_score = region_stats.loc[best_region, 'synthetic_score']
        
        insights.append(f"🎯 **Région la plus performante**: {best_region} (Score moyen: {best_score:.3f})")
        
        # Most expensive region
        expensive_region = region_stats['price'].idxmax()
        expensive_price = region_stats.loc[expensive_region, 'price']
        
        insights.append(f"💰 **Région la plus chère**: {expensive_region} (Prix moyen: {format_currency(expensive_price)})")
        
        # Best availability
        best_availability = region_stats['available'].idxmax()
        availability_rate = region_stats.loc[best_availability, 'available']
        
        insights.append(f"📦 **Meilleure disponibilité**: {best_availability} ({format_percentage(availability_rate)} de produits disponibles)")
    
    # Price insights
    if 'price' in df.columns:
        avg_price = df['price'].mean()
        price_std = df['price'].std()
        
        insights.append(f"💡 **Prix moyen global**: {format_currency(avg_price)} (±{format_currency(price_std)})")
        
        # Price variance by region
        if 'store_region' in df.columns:
            price_variance = df.groupby('store_region')['price'].std().max()
            insights.append(f"📊 **Plus grande variance de prix**: {format_currency(price_variance)}")
    
    # Display insights
    st.info("\n\n".join(insights))
    
    # Recommendations
    st.subheader("🎯 Recommandations Stratégiques")
    
    recommendations = []
    
    if 'store_region' in df.columns:
        # Find underperforming regions
        region_scores = df.groupby('store_region')['synthetic_score'].mean()
        low_performing = region_scores[region_scores < region_scores.median()].index.tolist()
        
        if low_performing:
            recommendations.append(f"⚠️ **Régions à optimiser**: {', '.join(low_performing)} - Scores inférieurs à la médiane")
        
        # Find high-potential regions
        region_counts = df['store_region'].value_counts()
        low_volume = region_counts[region_counts < region_counts.median()].index.tolist()
        
        if low_volume:
            recommendations.append(f"🚀 **Opportunités d'expansion**: {', '.join(low_volume[:3])} - Faible présence actuelle")
    
    recommendations.append("📈 **Stratégie recommandée**: Concentrer les efforts sur les régions à forte performance et développer les marchés émergents")
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

# Additional utility functions needed in other files

def get_geography_summary(df):
    """Get geography summary statistics"""
    if df.empty or 'store_region' not in df.columns:
        return {}
    
    return {
        'total_regions': df['store_region'].nunique(),
        'top_region': df['store_region'].value_counts().index[0],
        'avg_products_per_region': df.groupby('store_region').size().mean(),
        'price_variance_by_region': df.groupby('store_region')['price'].std().mean() if 'price' in df.columns else 0
    }

def get_regional_top_products(df, region, n=5):
    """Get top N products for a specific region"""
    if 'store_region' not in df.columns:
        return pd.DataFrame()
    
    region_df = df[df['store_region'] == region]
    return region_df.nlargest(n, 'synthetic_score')

def calculate_regional_metrics(df):
    """Calculate comprehensive regional metrics"""
    if 'store_region' not in df.columns:
        return pd.DataFrame()
    
    # Build aggregation dictionary based on available columns
    agg_dict = {
        'price': ['mean', 'median', 'std', 'count'],
        'synthetic_score': ['mean', 'max', 'min'],
        'available': ['sum', 'count']
    }
    
    # Only add stock_quantity if it exists
    if 'stock_quantity' in df.columns:
        agg_dict['stock_quantity'] = 'mean'
    
    metrics = df.groupby('store_region').agg(agg_dict).round(2)
    
    # Flatten column names
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns]
    
    # Calculate availability rate
    metrics['availability_rate'] = (metrics['available_sum'] / metrics['available_count']).round(3)
    
    return metrics.reset_index()

def show_page(analyzer=None):
    """Function to display the geography analysis page"""
    # Store analyzer in session state if provided
    if analyzer is not None:
        st.session_state.analyzer = analyzer
    
    render_geography_page()

if __name__ == "__main__":
    render_geography_page()