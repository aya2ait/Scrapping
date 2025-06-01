import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import des utilitaires
from utils.dashboard_utils import (
    format_currency, format_number, create_metric_card,
    get_color_palette, create_donut_chart, create_bar_chart
)

# Import du ProductAnalyzer
try:
    # Tentative d'import direct
    from paste import ProductAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        sys.path.append(str(Path(__file__).parent))
        from paste import ProductAnalyzer
        ANALYZER_AVAILABLE = True
    except ImportError:
        try:
            # Tentative depuis le répertoire parent
            sys.path.append(str(Path(__file__).parent.parent))
            from paste import ProductAnalyzer
            ANALYZER_AVAILABLE = True
        except ImportError:
            st.error("❌ ProductAnalyzer non trouvé. Assurez-vous que le fichier 'paste.py' contient votre classe ProductAnalyzer.")
            ANALYZER_AVAILABLE = False
            ProductAnalyzer = None

def show_page(analyzer=None):
    """Affiche la page de vue d'ensemble du dashboard"""
    
    st.markdown("## 📊 Vue d'ensemble des Produits")
    
    # Vérification de la disponibilité de ProductAnalyzer
    if not ANALYZER_AVAILABLE:
        st.error("❌ ProductAnalyzer non disponible. Impossible d'afficher les données.")
        st.info("Assurez-vous que votre fichier 'paste.py' contient la classe ProductAnalyzer.")
        return
    
    # Initialisation de l'analyzer si non fourni
    if analyzer is None:
        try:
            with st.spinner("Initialisation de l'analyseur de produits..."):
                analyzer = ProductAnalyzer()
                st.success("✅ ProductAnalyzer initialisé avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation de ProductAnalyzer: {str(e)}")
            st.info("Vérifiez que votre classe ProductAnalyzer est correctement définie.")
            return
    
    # Vérification que l'analyzer a la méthode nécessaire
    if not hasattr(analyzer, 'get_products_dataframe'):
        st.error("❌ La classe ProductAnalyzer doit avoir une méthode 'get_products_dataframe'.")
        return
    
    try:
        # Chargement des données avec plus de diagnostic
        with st.spinner("Chargement des données..."):
            df = analyzer.get_products_dataframe({})
            
        # Debug: Afficher des informations sur le DataFrame
        if st.checkbox("Afficher les informations de debug", key="debug_overview"):
            st.write(f"**Nombre de lignes:** {len(df)}")
            st.write(f"**Colonnes disponibles:** {list(df.columns)}")
            st.write(f"**Types de données:**")
            st.write(df.dtypes)
            st.write("**Aperçu des données:**")
            st.write(df.head())
            st.write("**Valeurs nulles par colonne:**")
            st.write(df.isnull().sum())
            
        if df.empty:
            st.warning("Aucune donnée de produit disponible.")
            return
            
        # Affichage forcé des premières données pour diagnostic
        st.info(f"Données chargées avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Métriques principales
        display_main_metrics(df)
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            display_availability_chart(df)
            
        with col2:
            display_price_distribution(df)
        
        # Graphiques secondaires
        col3, col4 = st.columns(2)
        
        with col3:
            display_top_stores(df)
            
        with col4:
            display_category_analysis(df)
        
        # Tableau des produits récents
        display_recent_products(df)
        
    except Exception as e:
        st.error(f"Erreur lors du chargement de la vue d'ensemble: {str(e)}")
        st.exception(e)

def find_column(df, possible_names):
    """Trouve la première colonne correspondante dans la liste"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def safe_numeric_conversion(series):
    """Conversion sécurisée en numérique"""
    try:
        # Tentative de conversion directe
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Suppression des valeurs nulles et infinies
        numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan).dropna()
        return numeric_series
    except:
        return pd.Series(dtype=float)

def display_main_metrics(df):
    """Affiche les métriques principales"""
    
    st.markdown("### 📈 Métriques Principales")
    
    # Calcul des métriques avec vérifications
    total_products = len(df)
    
    # Vérification pour la disponibilité
    availability_col = find_column(df, ['available', 'disponible', 'in_stock', 'stock'])
    available_products = 0
    
    if availability_col:
        try:
            if df[availability_col].dtype == 'bool':
                available_products = df[availability_col].sum()
            else:
                # Conversion en booléen
                bool_series = df[availability_col].astype(str).str.lower().isin(['true', '1', 'yes', 'oui', 'disponible', 'available'])
                available_products = bool_series.sum()
        except:
            available_products = 0
    
    # Vérification pour les boutiques
    store_col = find_column(df, ['store_domain', 'store', 'shop', 'boutique', 'seller'])
    total_stores = 0
    
    if store_col:
        try:
            total_stores = df[store_col].dropna().nunique()
        except:
            total_stores = 0
    
    # Vérification pour les prix
    price_col = find_column(df, ['price', 'prix', 'cost', 'amount', 'value'])
    avg_price = 0
    
    if price_col:
        try:
            numeric_prices = safe_numeric_conversion(df[price_col])
            if len(numeric_prices) > 0:
                avg_price = numeric_prices.mean()
        except:
            avg_price = 0
    
    # Affichage en colonnes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Total Produits",
            format_number(total_products)
        ), unsafe_allow_html=True)
    
    with col2:
        availability_rate = (available_products / total_products * 100) if total_products > 0 else 0
        st.markdown(create_metric_card(
            "Produits Disponibles",
            f"{format_number(available_products)} ({availability_rate:.1f}%)"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Boutiques",
            format_number(total_stores)
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "Prix Moyen",
            format_currency(avg_price) if avg_price > 0 else "N/A"
        ), unsafe_allow_html=True)

def display_availability_chart(df):
    """Affiche le graphique de disponibilité des produits"""
    
    st.markdown("### 🟢 Disponibilité des Produits")
    
    # Recherche de colonnes de disponibilité
    availability_col = find_column(df, ['available', 'disponible', 'in_stock', 'stock'])
    
    if availability_col is None:
        st.info("Données de disponibilité non disponibles")
        return
    
    try:
        # Préparation des données de disponibilité
        if df[availability_col].dtype == 'bool':
            availability_counts = df[availability_col].value_counts()
        else:
            # Conversion en booléen avec plus de flexibilité
            bool_series = df[availability_col].astype(str).str.lower().isin([
                'true', '1', 'yes', 'oui', 'disponible', 'available', 'in stock'
            ])
            availability_counts = bool_series.value_counts()
        
        # Vérification des données
        if availability_counts.empty:
            st.info("Aucune donnée de disponibilité valide")
            return
        
        labels = ['Disponible', 'Non disponible']
        values = [
            availability_counts.get(True, 0),
            availability_counts.get(False, 0)
        ]
        
        # Vérification que nous avons des données à afficher
        if sum(values) == 0:
            st.info("Aucune donnée de disponibilité à afficher")
            return
        
        # Création du graphique avec gestion d'erreur
        try:
            fig = create_donut_chart(values, labels, "Répartition de la Disponibilité")
            st.plotly_chart(fig, use_container_width=True)
        except:
            # Fallback avec un graphique simple
            fig = px.pie(values=values, names=labels, title="Répartition de la Disponibilité")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de disponibilité: {str(e)}")

def display_price_distribution(df):
    """Affiche la distribution des prix"""
    
    st.markdown("### 💰 Distribution des Prix")
    
    # Recherche de colonnes de prix
    price_col = find_column(df, ['price', 'prix', 'cost', 'amount', 'value'])
    
    if price_col is None:
        st.info("Données de prix non disponibles")
        return
    
    try:
        # Nettoyage et conversion des données de prix
        price_data = safe_numeric_conversion(df[price_col])
        
        if len(price_data) == 0:
            st.info("Aucune donnée de prix valide")
            return
        
        # Filtrage des valeurs aberrantes (prix négatifs ou très élevés)
        price_data = price_data[(price_data > 0) & (price_data < price_data.quantile(0.99))]
        
        if len(price_data) == 0:
            st.info("Aucune donnée de prix valide après filtrage")
            return
        
        # Création de l'histogramme
        try:
            fig = px.histogram(
                x=price_data,
                nbins=min(30, len(price_data)//2),  # Ajustement automatique des bins
                title="Distribution des Prix des Produits",
                labels={'x': 'Prix (€)', 'y': 'Nombre de Produits'},
                color_discrete_sequence=[get_color_palette()[0]]
            )
            
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except:
            # Fallback simple
            st.bar_chart(pd.DataFrame({'Prix': price_data}))
        
        # Affichage de statistiques supplémentaires
        st.write(f"**Statistiques des prix:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prix minimum", format_currency(price_data.min()))
        with col2:
            st.metric("Prix médian", format_currency(price_data.median()))
        with col3:
            st.metric("Prix maximum", format_currency(price_data.max()))
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de prix: {str(e)}")

def display_top_stores(df):
    """Affiche le top des boutiques par nombre de produits"""
    
    st.markdown("### 🏪 Top 10 Boutiques")
    
    # Recherche de colonnes de boutique
    store_col = find_column(df, ['store_domain', 'store', 'shop', 'boutique', 'seller'])
    
    if store_col is None:
        st.info("Données de boutiques non disponibles")
        return
    
    try:
        # Calcul du top 10 avec nettoyage
        store_counts = df[store_col].dropna().value_counts().head(10)
        
        if len(store_counts) == 0:
            st.info("Aucune donnée de boutique disponible")
            return
        
        # Création du graphique en barres
        try:
            fig = create_bar_chart(
                x_data=store_counts.index,
                y_data=store_counts.values,
                title="Nombre de Produits par Boutique",
                x_title="Boutique",
                y_title="Nombre de Produits"
            )
            
            # Rotation des labels pour une meilleure lisibilité
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        except:
            # Fallback simple
            st.bar_chart(store_counts)
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique des boutiques: {str(e)}")

def display_category_analysis(df):
    """Affiche l'analyse par catégorie"""
    
    st.markdown("### 📂 Analyse par Catégorie")
    
    # Recherche d'une colonne de catégorie
    category_col = find_column(df, ['category', 'categories', 'product_category', 'type', 'categorie'])
    
    if category_col is None:
        st.info("Données de catégorie non disponibles")
        return
    
    try:
        # Calcul des catégories avec nettoyage
        category_counts = df[category_col].dropna().value_counts().head(8)
        
        if len(category_counts) == 0:
            st.info("Aucune donnée de catégorie disponible")
            return
        
        # Création du graphique en donut
        try:
            fig = create_donut_chart(
                category_counts.values,
                category_counts.index,
                "Répartition par Catégorie"
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            # Fallback avec un graphique pie simple
            fig = px.pie(values=category_counts.values, names=category_counts.index, 
                        title="Répartition par Catégorie")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique des catégories: {str(e)}")

def display_recent_products(df):
    """Affiche les produits récents"""
    
    st.markdown("### 🆕 Produits Récents")
    
    try:
        # Recherche d'une colonne de date
        date_col = find_column(df, ['created_at', 'date_added', 'scraped_at', 'last_updated', 'timestamp', 'date'])
        
        # Sélection des colonnes à afficher
        name_col = find_column(df, ['name', 'title', 'product_name', 'nom'])
        price_col = find_column(df, ['price', 'prix', 'cost'])
        store_col = find_column(df, ['store_domain', 'store', 'shop', 'boutique'])
        availability_col = find_column(df, ['available', 'disponible', 'in_stock'])
        
        display_cols = []
        col_mapping = {}
        
        # Construction de la liste des colonnes à afficher
        if name_col:
            display_cols.append(name_col)
            col_mapping[name_col] = 'Nom'
        if price_col:
            display_cols.append(price_col)
            col_mapping[price_col] = 'Prix'
        if store_col:
            display_cols.append(store_col)
            col_mapping[store_col] = 'Boutique'
        if availability_col:
            display_cols.append(availability_col)
            col_mapping[availability_col] = 'Disponible'
        if date_col:
            display_cols.append(date_col)
            col_mapping[date_col] = 'Date'
        
        # Si aucune colonne spécifique n'est trouvée, prendre les premières colonnes
        if not display_cols:
            display_cols = df.columns[:min(5, len(df.columns))].tolist()
            col_mapping = {col: col.title() for col in display_cols}
        
        # Préparation des données
        if date_col is None:
            st.info("Affichage des premiers produits (pas de colonne de date détectée)")
            recent_products = df[display_cols].head(10)
        else:
            # Conversion de la colonne de date
            df_copy = df.copy()
            try:
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                # Tri par date décroissante
                recent_products = df_copy.sort_values(date_col, ascending=False, na_position='last').head(10)
                recent_products = recent_products[display_cols]
            except:
                st.info("Erreur de conversion de date, affichage des premiers produits")
                recent_products = df[display_cols].head(10)
        
        # Formatage pour l'affichage
        display_df = recent_products.copy()
        
        # Formatage des prix si présent
        if price_col and price_col in display_df.columns:
            def format_price(x):
                try:
                    if pd.notna(x):
                        numeric_val = float(x)
                        return format_currency(numeric_val)
                    return 'N/A'
                except:
                    return str(x)
            
            display_df[price_col] = display_df[price_col].apply(format_price)
        
        # Formatage des dates si présent
        if date_col and date_col in display_df.columns:
            try:
                display_df[date_col] = pd.to_datetime(display_df[date_col], errors='coerce')
                display_df[date_col] = display_df[date_col].dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass  # Garder les valeurs originales si la conversion échoue
        
        # Renommage des colonnes pour l'affichage
        display_df = display_df.rename(columns=col_mapping)
        
        # Limitation de la largeur des cellules pour une meilleure présentation
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Affichage d'informations supplémentaires
        if len(recent_products) > 0:
            st.success(f"Affichage de {len(recent_products)} produits")
        else:
            st.warning("Aucun produit à afficher")
        
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des produits récents: {str(e)}")
        # Affichage de fallback
        try:
            st.info("Affichage des premières lignes disponibles:")
            st.dataframe(df.head(10), use_container_width=True)
        except:
            st.error("Impossible d'afficher les données")