import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Ajouter le chemin du projet pour les imports
sys.path.append(os.path.dirname(__file__))

# Imports des composants du dashboard
from module_view import overview, top_products, geography, shops_ranking, api_extraction,mongodb_interface
from utils.dashboard_utils import init_analyzer, load_custom_css

# Configuration de la page
st.set_page_config(
    page_title="🚀 Products BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du CSS personnalisé
load_custom_css()

# Initialisation de l'analyseur (cache pour performance)
@st.cache_resource
def get_analyzer():
    try:
        analyzer = init_analyzer()
        if analyzer is None:
            st.error("Impossible d'initialiser l'analyseur de données")
        return analyzer
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de l'analyseur: {e}")
        return None

def main():
    # Titre principal avec style
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Products Business Intelligence Dashboard</h1>
        <p>Analyse avancée et visualisation des produits e-commerce</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour navigation
    st.sidebar.markdown("### 🧭 Navigation")
    
    # Menu de navigation
    pages = {
        "📊 Vue d'ensemble": "overview",
        "🏆 Top Produits": "top_products", 
        "🌍 Analyse Géographique": "geography",
        "🏪 Classement Boutiques": "shops_ranking",
        "🔧 Extraction API": "api_extraction",
        "Export vers une DB": "mongodb_interface"
    }
    
    selected_page = st.sidebar.selectbox(
        "Choisir une page",
        list(pages.keys()),
        index=0
    )
    
    # Affichage des informations système
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Informations Système")
    
    # Vérification de l'état de l'API (uniquement pour info dans la sidebar)
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("✅ API FastAPI connectée")
        else:
            st.sidebar.warning("⚠️ API FastAPI non accessible")
    except:
        st.sidebar.warning("⚠️ API FastAPI non accessible")
    
    # Initialisation de l'analyseur
    analyzer = get_analyzer()
    
    if analyzer and analyzer.client:
        st.sidebar.success("✅ Base de données connectée")
        
        # Statistiques rapides
        try:
            df = analyzer.get_products_dataframe({})
            if not df.empty:
                st.sidebar.metric("Total Produits", len(df))
                if 'available' in df.columns:
                    available_count = df['available'].sum()
                    st.sidebar.metric("Produits Disponibles", available_count)
                if 'store_domain' in df.columns:
                    stores_count = df['store_domain'].nunique()
                    st.sidebar.metric("Boutiques", stores_count)
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement des stats: {e}")
    else:
        st.sidebar.error("❌ Problème de connexion à la base")
    
    st.sidebar.markdown(f"**Dernière mise à jour:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Affichage de la page sélectionnée
    page_key = pages[selected_page]
    
    try:
        if page_key == "overview":
            overview.show_page(analyzer)
        elif page_key == "top_products":
            top_products.show_page(analyzer)
        elif page_key == "geography":
            geography.show_page(analyzer)
        elif page_key == "shops_ranking":
            shops_ranking.show_page(analyzer)
        elif page_key == "api_extraction":
            api_extraction.show_page() 
        elif page_key == "mongodb_interface":
            mongodb_interface.show_page() 
    except Exception as e:
        st.error(f"Erreur lors du chargement de la page: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()