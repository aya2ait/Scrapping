import streamlit as st

# ⚠️ IMPORTANT: st.set_page_config() DOIT être la PREMIÈRE commande Streamlit
# Configuration de la page - AVANT tous les autres imports et commandes
st.set_page_config(
    page_title="🚀 Products BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Maintenant on peut faire les autres imports
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
from module_view import overview, top_products, geography, shops_ranking, api_extraction, mongodb_interface, model
from utils.dashboard_utils import init_analyzer, load_custom_css

# Chargement du CSS personnalisé (après set_page_config)
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

def check_api_status():
    """Vérifier l'état de l'API de manière sécurisée"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def display_sidebar_info(analyzer):
    """Afficher les informations dans la sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Informations Système")
    
    # Vérification de l'état de l'API
    if check_api_status():
        st.sidebar.success("✅ API FastAPI connectée")
    else:
        st.sidebar.warning("⚠️ API FastAPI non accessible")
    
    # État de l'analyseur et statistiques
    if analyzer and analyzer.client:
        st.sidebar.success("✅ Base de données connectée")
        
        # Statistiques rapides avec gestion d'erreur
        try:
            df = analyzer.get_products_dataframe({})
            if not df.empty:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    if 'available' in df.columns:
                        available_count = df['available'].sum()
                        st.metric("Dispo", available_count)
                
                if 'store_domain' in df.columns:
                    stores_count = df['store_domain'].nunique()
                    st.sidebar.metric("Boutiques", stores_count)
            else:
                st.sidebar.info("📊 Aucune donnée disponible")
                
        except Exception as e:
            st.sidebar.error(f"Erreur stats: {str(e)[:50]}...")
    else:
        st.sidebar.error("❌ Problème de connexion à la base")
    
    st.sidebar.markdown(f"**Mise à jour:** {datetime.now().strftime('%H:%M:%S')}")

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
        "🔧 Extraction API": "api_extraction",
        "💾 Export vers une DB": "mongodb_interface",
        "🏆 Top Produits": "top_products", 
        "🌍 Analyse Géographique": "geography",
        "🏪 Classement Boutiques": "shops_ranking",
        "🤖 Modèle IA": "model"
    }
    
    selected_page = st.sidebar.selectbox(
        "Choisir une page",
        list(pages.keys()),
        index=0
    )
    
    # Initialisation de l'analyseur
    with st.spinner("Initialisation de l'analyseur..."):
        analyzer = get_analyzer()
    
    # Affichage des informations système
    display_sidebar_info(analyzer)
    
    # Affichage de la page sélectionnée
    page_key = pages[selected_page]
    
    # Container principal pour le contenu
    with st.container():
        try:
            if page_key == "overview":
                overview.show_page(analyzer)
            elif page_key == "api_extraction":
                api_extraction.show_page() 
            elif page_key == "mongodb_interface":
                mongodb_interface.show_page()
            elif page_key == "top_products":
                top_products.show_page(analyzer)
            elif page_key == "geography":
                geography.show_page(analyzer)
            elif page_key == "shops_ranking":
                shops_ranking.show_page(analyzer)
            elif page_key == "model":
                model.run_buyer_analysis()
            else:
                st.error("Page non trouvée")
                
        except ImportError as e:
            st.error(f"❌ Erreur d'import du module: {e}")
            st.info("Vérifiez que tous les modules requis sont présents dans module_view/")
            
        except AttributeError as e:
            st.error(f"❌ Fonction show_page() manquante: {e}")
            st.info("Vérifiez que chaque module a une fonction show_page() définie")
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de la page: {e}")
            with st.expander("Voir les détails de l'erreur"):
                st.exception(e)

if __name__ == "__main__":
    main()