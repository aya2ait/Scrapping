import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"  # Modifier selon votre configuration

# === FONCTIONS API ===

def check_api_health():
    """Vérifier si l'API est accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_task_status(task_id):
    """Récupérer le statut d'une tâche"""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def list_all_tasks():
    """Lister toutes les tâches"""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def start_extraction(stores_data, extraction_type="all", scraping_config=None):
    """Démarrer une extraction"""
    try:
        endpoint_map = {
            "all": "/extract/all",
            "shopify": "/extract/shopify", 
            "woocommerce": "/extract/woocommerce",
            "single": "/extract/single"
        }
        
        endpoint = endpoint_map.get(extraction_type, "/extract/all")
        
        if extraction_type == "single":
            payload = {
                "store": stores_data[0],
                "scraping_config": scraping_config,
                "output_format": "csv"
            }
        else:
            payload = {
                "stores": stores_data,
                "scraping_config": scraping_config,
                "output_format": "csv"
            }
        
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel API: {str(e)}")
        return None

def download_task_result(task_id):
    """Télécharger le résultat d'une tâche"""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}/download")
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

# === FONCTIONS D'AFFICHAGE ===

def show_extraction_form():
    """Formulaire pour lancer une nouvelle extraction"""
    
    st.subheader("🚀 Lancer une Nouvelle Extraction")
    
    # Type d'extraction
    extraction_type = st.selectbox(
        "Type d'extraction",
        ["all", "shopify", "woocommerce", "single"],
        format_func=lambda x: {
            "all": "🌐 Toutes les plateformes",
            "shopify": "🛍️ Shopify uniquement", 
            "woocommerce": "🛒 WooCommerce uniquement",
            "single": "🏪 Boutique unique"
        }[x]
    )
    
    # Configuration des boutiques
    st.subheader("🏪 Configuration des Boutiques")
    
    if extraction_type == "single":
        # Configuration pour une seule boutique
        col1, col2 = st.columns(2)
        with col1:
            domain = st.text_input("Domaine", placeholder="example.com")
            name = st.text_input("Nom de la boutique", placeholder="Ma Boutique")
            platform = st.selectbox("Plateforme", ["shopify", "woocommerce", "generic"])
        
        with col2:
            region = st.text_input("Région", value="Unknown")
            currency = st.text_input("Devise", value="USD")
            priority = st.number_input("Priorité", min_value=1, value=1)
        
        stores_data = [{
            "domain": domain,
            "name": name,
            "platform": platform,
            "region": region,
            "currency": currency,
            "priority": priority
        }]
    
    else:
        # Configuration pour plusieurs boutiques
        st.info("💡 Configurez vos boutiques en format JSON ou utilisez l'interface ci-dessous")
        
        # Option 1: JSON
        with st.expander("📝 Configuration JSON"):
            sample_config = [
                {
                    "domain": "example-shopify.com",
                    "name": "Example Shopify Store",
                    "platform": "shopify",
                    "region": "US",
                    "currency": "USD",
                    "priority": 1
                },
                {
                    "domain": "example-woo.com", 
                    "name": "Example WooCommerce Store",
                    "platform": "woocommerce",
                    "region": "EU",
                    "currency": "EUR",
                    "priority": 2
                }
            ]
            
            stores_json = st.text_area(
                "Configuration des boutiques (JSON)",
                value=json.dumps(sample_config, indent=2),
                height=200
            )
            
            try:
                stores_data = json.loads(stores_json)
            except:
                st.error("Format JSON invalide")
                stores_data = []
        
        # Option 2: Interface graphique simple
        with st.expander("🖱️ Configuration Graphique"):
            num_stores = st.number_input("Nombre de boutiques", min_value=1, max_value=10, value=2)
            
            stores_data = []
            for i in range(num_stores):
                st.write(f"**Boutique {i+1}**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    domain = st.text_input(f"Domaine {i+1}", key=f"domain_{i}")
                    name = st.text_input(f"Nom {i+1}", key=f"name_{i}")
                
                with col2:
                    platform = st.selectbox(f"Plateforme {i+1}", ["shopify", "woocommerce", "generic"], key=f"platform_{i}")
                    region = st.text_input(f"Région {i+1}", value="Unknown", key=f"region_{i}")
                
                with col3:
                    currency = st.text_input(f"Devise {i+1}", value="USD", key=f"currency_{i}")
                    priority = st.number_input(f"Priorité {i+1}", min_value=1, value=i+1, key=f"priority_{i}")
                
                if domain and name:
                    stores_data.append({
                        "domain": domain,
                        "name": name,
                        "platform": platform,
                        "region": region,
                        "currency": currency,
                        "priority": priority
                    })
    
    # Configuration de scraping
    st.subheader("⚙️ Configuration de Scraping")
    
    with st.expander("🔧 Paramètres Avancés"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_retries = st.number_input("Nombre de tentatives", min_value=1, max_value=10, value=3)
            timeout = st.number_input("Timeout (secondes)", min_value=10, max_value=120, value=30)
            delay_requests = st.number_input("Délai entre requêtes (s)", min_value=0.5, max_value=10.0, value=1.5)
        
        with col2:
            delay_domains = st.number_input("Délai entre domaines (s)", min_value=1.0, max_value=30.0, value=2.0)
            max_products = st.number_input("Max produits par boutique", min_value=100, max_value=50000, value=10000)
            use_selenium = st.checkbox("Utiliser Selenium", value=False)
    
    scraping_config = {
        "max_retries": max_retries,
        "timeout": timeout,
        "delay_between_requests": delay_requests,
        "delay_between_domains": delay_domains,
        "max_products_per_store": max_products,
        "use_selenium": use_selenium,
        "headless": True
    }
    
    # Bouton de lancement
    st.markdown("---")
    
    if st.button("🚀 Lancer l'Extraction", type="primary"):
        if not stores_data:
            st.error("Veuillez configurer au moins une boutique")
        else:
            with st.spinner("Lancement de l'extraction..."):
                result = start_extraction(stores_data, extraction_type, scraping_config)
                
                if result:
                    st.success(f"✅ Extraction lancée avec succès!")
                    st.info(f"**Task ID:** {result['task_id']}")
                    st.info(f"**Message:** {result['message']}")
                    
                    # Stocker le task_id dans la session
                    if 'active_tasks' not in st.session_state:
                        st.session_state.active_tasks = []
                    st.session_state.active_tasks.append(result['task_id'])

def show_tasks_monitoring():
    """Afficher le suivi des tâches"""
    
    st.subheader("📋 Suivi des Tâches d'Extraction")
    
    # Bouton pour actualiser
    if st.button("🔄 Actualiser les tâches"):
        st.rerun()
    
    # Récupérer toutes les tâches
    tasks_data = list_all_tasks()
    
    if tasks_data and tasks_data.get('tasks'):
        tasks = tasks_data['tasks']
        
        # Affichage sous forme de tableau
        tasks_df = pd.DataFrame(tasks)
        
        # Colonnes à afficher
        display_columns = ['task_id', 'type', 'status', 'created_at', 'stores_count']
        if 'total_products' in tasks_df.columns:
            display_columns.append('total_products')
        
        # Filtrer les colonnes existantes
        available_columns = [col for col in display_columns if col in tasks_df.columns]
        
        if available_columns:
            st.dataframe(
                tasks_df[available_columns],
                use_container_width=True,
                column_config={
                    "task_id": "ID Tâche",
                    "type": "Type",
                    "status": "Statut",
                    "created_at": "Créé le",
                    "stores_count": "Nb Boutiques",
                    "total_products": "Nb Produits"
                }
            )
        
        # Détails des tâches
        st.subheader("🔍 Détails des Tâches")
        
        for task in tasks:
            with st.expander(f"📋 {task['task_id'][:8]}... - {task['status'].upper()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {task['type']}")
                    st.write(f"**Statut:** {task['status']}")
                    st.write(f"**Créé le:** {task['created_at']}")
                    if 'stores_count' in task:
                        st.write(f"**Nombre de boutiques:** {task['stores_count']}")
                
                with col2:
                    if task['status'] == 'completed':
                        st.success("✅ Terminé")
                        if 'total_products' in task:
                            st.write(f"**Produits extraits:** {task['total_products']}")
                        
                        # Bouton de téléchargement
                        if st.button(f"⬇️ Télécharger", key=f"download_{task['task_id']}"):
                            data = download_task_result(task['task_id'])
                            if data:
                                st.download_button(
                                    label="💾 Sauvegarder CSV",
                                    data=data,
                                    file_name=f"extraction_{task['task_id'][:8]}.csv",
                                    mime="text/csv"
                                )
                    
                    elif task['status'] == 'running':
                        st.info("🔄 En cours...")
                    
                    elif task['status'] == 'failed':
                        st.error("❌ Échec")
                        if 'error' in task:
                            st.write(f"**Erreur:** {task['error']}")
    
    else:
        st.info("Aucune tâche trouvée")

def show_api_configuration():
    """Configuration de l'API"""
    
    st.subheader("⚙️ Configuration de l'API")
    
    # Configuration de l'URL
    current_url = st.text_input("URL de l'API", value=API_BASE_URL)
    
    if st.button("🔗 Tester la connexion"):
        try:
            response = requests.get(f"{current_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connexion réussie!")
                st.json(response.json())
            else:
                st.error(f"❌ Erreur: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Impossible de se connecter: {str(e)}")
    
    # Exemples de configuration
    st.subheader("📖 Exemples de Configuration")
    
    with st.expander("🏪 Exemple Configuration Boutiques"):
        st.json([
            {
                "domain": "example-shopify.com",
                "name": "Example Shopify Store", 
                "platform": "shopify",
                "region": "US",
                "currency": "USD",
                "priority": 1
            }
        ])
    
    with st.expander("⚙️ Exemple Configuration Scraping"):
        st.json({
            "max_retries": 3,
            "timeout": 30,
            "delay_between_requests": 1.5,
            "delay_between_domains": 2.0,
            "max_products_per_store": 10000,
            "use_selenium": False,
            "headless": True
        })

# === FONCTION PRINCIPALE ===

def show_api_extraction_page():
    """Fonction principale pour afficher la page d'extraction API"""
    
    st.markdown("""
    <div class="main-header">
        <h2>🔧 Extraction de Données via API</h2>
        <p>Interface pour lancer des extractions de produits e-commerce</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérification de l'état de l'API
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if check_api_health():
            st.success("✅ API connectée et fonctionnelle")
        else:
            st.error("❌ API non accessible. Vérifiez que FastAPI est démarré.")
            st.info("Pour démarrer l'API: `uvicorn main:app --reload --port 8000`")
            return
    
    with col2:
        if st.button("🔄 Actualiser le statut API"):
            st.rerun()
    
    # Tabs pour organiser les fonctionnalités
    tab1, tab2, tab3 = st.tabs([
        "🚀 Nouvelle Extraction", 
        "📋 Suivi des Tâches", 
        "⚙️ Configuration"
    ])
    
    with tab1:
        show_extraction_form()
    
    with tab2:
        show_tasks_monitoring()
    
    with tab3:
        show_api_configuration()

def show_page():
    """Fonction principale pour afficher la page d'extraction API"""
    
    # Configuration du style CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-card {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Extraction de Données E-commerce</h1>
        <p>Interface pour lancer et suivre des extractions de produits via API</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérification de l'état de l'API
    col1, col2 = st.columns([3, 1])
    
    with col1:
        api_status = check_api_health()
        if api_status:
            st.markdown("""
            <div class="status-card success-card">
                <strong>✅ API connectée et fonctionnelle</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card error-card">
                <strong>❌ API non accessible</strong><br>
                Vérifiez que FastAPI est démarré sur le port 8000
            </div>
            """, unsafe_allow_html=True)
            st.code("uvicorn main:app --reload --port 8000", language="bash")
            return
    
    with col2:
        if st.button("🔄 Actualiser", type="secondary"):
            st.rerun()
    
    # Navigation par onglets
    tab1, tab2, tab3 = st.tabs([
        "🚀 Nouvelle Extraction", 
        "📋 Suivi des Tâches", 
        "⚙️ Configuration"
    ])
    
    with tab1:
        show_extraction_form()
    
    with tab2:
        show_tasks_monitoring()
    
    with tab3:
        show_api_configuration()
    
    # Footer avec informations
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>🚀 Interface d'extraction e-commerce | Powered by FastAPI & Streamlit</small>
    </div>
    """, unsafe_allow_html=True)