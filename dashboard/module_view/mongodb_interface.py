import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import io
from typing import Dict, Any

def show_page():
    """
    Page Streamlit pour l'import de CSV vers MongoDB
    """
    st.title("📊 Import CSV vers MongoDB")
    st.markdown("---")
    
    # Configuration de l'API
    st.sidebar.header("🔧 Configuration API")
    api_base_url = st.sidebar.text_input(
        "URL de base de l'API",
        value="http://localhost:8000",
        help="URL de votre API FastAPI"
    )
    
    # Configuration MongoDB
    st.header("🗄️ Configuration MongoDB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        connection_string = st.text_input(
            "URI de connexion MongoDB",
            value="mongodb://localhost:27017/",
            help="URI de connexion à votre base MongoDB"
        )
        database_name = st.text_input(
            "Nom de la base de données",
            value="products_db",
            help="Nom de la base de données MongoDB"
        )
    
    with col2:
        collection_name = st.text_input(
            "Nom de la collection",
            value="products",
            help="Nom de la collection MongoDB"
        )
    
    # Test de connexion MongoDB
    st.subheader("🔍 Test de connexion")
    
    if st.button("🔗 Tester la connexion MongoDB", type="secondary"):
        with st.spinner("Test de connexion en cours..."):
            try:
                health_data = {
                    "connection_string": connection_string,
                    "database_name": database_name,
                    "collection_name": collection_name
                }
                
                response = requests.post(
                    f"{api_base_url}/mongodb/health",
                    json=health_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Connexion MongoDB réussie!")
                    
                    # Afficher les détails de la connexion
                    st.json(result)
                else:
                    st.error(f"❌ Erreur de connexion: {response.status_code}")
                    st.error(response.text)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur de requête: {str(e)}")
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {str(e)}")
    
    st.markdown("---")
    
    # Section de gestion des fichiers CSV
    st.header("📁 Gestion des fichiers CSV")
    
    # Onglets pour les différentes fonctionnalités
    tab1, tab2, tab3 = st.tabs(["📤 Import CSV", "🔍 Validation CSV", "📋 Template CSV"])
    
    # TAB 1: Import CSV
    with tab1:
        st.subheader("📤 Importer un fichier CSV")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV",
            type=['csv'],
            help="Sélectionnez le fichier CSV à importer dans MongoDB"
        )
        
        if uploaded_file is not None:
            # Aperçu du fichier
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.success(f"✅ Fichier chargé: {uploaded_file.name}")
                st.info(f"📊 Aperçu: {len(df_preview)} lignes (preview), {len(df_preview.columns)} colonnes")
                
                # Afficher l'aperçu
                with st.expander("👀 Aperçu des données"):
                    st.dataframe(df_preview)
                
                # Reset file pointer for actual processing
                uploaded_file.seek(0)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")
                uploaded_file = None
        
        # Options d'import
        st.subheader("⚙️ Options d'import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            clear_collection = st.checkbox(
                "🗑️ Vider la collection avant l'import",
                value=False,
                help="Attention: Cette option supprimera tous les documents existants"
            )
        
        with col2:
            batch_size = st.number_input(
                "📦 Taille des lots",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Nombre de documents à insérer par lot"
            )
        
        # Bouton d'import
        if st.button("🚀 Lancer l'import", type="primary", disabled=uploaded_file is None):
            if uploaded_file is not None:
                with st.spinner("Import en cours... Veuillez patienter..."):
                    try:
                        # Préparer les données pour l'API
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                        data = {
                            'connection_string': connection_string,
                            'database_name': database_name,
                            'collection_name': collection_name,
                            'clear_collection': clear_collection,
                            'batch_size': batch_size
                        }
                        
                        # Appel à l'API d'import
                        response = requests.post(
                            f"{api_base_url}/mongodb/import-csv",
                            files=files,
                            data=data,
                            timeout=300  # 5 minutes timeout
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Afficher les résultats
                            st.success("🎉 Import terminé!")
                            
                            # Métriques
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("📊 Total", result['total_records'])
                            with col2:
                                st.metric("✅ Insérés", result['inserted_records'])
                            with col3:
                                st.metric("❌ Échecs", result['failed_records'])
                            with col4:
                                success_rate = (result['inserted_records'] / result['total_records'] * 100) if result['total_records'] > 0 else 0
                                st.metric("📈 Taux de succès", f"{success_rate:.1f}%")
                            
                            # Statut détaillé
                            if result['status'] == 'success':
                                st.success("✅ Import réussi à 100%")
                            elif result['status'] == 'partial_success':
                                st.warning("⚠️ Import partiellement réussi")
                            else:
                                st.error("❌ Import échoué")
                            
                            # Erreurs s'il y en a
                            if result['errors']:
                                with st.expander("⚠️ Erreurs détectées"):
                                    for error in result['errors']:
                                        st.error(error)
                            
                            # Données d'exemple
                            if result['sample_data']:
                                with st.expander("🔍 Aperçu des données importées"):
                                    st.json(result['sample_data'])
                        
                        else:
                            st.error(f"❌ Erreur d'import: {response.status_code}")
                            st.error(response.text)
                            
                    except requests.exceptions.Timeout:
                        st.error("⏰ Timeout: L'import prend trop de temps. Vérifiez votre connexion.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Erreur de requête: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Erreur inattendue: {str(e)}")
    
    # TAB 2: Validation CSV
    with tab2:
        st.subheader("🔍 Validation de fichier CSV")
        
        validation_file = st.file_uploader(
            "Choisissez un fichier CSV à valider",
            type=['csv'],
            key="validation_file",
            help="Validez la structure de votre CSV avant l'import"
        )
        
        if validation_file is not None:
            if st.button("🔍 Valider le fichier", type="secondary"):
                with st.spinner("Validation en cours..."):
                    try:
                        files = {'file': (validation_file.name, validation_file.getvalue(), 'text/csv')}
                        
                        response = requests.post(
                            f"{api_base_url}/mongodb/validate-csv",
                            files=files,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("✅ Validation terminée!")
                            
                            # Informations générales
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("📊 Colonnes", result['total_columns'])
                            with col2:
                                st.metric("📋 Lignes (sample)", result['sample_rows'])
                            with col3:
                                st.metric("📁 Fichier", result['filename'])
                            
                            # Colonnes détectées
                            with st.expander("📋 Colonnes détectées"):
                                for i, col in enumerate(result['columns'], 1):
                                    col_type = result['column_types'].get(col, 'Unknown')
                                    missing = result['missing_values'].get(col, 0)
                                    st.write(f"{i}. **{col}** ({col_type}) - {missing} valeurs manquantes")
                            
                            # Recommandations
                            if result['recommendations']:
                                st.subheader("💡 Recommandations")
                                for rec in result['recommendations']:
                                    st.warning(f"⚠️ {rec}")
                            else:
                                st.success("✅ Aucun problème détecté!")
                            
                            # Aperçu des données
                            if result['sample_data']:
                                with st.expander("👀 Aperçu des données"):
                                    df_sample = pd.DataFrame(result['sample_data'])
                                    st.dataframe(df_sample)
                        
                        else:
                            st.error(f"❌ Erreur de validation: {response.status_code}")
                            st.error(response.text)
                            
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Erreur de requête: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Erreur inattendue: {str(e)}")
    
    # TAB 3: Template CSV
    with tab3:
        st.subheader("📋 Template CSV")
        st.info("💡 Téléchargez un template CSV avec les colonnes recommandées pour votre import.")
        
        if st.button("📥 Télécharger le template", type="secondary"):
            try:
                response = requests.get(
                    f"{api_base_url}/mongodb/import-template",
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Créer un lien de téléchargement
                    csv_content = response.content
                    
                    st.download_button(
                        label="💾 Télécharger products_template.csv",
                        data=csv_content,
                        file_name="products_template.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    st.success("✅ Template prêt à télécharger!")
                    
                    # Afficher un aperçu du template
                    try:
                        df_template = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
                        with st.expander("👀 Aperçu du template"):
                            st.dataframe(df_template)
                            
                            st.subheader("📋 Description des colonnes")
                            column_descriptions = {
                                'title': 'Nom du produit',
                                'vendor': 'Nom du vendeur/marque',
                                'price': 'Prix du produit (numérique)',
                                'compare_at_price': 'Prix de comparaison (optionnel)',
                                'available': 'Disponibilité (true/false)',
                                'stock_quantity': 'Quantité en stock (numérique)',
                                'store_domain': 'Domaine du magasin',
                                'store_region': 'Région du magasin',
                                'platform': 'Plateforme e-commerce',
                                'tags': 'Tags séparés par des virgules',
                                'created_at': 'Date de création (YYYY-MM-DD)'
                            }
                            
                            for col, desc in column_descriptions.items():
                                st.write(f"• **{col}**: {desc}")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Impossible d'afficher l'aperçu: {str(e)}")
                
                else:
                    st.error(f"❌ Erreur lors du téléchargement: {response.status_code}")
                    st.error(response.text)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur de requête: {str(e)}")
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {str(e)}")
    
    # Section d'aide
    st.markdown("---")
    st.header("📚 Aide et Documentation")
    
    with st.expander("🔧 Configuration requise"):
        st.markdown("""
        **Prérequis:**
        - API FastAPI en cours d'exécution
        - MongoDB accessible
        - Fichier CSV correctement formaté
        
        **Colonnes recommandées:**
        - `title`: Nom du produit (obligatoire)
        - `vendor`: Vendeur/marque
        - `price`: Prix (numérique)
        - `available`: Disponibilité (boolean)
        - `stock_quantity`: Stock (numérique)
        - `store_domain`: Domaine du magasin
        - `platform`: Plateforme e-commerce
        """)
    
    with st.expander("⚠️ Conseils d'utilisation"):
        st.markdown("""
        **Bonnes pratiques:**
        1. **Validez** toujours votre CSV avant l'import
        2. **Testez** la connexion MongoDB avant l'import
        3. **Sauvegardez** vos données avant de vider une collection
        4. **Utilisez des lots** de taille raisonnable (1000-2000)
        5. **Vérifiez** les formats de dates et prix
        
        **Formats supportés:**
        - Dates: YYYY-MM-DD, DD/MM/YYYY
        - Prix: Nombres décimaux (avec point)
        - Booléens: true/false, 1/0, yes/no
        - Tags: Séparés par des virgules
        """)
    
    with st.expander("🐛 Résolution de problèmes"):
        st.markdown("""
        **Erreurs communes:**
        - **Timeout**: Réduisez la taille des lots ou vérifiez la connexion
        - **Parsing Error**: Vérifiez l'encodage et les séparateurs du CSV
        - **Connection Failed**: Vérifiez l'URI MongoDB et les permissions
        - **Invalid Data**: Utilisez la validation pour identifier les problèmes
        
        **Solutions:**
        1. Vérifiez les logs de l'API FastAPI
        2. Testez avec un fichier CSV plus petit
        3. Validez les formats de données
        4. Vérifiez la connectivité réseau
        """)

# Exemple d'utilisation dans une app Streamlit principale
if __name__ == "__main__":
    st.set_page_config(
        page_title="Import CSV MongoDB",
        page_icon="📊",
        layout="wide"
    )
    
    show_page()