import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
import os
from typing import Dict, List, Any, Optional

# LangChain imports for complex LLM orchestration and conversational interface
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Import du ProductAnalyzer
try:
    # Adjust this import based on where ProductAnalyzer is truly located
    # For this example, assuming 'paste.py' is in the same directory or on PYTHONPATH
    from paste import ProductAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

class BuyerLLMAnalyzer:
    """
    Analyseur LLM pour acheteurs utilisant Groq, avec orchestration LangChain.
    Permet une **orchestration complexe d'appels LLM**, un **chaînage de tâches**
    et une **interface conversationnelle** pour un chat interactif.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API Groq manquante. Définissez GROQ_API_KEY dans les variables d'environnement.")

        # Initialisation du modèle Groq via LangChain
        self.llm = ChatGroq(api_key=self.api_key, model="llama-3.3-70b-versatile", temperature=0.3)
        self.output_parser = StrOutputParser()

    def _prepare_buyer_context(self, data: Dict, analysis_type: str) -> str:
        """Prépare le contexte des données pour l'acheteur"""

        context_parts = []

        # Statistiques pour l'acheteur
        if 'statistics' in data:
            stats = data['statistics']
            context_parts.append(f"""
Options d'achat disponibles:
- Produits analysés: {data.get('total_products', 'N/A')}
- Score moyen de qualité: {stats.get('average_score', 'N/A')}
- Prix le plus bas: {stats.get('price_range', {}).get('min', 'N/A')}€
- Prix le plus élevé: {stats.get('price_range', {}).get('max', 'N/A')}€
- Prix moyen du marché: {stats.get('price_range', {}).get('avg', 'N/A')}€
- Pourcentage de produits en stock: {stats.get('availability_rate', 'N/A')}%
            """)

        # Meilleures options d'achat
        if 'top_k_products' in data:
            top_products = data['top_k_products'][:10]
            context_parts.append("\nMeilleurs choix d'achat (Top 10):")

            for i, product in enumerate(top_products, 1):
                availability_status = "✅ En stock" if product.get('available') else "❌ Rupture de stock"
                stock_info = f" (Stock: {product.get('stock_quantity', 'N/A')})" if product.get('stock_quantity') else ""

                context_parts.append(f"""
{i}. {product.get('title', 'N/A')}
   - Vendeur: {product.get('vendor', 'N/A')}
   - Prix: {product.get('price', 'N/A')}€
   - Score qualité: {product.get('synthetic_score', 'N/A')}/1.0
   - Disponibilité: {availability_status}{stock_info}
   - Plateforme: {product.get('platform', 'N/A')}
   - Région: {product.get('store_region', 'N/A')}
                """)

        # Analyse par région (utile pour les frais de port)
        if 'geographical_analysis' in data and data['geographical_analysis']:
            geo_data = data['geographical_analysis']
            context_parts.append(f"\nDisponibilité par région: {list(geo_data.keys())}")

        # Analyse des vendeurs (fiabilité)
        if 'shops_analysis' in data and data['shops_analysis']:
            shops_data = data['shops_analysis']
            if 'top_shops' in shops_data:
                context_parts.append(f"\nVendeurs recommandés: {list(shops_data['top_shops'].keys())[:5]}")

        return "\n".join(context_parts)

    def _get_buyer_analysis_prompt_template(self, analysis_type: str) -> ChatPromptTemplate:
        """Génère le prompt LangChain selon le type d'analyse pour l'acheteur"""

        system_message_content = "Tu es un expert en conseil d'achat et comparaison de produits. Tu aides les clients à faire les meilleurs choix d'achat en analysant les produits disponibles, leurs prix, leur qualité et leur rapport qualité-prix."

        base_template = """
Voici les produits disponibles à l'achat:

{context}

"""
        if analysis_type == "general":
            user_message_content = base_template + """
En tant que conseiller d'achat, fournis une analyse complète pour aider le client à faire le meilleur choix:

1. **Recommandations d'achat prioritaires**: Quels sont les 3-5 meilleurs produits à acheter et pourquoi?
2. **Rapport qualité-prix**: Quels produits offrent le meilleur rapport qualité-prix?
3. **Alertes importantes**: Y a-t-il des produits en rupture de stock ou à acheter rapidement?
4. **Comparaison des prix**: Analyse des écarts de prix et des bonnes affaires
5. **Vendeurs recommandés**: Quels vendeurs privilégier pour la fiabilité?
6. **Conseils d'achat**: Tips pratiques pour optimiser l'achat (timing, région, etc.)

Sois pratique, orienté client et aide à économiser de l'argent tout en obtenant la meilleure qualité.
"""
        elif analysis_type == "budget":
            user_message_content = base_template + """
Analyse budget et économies pour l'acheteur:

1. **Options économiques**: Meilleurs produits dans chaque gamme de prix
2. **Bonnes affaires**: Produits avec le meilleur rapport qualité-prix
3. **Comparaison des prix**: Écarts de prix entre vendeurs pour les mêmes produits
4. **Timing d'achat**: Quand acheter pour économiser (stock, promotions possibles)
5. **Budget recommandé**: Quel budget prévoir pour un achat de qualité?
6. **Pièges à éviter**: Prix trop bas suspects, vendeurs peu fiables

Focus sur l'optimisation du budget et les économies possibles.
"""
        elif analysis_type == "quality":
            user_message_content = base_template + """
Analyse qualité et fiabilité pour l'acheteur:

1. **Produits premium**: Meilleurs produits en termes de qualité (scores élevés)
2. **Fiabilité des vendeurs**: Vendeurs les plus fiables et réputés
3. **Disponibilité et stock**: Produits garantis en stock vs risque de rupture
4. **Critères de qualité**: Qu'est-ce qui rend ces produits meilleurs que d'autres?
5. **Investissement durable**: Produits qui valent l'investissement à long terme
6. **Signaux d'alerte**: Produits ou vendeurs à éviter

Priorité à la qualité et à la fiabilité de l'achat.
"""
        elif analysis_type == "urgency":
            user_message_content = base_template + """
Analyse urgence et disponibilité pour l'acheteur:

1. **Achat immédiat**: Produits à acheter maintenant (stock limité)
2. **Produits en tension**: Articles qui risquent la rupture de stock
3. **Alternatives disponibles**: Solutions de secours si produit principal indisponible
4. **Délais de livraison**: Estimation des délais selon les régions/vendeurs
5. **Stock monitoring**: Produits à surveiller pour disponibilité
6. **Plan d'achat**: Stratégie d'achat selon l'urgence du besoin

Focus sur la disponibilité et l'urgence d'achat.
"""
        else:
            user_message_content = base_template + "Fournis des conseils d'achat personnalisés basés sur ces données produits."

        return ChatPromptTemplate.from_messages([
            ("system", system_message_content),
            ("user", user_message_content)
        ])

    def analyze_for_buyer(self, products_data: Dict, analysis_type: str = "general") -> str:
        """
        Analyse des produits du point de vue d'un acheteur en utilisant LangChain.
        """
        context = self._prepare_buyer_context(products_data, analysis_type)
        prompt_template = self._get_buyer_analysis_prompt_template(analysis_type)

        # Chaîne LangChain pour l'analyse
        analysis_chain = (
            {"context": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | self.output_parser
        )

        try:
            # Exécution de la chaîne LangChain
            response_content = analysis_chain.invoke(context)
            return response_content
        except Exception as e:
            return f"Erreur lors de l'analyse LangChain: {str(e)}"

    def get_conversational_chain(self) -> Any:
        """
        Retourne une chaîne conversationnelle LangChain pour un chat interactif.
        """
        # Le prompt inclut un historique de messages pour le contexte conversationnel
        conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant d'achat intelligent. Tu réponds aux questions des utilisateurs sur les produits, les prix et les meilleurs choix en te basant sur les données de produits que tu as reçues. Sois concis et utile."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ])

        # Création de la chaîne conversationnelle
        conversational_chain = conversational_prompt | self.llm | self.output_parser
        return conversational_chain

def display_buyer_analysis_results(top_k_df, df_scored, ai_analysis, analysis_type, llm_data):
    """Affiche les résultats de l'analyse pour l'acheteur"""

    # Analyse IA pour acheteur
    st.markdown("## 🛒 Conseil d'Achat IA")

    analysis_types_labels = {
        "general": "🎯 Recommandations Générales",
        "budget": "💰 Optimisation Budget",
        "quality": "⭐ Analyse Qualité",
        "urgency": "⚡ Urgence d'Achat"
    }

    st.markdown(f"### {analysis_types_labels.get(analysis_type, 'Conseil d\'Achat')}")

    # Afficher l'analyse dans un container stylisé
    st.markdown(
        f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #FF6B35;">
        {ai_analysis.replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Métriques acheteur
    st.markdown("## 📊 Aperçu du Marché")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Options disponibles", len(top_k_df))

    with col2:
        if 'price' in top_k_df.columns:
            min_price = top_k_df['price'].min()
            st.metric("Prix minimum", f"{min_price:.2f}€")
        else:
            st.metric("Prix minimum", "N/A")

    with col3:
        if 'price' in top_k_df.columns:
            avg_price = top_k_df['price'].mean()
            st.metric("Prix moyen", f"{avg_price:.2f}€")
        else:
            st.metric("Prix moyen", "N/A")

    with col4:
        if 'available' in top_k_df.columns:
            availability = top_k_df['available'].mean() * 100
            st.metric("Produits en stock", f"{availability:.1f}%")
        else:
            st.metric("Disponibilité", "N/A")

    # Tableau des meilleures options d'achat
    st.markdown("## 🏆 Meilleures Options d'Achat")

    # Préparer l'affichage orienté acheteur
    display_df = top_k_df.copy()
    display_df['Rang'] = range(1, len(display_df) + 1)
    display_df['Score Qualité'] = display_df['synthetic_score'].round(3)

    if 'price' in display_df.columns:
        display_df['Prix'] = display_df['price'].apply(lambda x: f"{x:.2f}€")

    if 'available' in display_df.columns:
        display_df['Disponibilité'] = display_df['available'].apply(
            lambda x: "🟢 En stock" if x else "🔴 Rupture"
        )

    # Ajouter recommandation d'achat
    def get_recommendation(row):
        score = row['synthetic_score']
        available = row.get('available', False)

        if not available:
            return "⏳ Attendre"
        elif score >= 0.8:
            return "⭐ Excellent choix"
        elif score >= 0.6:
            return "👍 Bon choix"
        else:
            return "🤔 À étudier"

    display_df['Recommandation'] = display_df.apply(get_recommendation, axis=1)

    # Colonnes à afficher pour l'acheteur
    columns_to_show = ['Rang', 'title', 'vendor', 'Prix', 'Disponibilité', 'Score Qualité', 'Recommandation']

    if 'store_region' in display_df.columns:
        display_df['Région'] = display_df['store_region']
        columns_to_show.append('Région')

    column_names = {
        'title': 'Produit',
        'vendor': 'Vendeur'
    }

    st.dataframe(
        display_df[columns_to_show].rename(columns=column_names),
        use_container_width=True
    )

    # Graphiques orientés acheteur
    st.markdown("## 📈 Analyse des Options")

    col1, col2 = st.columns(2)

    with col1:
        # Analyse prix vs qualité
        if 'price' in top_k_df.columns:
            fig_scatter = px.scatter(
                top_k_df,
                x='price',
                y='synthetic_score',
                size='stock_quantity' if 'stock_quantity' in top_k_df.columns else None,
                color='available' if 'available' in top_k_df.columns else None,
                title="Prix vs Qualité - Trouvez la Meilleure Affaire",
                labels={'price': 'Prix (€)', 'synthetic_score': 'Score Qualité'},
                hover_data=['title', 'vendor'],
                color_discrete_map={True: '#00CC96', False: '#FF6B6B'}
            )
            fig_scatter.add_shape(
                type="line",
                x0=top_k_df['price'].min(), y0=0.7,
                x1=top_k_df['price'].max(), y1=0.7,
                line=dict(color="orange", width=2, dash="dash"),
            )
            fig_scatter.add_annotation(
                x=top_k_df['price'].mean(), y=0.72,
                text="Seuil Qualité Recommandé",
                showarrow=False,
                font=dict(color="orange")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Distribution des prix
        fig_price_dist = px.histogram(
            top_k_df,
            x='price' if 'price' in top_k_df.columns else 'synthetic_score',
            title="Distribution des Prix du Marché",
            nbins=15,
            color_discrete_sequence=['#FF6B35']
        )

        if 'price' in top_k_df.columns:
            # Ajouter ligne prix moyen
            avg_price = top_k_df['price'].mean()
            fig_price_dist.add_vline(
                x=avg_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Prix moyen: {avg_price:.2f}€"
            )

        fig_price_dist.update_layout(height=400)
        st.plotly_chart(fig_price_dist, use_container_width=True)

    # Analyse des vendeurs
    if 'vendor' in top_k_df.columns:
        vendor_analysis = top_k_df.groupby('vendor').agg({
            'synthetic_score': 'mean',
            'price': 'mean' if 'price' in top_k_df.columns else 'count',
            'available': 'mean' if 'available' in top_k_df.columns else 'count'
        }).round(3)

        fig_vendors = px.bar(
            vendor_analysis.reset_index(),
            x='vendor',
            y='synthetic_score',
            title="Scores Moyens par Vendeur",
            color='synthetic_score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_vendors, use_container_width=True)

    # Section conseils d'achat
    st.markdown("## 💡 Conseils d'Achat Rapides")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Meilleure affaire
        if not top_k_df.empty and 'price' in top_k_df.columns:
            # Avoid division by zero for synthetic_score or price
            best_value_products = top_k_df[top_k_df['synthetic_score'] > 0]
            if not best_value_products.empty:
                # Calculate ratio, handle division by zero for price
                best_value_products = best_value_products[best_value_products['price'] > 0]
                if not best_value_products.empty:
                    best_value = best_value_products.loc[(best_value_products['synthetic_score'] / best_value_products['price']).idxmax()]
                    st.success(f"🎯 **Meilleure affaire**: {best_value.iloc[0]['title'][:30]}... à {best_value.iloc[0]['price']:.2f}€")
                else:
                    st.info("Aucune meilleure affaire trouvée (prix non positifs).")
            else:
                st.info("Aucune meilleure affaire trouvée (scores non positifs).")
        else:
            st.info("Aucune meilleure affaire trouvée (données insuffisantes).")

    with col2:
        # Alerte stock
        if 'available' in top_k_df.columns:
            low_stock = top_k_df[top_k_df['available'] == False]
            if not low_stock.empty:
                st.warning(f"⚠️ **{len(low_stock)} produits** en rupture de stock")
            else:
                st.info("✅ Tous les produits sont disponibles")

    with col3:
        # Prix moyen gamme
        if 'price' in top_k_df.columns and not top_k_df.empty:
            price_ranges = pd.cut(top_k_df['price'], bins=3, labels=['Budget', 'Moyen', 'Premium'])
            most_common_range = price_ranges.value_counts().index[0]
            st.info(f"📊 **Gamme dominante**: {most_common_range}")
        else:
            st.info("N/A (Prix ou données manquantes)")

    # Export et actions
    st.markdown("## 💾 Sauvegarde et Partage")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export liste d'achat
        shopping_list = top_k_df[['title', 'vendor', 'price', 'available']].copy()
        shopping_list['À acheter'] = shopping_list['available'].apply(lambda x: '✓' if x else '✗')
        csv_shopping = shopping_list.to_csv(index=False)

        st.download_button(
            "🛒 Liste d'Achat CSV",
            csv_shopping,
            file_name=f"liste_achat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # Export conseil d'achat
        buying_guide = f"""
Guide d'Achat Personnalisé - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Type d'analyse: {analysis_type}

{ai_analysis}

Résumé des options:
- Nombre de produits analysés: {len(top_k_df)}
- Prix moyen: {top_k_df['price'].mean():.2f}€ (si disponible)
- Score qualité moyen: {top_k_df['synthetic_score'].mean():.3f}
- Taux de disponibilité: {top_k_df['available'].mean()*100:.1f}%

Meilleur choix: {top_k_df.iloc[0]['title']}
Prix: {top_k_df.iloc[0]['price']:.2f}€
Vendeur: {top_k_df.iloc[0]['vendor']}
"""

        st.download_button(
            "📋 Guide d'Achat",
            buying_guide,
            file_name=f"guide_achat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col3:
        # Nouvelle recherche
        if st.button("🔄 Nouvelle Recherche", use_container_width=True):
            st.session_state.clear() # Clear session state for a fresh start
            st.rerun()

    # Chat interactif
    st.markdown("## 💬 Chat Interactif (LangChain)")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_analyzer_chat" not in st.session_state:
        st.session_state.llm_analyzer_chat = BuyerLLMAnalyzer(llm_data.get('api_key')) # Pass API key here
        st.session_state.conversational_chain = st.session_state.llm_analyzer_chat.get_conversational_chain()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Posez une question sur les produits..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("L'IA est en train de réfléchir..."):
            try:
                # Prepare chat history for LangChain
                chat_history_lc = []
                for msg in st.session_state.messages[:-1]: # Exclude current user message
                    if msg["role"] == "user":
                        chat_history_lc.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history_lc.append(AIMessage(content=msg["content"]))

                # Invoke the conversational chain
                response = st.session_state.conversational_chain.invoke({
                    "question": prompt,
                    "chat_history": chat_history_lc
                })
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Erreur lors du chat avec l'IA: {str(e)}")


def show_buyer_analysis_page():
    """Page d'analyse pour acheteurs - Trouvez les meilleurs produits à acheter"""

    st.title("🛒 Assistant d'Achat Intelligent")
    st.markdown("Trouvez les meilleurs produits à acheter avec l'aide de l'IA")

    # Vérifications préalables
    if not ANALYZER_AVAILABLE:
        st.error("❌ ProductAnalyzer non disponible. Assurez-vous que le fichier 'paste.py' est accessible.")
        return

    # Configuration de l'API Groq
    with st.sidebar:
        st.markdown("### 🔑 Configuration")

        # Clé API
        groq_api_key = st.text_input(
            "Clé API Groq",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Obtenez votre clé API sur https://console.groq.com/"
        )

        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        st.markdown("---")

        # Type d'analyse acheteur
        analysis_type = st.selectbox(
            "Type de conseil d'achat",
            ["general", "budget", "quality", "urgency"],
            format_func=lambda x: {
                "general": "🎯 Conseils généraux",
                "budget": "💰 Optimisation budget",
                "quality": "⭐ Focus qualité",
                "urgency": "⚡ Achat urgent"
            }[x]
        )

        st.markdown("---")

        # Préférences d'achat
        st.markdown("### 🎯 Vos Préférences")

        # Budget maximum
        max_budget = st.number_input("Budget maximum (€)", min_value=0, value=500, step=10)

        # Nombre de produits à analyser
        k = st.slider("Nombre de produits à comparer", 5, 30, 15)

        # Priorités personnelles
        st.markdown("#### Vos priorités")
        price_importance = st.slider("Importance du prix", 0.0, 1.0, 0.4, 0.05)
        quality_importance = st.slider("Importance de la qualité", 0.0, 1.0, 0.35, 0.05)
        availability_importance = st.slider("Importance de la disponibilité", 0.0, 1.0, 0.25, 0.05)

    if not groq_api_key:
        st.warning("⚠️ Veuillez configurer votre clé API Groq dans la sidebar")
        return

    # Interface principal
    col1, col2 = st.columns([2, 1])

    with col2:
        # Configuration de recherche
        st.markdown("### 🔍 Critères de Recherche")

        # Région préférée
        preferred_region = st.selectbox(
            "Région préférée",
            ["Toutes", "US", "EU", "CA", "AU"],
            index=0,
            help="Peut affecter les frais de port"
        )

        # Disponibilité
        stock_preference = st.radio(
            "Disponibilité",
            ["Tous les produits", "En stock seulement", "Stock élevé seulement"],
            index=1
        )

        # Gamme de prix
        if max_budget > 0:
            price_range = st.slider(
                "Gamme de prix souhaitée (€)",
                0, max_budget,
                (0, max_budget),
                step=5
            )
        else:
            price_range = (0, 1000)

    with col1:
        st.markdown("### 🎯 Recherche de Produits")

        # Bouton de recherche
        if st.button("🔍 Trouver les Meilleurs Produits", type="primary", use_container_width=True):

            with st.spinner("Recherche des meilleures options d'achat..."):
                try:
                    # Initialiser l'analyseur
                    analyzer = ProductAnalyzer()

                    if analyzer.client is None:
                        st.error("❌ Connexion base de données impossible")
                        return

                    # Construire les filtres
                    filters = {}

                    # Filtre de disponibilité
                    if stock_preference == "En stock seulement":
                        filters['available'] = True
                    elif stock_preference == "Stock élevé seulement":
                        filters['stock_quantity'] = {"$gte": 10}
                        # If 'available' is also a filter, ensure it's True
                        if 'available' not in filters:
                            filters['available'] = True


                    # Filtre de région
                    if preferred_region != "Toutes":
                        filters['store_region'] = preferred_region

                    # Filtre de prix
                    if price_range[0] > 0 or price_range[1] < max_budget:
                        price_filter = {}
                        if price_range[0] > 0:
                            price_filter["$gte"] = price_range[0]
                        if price_range[1] < max_budget:
                            price_filter["$lte"] = price_range[1]
                        filters['price'] = price_filter

                    # Critères orientés acheteur
                    criteria = {
                        'weights': {
                            'price': price_importance,
                            'availability': availability_importance,
                            'stock': 0.15,
                            'vendor_popularity': quality_importance * 0.3,
                            'recency': quality_importance * 0.2
                        },
                        'price_preference': 'low'  # Les acheteurs préfèrent les prix bas
                    }

                    # Obtenir les données
                    st.info("📊 Recherche de produits...")
                    df = analyzer.get_products_dataframe(filters)

                    if df.empty:
                        st.warning("Aucun produit trouvé avec ces critères. Essayez d'élargir votre recherche.")
                        return

                    # Calculer les scores
                    st.info("🧮 Évaluation des options...")
                    df_scored = analyzer.calculate_synthetic_score(df, criteria)

                    # Sélectionner les meilleures options
                    top_k_df = analyzer.get_top_k_products(df_scored, k, 'synthetic_score')

                    # Analyses supplémentaires
                    geo_analysis = analyzer.analyze_by_geography(df_scored)
                    shops_analysis = analyzer.analyze_shops_ranking(df_scored)

                    # Préparer les données pour l'analyse IA
                    llm_data = {
                        'total_products': len(df),
                        'k': k,
                        'budget': max_budget,
                        'top_k_products': [],
                        'statistics': {
                            'average_score': df_scored['synthetic_score'].mean(),
                            'score_std': df_scored['synthetic_score'].std(),
                            'price_range': {
                                'min': df_scored['price'].min() if 'price' in df_scored.columns else None,
                                'max': df_scored['price'].max() if 'price' in df_scored.columns else None,
                                'avg': df_scored['price'].mean() if 'price' in df_scored.columns else None
                            },
                            'availability_rate': df_scored['available'].mean() * 100 if 'available' in df_scored.columns else None
                        },
                        'geographical_analysis': geo_analysis,
                        'shops_analysis': shops_analysis,
                        'api_key': groq_api_key # Pass the API key to llm_data for the chat
                    }

                    # Convertir les meilleures options pour l'IA
                    for idx, row in top_k_df.iterrows():
                        product_data = {
                            'title': row.get('title', ''),
                            'vendor': row.get('vendor', ''),
                            'price': row.get('price', 0),
                            'synthetic_score': row.get('synthetic_score', 0),
                            'available': row.get('available', False),
                            'stock_quantity': row.get('stock_quantity', 0),
                            'platform': row.get('platform', ''),
                            'store_region': row.get('store_region', '')
                        }
                        llm_data['top_k_products'].append(product_data)

                    # Analyse IA pour l'acheteur
                    st.info("🤖 Génération des conseils d'achat...")
                    llm_analyzer = BuyerLLMAnalyzer(groq_api_key)
                    ai_analysis = llm_analyzer.analyze_for_buyer(llm_data, analysis_type)

                    # Affichage des résultats pour l'acheteur
                    display_buyer_analysis_results(top_k_df, df_scored, ai_analysis, analysis_type, llm_data)

                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche: {str(e)}")
                    st.exception(e)

# Fonction utilitaire pour l'utilisation externe
def run_buyer_analysis():
    """Point d'entrée pour l'analyse acheteur"""
    show_buyer_analysis_page()

if __name__ == "__main__":
    run_buyer_analysis()