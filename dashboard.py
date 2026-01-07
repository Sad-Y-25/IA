import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Traffic AI Control Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ (LAYOUT & DESIGN) ---
st.markdown("""
<style>
    /* 1. Réduire le vide en haut de page */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Fond global */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Cartes KPI (Metrics) */
    div[data-testid="metric-container"] {
        background-color: #21262D;
        border: 1px solid #30363D;
        padding: 10px 15px;
        border-radius: 8px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Bouton Actualiser (Alignement et Style) */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        height: 3em;
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
    }

    /* Titres H2 H3 plus compacts */
    h2 { font-size: 1.8rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.3rem !important; margin-top: 0rem !important; }
    
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS ---
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

def calculate_kpis(df):
    if df is None or df.empty: return 0, 0, 0
    avg_queue = df['queue'].mean()
    avg_wait = df['avg_wait'].mean() if 'avg_wait' in df.columns else 0
    total_cars = df['cars'].iloc[-1] if 'cars' in df.columns else 0
    return avg_queue, avg_wait, total_cars

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #58A6FF; margin:0;'>🚦 TrafficFlow AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8B949E; font-size: 12px; margin-bottom: 20px;'>Système de Gestion Intelligent</p>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Monitoring", "Benchmark", "About"],
        icons=["activity", "bar-chart-fill", "info-circle"],
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#58A6FF", "font-size": "16px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "#30363D"},
            "nav-link-selected": {"background-color": "#238636"},
        }
    )
    st.markdown("---")
    st.caption("État du Système")
    c1, c2 = st.columns([1, 5])
    with c1: st.write("🟢")
    with c2: st.markdown("**Mode Statique**\n<span style='color:#8B949E; font-size:11px'>Données chargées</span>", unsafe_allow_html=True)

# ==============================================================================
# MODE 1 : MONITORING
# ==============================================================================
if selected == "Monitoring":
    # --- HEADER COMPACT ---
    col_header, col_refresh = st.columns([6, 1], gap="small")
    with col_header:
        st.markdown("## 📡 Monitoring de la Simulation")
    with col_refresh:
        if st.button("🔄 Actualiser"):
            st.rerun()

    # Chargement
    df = load_data("simulation_data.csv")
    
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        step_id = int(latest['step'])
        
        # --- KPIS (Ligne unique) ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Step", step_id)
        k2.metric("File d'Attente", int(latest['queue']), delta_color="inverse")
        k3.metric("Attente Moy.", f"{latest['avg_wait']:.1f}s")
        k4.metric("Reward", f"{latest['reward']:.0f}")

        # --- GRAPHIQUES (Structure 70% / 30%) ---
        st.markdown("---")
        col_main, col_side = st.columns([7, 3], gap="medium")
        
        with col_main:
            st.subheader("📉 Historique de Congestion")
            fig_line = px.line(df, x="step", y="queue")
            
            # Optimisation Layout Graphique
            fig_line.update_layout(
                margin=dict(l=10, r=10, t=10, b=10), # Marges nulles
                height=320, # Hauteur fixe
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E6EDF3", size=11),
                xaxis=dict(showgrid=False, title="Simulation Step"),
                yaxis=dict(showgrid=True, gridcolor='#30363D', title="Véhicules")
            )
            fig_line.update_traces(line_color='#58A6FF', line_width=2.5)
            
            # Marqueurs Ambulance (Toujours visibles sur le graphe, mais pas en alerte texte)
            if df['ambulance'].sum() > 0:
                amb_data = df[df['ambulance'] > 0]
                fig_line.add_scatter(
                    x=amb_data['step'], y=amb_data['queue'], 
                    mode='markers', name='Ambulance', 
                    marker=dict(color='#FF5555', size=8, symbol='x')
                )
            
            st.plotly_chart(fig_line, use_container_width=True)
            
        with col_side:
            st.subheader("📷 Vision (Caméra)")
            vision_data = pd.DataFrame({
                'Type': ['Piétons', 'Autos', 'Amb.', 'Camions'],
                'Compte': [latest['peds'], latest['cars'], latest['ambulance'], latest['trucks']]
            })
            fig_bar = px.bar(vision_data, x='Type', y='Compte', color='Type', 
                             color_discrete_map={'Amb.': '#FF5555', 'Autos': '#58A6FF', 'Camions': '#8B949E', 'Piétons': '#FFA657'})
            
            fig_bar.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E6EDF3", size=11),
                showlegend=False,
                xaxis=dict(title=None),
                yaxis=dict(showgrid=True, gridcolor='#30363D')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("⚠️ Aucune donnée. Lancez `python run_simulation.py`.")

# ==============================================================================
# MODE 2 : BENCHMARK
# ==============================================================================
elif selected == "Benchmark":
    st.markdown("## 🏆 Analyse Comparative")
    
    with st.expander("📁 Sources de données", expanded=False):
        c1, c2 = st.columns(2)
        df_ai = load_data(c1.text_input("IA CSV", "simulation_data.csv"))
        df_classic = load_data(c2.text_input("Classique CSV", "resultats_classic.csv"))

    if df_ai is not None and df_classic is not None:
        q_ai, w_ai, c_ai = calculate_kpis(df_ai)
        q_cl, w_cl, c_cl = calculate_kpis(df_classic)
        delta_q = ((q_ai - q_cl) / q_cl) * 100 if q_cl != 0 else 0
        
        # KPIs Comparatifs
        k1, k2, k3 = st.columns(3)
        k1.metric("File Moyenne (Veh)", f"{q_ai:.1f}", f"{delta_q:.1f}%", delta_color="inverse")
        k2.metric("Attente Moyenne (s)", f"{w_ai:.1f}", "0.0% (Simulé)", delta_color="off")
        k3.metric("Stabilité (Écart Type)", f"{df_ai['queue'].std():.1f}", f"vs {df_classic['queue'].std():.1f}", delta_color="off")

        st.markdown("---")

        # Tabs Graphiques
        tab1, tab2 = st.tabs(["📉 Courbes de Congestion", "📊 Comparaison Globale"])

        with tab1:
            df_combined = pd.concat([
                df_ai.assign(Système='IA (Smart)'),
                df_classic.assign(Système='Classique (Timer)')
            ])
            fig = px.line(df_combined, x="step", y="queue", color="Système",
                          color_discrete_map={'IA (Smart)': '#238636', 'Classique (Timer)': '#da3633'})
            fig.update_layout(
                height=350, 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)', 
                font_color="white",
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            comp_data = pd.DataFrame({
                'Système': ['IA Smart', 'Classique'],
                'File Moyenne': [q_ai, q_cl]
            })
            fig_comp = px.bar(comp_data, x='Système', y='File Moyenne', color='Système', 
                              color_discrete_map={'IA Smart': '#238636', 'Classique': '#da3633'})
            fig_comp.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.warning("⚠️ Données manquantes.")

# ==============================================================================
# MODE 3 : ABOUT
# ==============================================================================
elif selected == "About":
    st.markdown("## ℹ️ À propos")
    st.info("Projet académique : Optimisation de trafic par IA et Vision.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Technologies :**")
        st.code("Python, SUMO, Q-Learning, OpenCV")
    with col2:
        st.markdown("**Objectif :**")
        st.write("Minimiser la congestion et prioriser les urgences.")