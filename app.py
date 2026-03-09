import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Lloyds Churn", page_icon="🐴", layout="wide", initial_sidebar_state="collapsed")

LOGO_B64 = "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"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#f5f3ee!important;color:#1a1a1a!important;font-family:'Source Sans 3',sans-serif!important}
[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"]{display:none!important}
.main .block-container{max-width:1080px!important;padding:0 2rem 4rem!important;margin:0 auto!important}
.top-banner{background:#006633;margin:0 -2rem 0;height:5px}
.header-bar{background:#fff;border-bottom:1px solid #e0ddd5;padding:1.1rem 2rem;margin:0 -2rem 2.5rem;display:flex;align-items:center;gap:1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.header-logo{height:46px;width:auto}
.header-divider{width:1px;height:34px;background:#d0cdc5}
.header-title{font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:600;color:#1a1a1a;margin:0}
.header-subtitle{font-size:.76rem;color:#6b6b6b;margin:.15rem 0 0}
.header-right{margin-left:auto;text-align:right}
.header-name{font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;color:#1a1a1a;margin:0}
.header-role{font-size:.72rem;color:#006633;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin:.1rem 0 0}
.page-title{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#1a1a1a;margin:0 0 .4rem}
.page-desc{font-size:.95rem;color:#555;margin:0 0 2rem;max-width:560px;line-height:1.6}
.section-head{display:flex;align-items:center;gap:.7rem;margin-bottom:1rem}
.section-dot{width:9px;height:9px;border-radius:50%;background:#006633;flex-shrink:0}
.section-title{font-size:.72rem;font-weight:600;color:#006633;letter-spacing:.12em;text-transform:uppercase}
.card{background:#fff;border:1px solid #e4e1d9;border-radius:3px;padding:1.6rem 1.8rem;margin-bottom:1.4rem;box-shadow:0 1px 3px rgba(0,0,0,.05)}
[data-testid="stNumberInput"] label,[data-testid="stSelectbox"] label{font-size:.75rem!important;font-weight:600!important;color:#555!important;letter-spacing:.06em!important;text-transform:uppercase!important}
[data-testid="stNumberInput"] input,[data-testid="stSelectbox"]>div>div{background:#fafaf8!important;border:1px solid #ccc9c0!important;border-radius:3px!important;color:#1a1a1a!important;font-size:.95rem!important}
[data-testid="stButton"]>button{background:#006633!important;color:#fff!important;font-weight:600!important;font-size:.88rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;border:none!important;border-radius:3px!important;padding:.75rem 2rem!important;width:100%!important;height:auto!important;box-shadow:0 2px 6px rgba(0,102,51,.2)!important;margin-top:.5rem!important}
[data-testid="stButton"]>button:hover{background:#005229!important}
.result-churn{background:#fff8f7;border-left:4px solid #b91c1c;border-radius:3px;padding:1.6rem 1.8rem;margin-top:1rem;animation:fadeIn .3s ease}
.result-safe{background:#f4fbf6;border-left:4px solid #006633;border-radius:3px;padding:1.6rem 1.8rem;margin-top:1rem;animation:fadeIn .3s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.res-badge{display:inline-block;font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:.25rem .7rem;border-radius:2px;margin-bottom:.7rem}
.res-badge-churn{background:#fee2e2;color:#b91c1c}
.res-badge-safe{background:#dcfce7;color:#006633}
.res-title{font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;color:#1a1a1a;margin:0 0 1.2rem}
.prob-wrap{display:flex;align-items:center;gap:1.2rem}
.prob-big{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;line-height:1;flex-shrink:0}
.prob-big-churn{color:#b91c1c}.prob-big-safe{color:#006633}
.prob-right{flex:1;display:flex;flex-direction:column;gap:.35rem}
.prob-label{font-size:.8rem;color:#777}
.prob-track{background:#eae7e0;height:5px;border-radius:3px;overflow:hidden}
.prob-fill-churn{background:#b91c1c;height:100%;border-radius:3px}
.prob-fill-safe{background:#006633;height:100%;border-radius:3px}
.prob-ticks{display:flex;justify-content:space-between;font-size:.68rem;color:#bbb;margin-top:.2rem}
.watermark{position:fixed;right:-60px;bottom:-60px;width:560px;height:560px;opacity:.045;pointer-events:none;z-index:0}
.lloyds-footer{border-top:1px solid #e0ddd5;padding:1.4rem 0 0;margin-top:3rem;display:flex;justify-content:space-between;font-size:.74rem;color:#aaa}
.fg{color:#006633;font-weight:600}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
pipeline = load_model()

st.markdown('<img class="watermark" src="data:image/png;base64,' + LOGO_B64 + '" />', unsafe_allow_html=True)
st.markdown('<div class="top-banner"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-bar">'
    '<img src="data:image/png;base64,' + LOGO_B64 + '" class="header-logo" alt="Lloyds"/>'
    '<div class="header-divider"></div>'
    '<div>'
    '<div class="header-title">Analyse Client</div>'
    '<div class="header-subtitle">Outil de prédiction interne &nbsp;·&nbsp; Usage restreint</div>'
    '</div>'
    '<div class="header-right">'
    '<div class="header-name">Samba Diakho</div>'
    '<div class="header-role">Data Scientist Junior</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="page-title">Prédiction du Churn Client</div>', unsafe_allow_html=True)
st.markdown('<div class="page-desc">Renseignez le profil du client. Le modèle évalue la probabilité de résiliation prochaine.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-head"><div class="section-dot"></div><div class="section-title">Profil &amp; Démographie</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=34)
    gender = st.selectbox("Genre", ["M","F"], format_func=lambda x: "Homme" if x=="M" else "Femme")
    marital_status = st.selectbox("Statut marital", ["Single","Married","Widowed","Divorced"],
        format_func=lambda x: {"Single":"Célibataire","Married":"Marié·e","Widowed":"Veuf·ve","Divorced":"Divorcé·e"}[x])
    income_level = st.selectbox("Niveau de revenu", ["Low","Medium","High"],
        format_func=lambda x: {"Low":"Faible","Medium":"Moyen","High":"Élevé"}[x])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div class="section-dot"></div><div class="section-title">Transactions &amp; Activité</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    amount_spent = st.number_input("Montant dépensé (£)", min_value=0.0, value=250.0, step=10.0)
    login_freq = st.number_input("Connexions par mois", min_value=0, value=8)
    product_category = st.selectbox("Catégorie de produit",
        ["Electronics","Clothing","Furniture","Groceries","Books"],
        format_func=lambda x: {"Electronics":"Électronique","Clothing":"Vêtements","Furniture":"Mobilier","Groceries":"Alimentation","Books":"Livres"}[x])
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-head"><div class="section-dot"></div><div class="section-title">Service &amp; Relation Client</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    service_usage = st.selectbox("Canal utilisé", ["Mobile App","Website","Online Banking"],
        format_func=lambda x: {"Mobile App":"Application mobile","Website":"Site web","Online Banking":"Banque en ligne"}[x])
    interaction_type = st.selectbox("Type d’interaction",
        ["Inquiry","Feedback","Complaint","nan"],
        format_func=lambda x: {"Inquiry":"Renseignement","Feedback":"Retour d’expérience","Complaint":"Réclamation","nan":"Non renseigné"}[x])
    resolution_status = st.selectbox("Statut de résolution",
        ["Resolved","Unresolved","nan"],
        format_func=lambda x: {"Resolved":"Résolu","Unresolved":"Non résolu","nan":"Non renseigné"}[x])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div class="section-dot"></div><div class="section-title">Résultat</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button("Lancer l’analyse prédictive"):
        # Convert "nan" strings to actual NaN for the model
        it_val = np.nan if interaction_type == "nan" else interaction_type
        rs_val = np.nan if resolution_status == "nan" else resolution_status

        df = pd.DataFrame({
            "AmountSpent":[amount_spent],"Age":[age],"LoginFrequency":[login_freq],
            "ProductCategory":[product_category],"Gender":[gender],"MaritalStatus":[marital_status],
            "IncomeLevel":[income_level],"ResolutionStatus":[rs_val],
            "ServiceUsage":[service_usage],"InteractionType":[it_val],
        })
        pred  = pipeline.predict(df)[0]
        proba = pipeline.predict_proba(df)[0][1]
        pct   = round(proba * 100, 1)
        bw    = str(pct) + "%"
        if pred == 1:
            html = (
                '<div class="result-churn">'
                '<div class="res-badge res-badge-churn">&#9888; Risque élevé</div>'
                '<div class="res-title">Ce client est susceptible de partir</div>'
                '<div class="prob-wrap">'
                '<div class="prob-big prob-big-churn">' + str(pct) + '%</div>'
                '<div class="prob-right">'
                '<div class="prob-label">Probabilité de churn</div>'
                '<div class="prob-track"><div class="prob-fill-churn" style="width:' + bw + '"></div></div>'
                '<div class="prob-ticks"><span>0%</span><span>50%</span><span>100%</span></div>'
                '</div></div></div>'
            )
        else:
            html = (
                '<div class="result-safe">'
                '<div class="res-badge res-badge-safe">&#10003; Faible risque</div>'
                '<div class="res-title">Ce client devrait rester fidèle</div>'
                '<div class="prob-wrap">'
                '<div class="prob-big prob-big-safe">' + str(pct) + '%</div>'
                '<div class="prob-right">'
                '<div class="prob-label">Probabilité de churn</div>'
                '<div class="prob-track"><div class="prob-fill-safe" style="width:' + bw + '"></div></div>'
                '<div class="prob-ticks"><span>0%</span><span>50%</span><span>100%</span></div>'
                '</div></div></div>'
            )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="lloyds-footer">'
    '<span><span class="fg">Lloyds Banking Group</span> &nbsp;·&nbsp; Usage interne uniquement</span>'
    '<span>Outil ML confidentiel &nbsp;·&nbsp; Ne pas diffuser</span>'
    '</div>',
    unsafe_allow_html=True
)