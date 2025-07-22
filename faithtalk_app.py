
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# Carregar e preparar a Bíblia
# ============================
@st.cache_data
def carregar_versiculos():
    df = pd.read_csv("versiculos_biblia.csv")
    return df

df_versiculos = carregar_versiculos()
versiculos_texto = df_versiculos["texto"].tolist()

# ========================
# Vetorização com TF-IDF
# ========================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(versiculos_texto)

# ========================
# Função de busca espiritual
# ========================
def buscar_versiculos(entrada_usuario, top_n=3):
    entrada_vetor = vectorizer.transform([entrada_usuario])
    similaridades = cosine_similarity(entrada_vetor, tfidf_matrix).flatten()
    indices_mais_similares = similaridades.argsort()[::-1][:top_n]
    return df_versiculos.iloc[indices_mais_similares]

# ========================
# Layout do App
# ========================
st.set_page_config(page_title="FaithTalk AI – Conselhos Espirituais com a Bíblia", page_icon="📖", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #f7f2ea;
        font-family: 'Georgia', serif;
    }
    .title {
        color: #4a2c2a;
        text-align: center;
        font-size: 2.4rem;
        font-weight: bold;
    }
    .verse {
        background-color: #fff7f0;
        border-left: 6px solid #d4a373;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📖 FaithTalk AI</div>', unsafe_allow_html=True)
st.markdown("### Compartilhe seu sentimento, dor ou gratidão. Receba a resposta da Palavra de Deus.")

entrada = st.text_area("✍️ Escreva aqui seu momento:")

if st.button("🔍 Buscar versículos"):
    if entrada.strip() == "":
        st.warning("Por favor, escreva algo para receber orientação espiritual.")
    else:
        resultados = buscar_versiculos(entrada)
        st.success("Versículos encontrados para você meditar:")
        for _, row in resultados.iterrows():
            st.markdown(f"""
            <div class="verse">
                <strong>{row['livro']} {row['capitulo']}:{row['versiculo']}</strong><br>
                {row['texto']}
            </div>
            """, unsafe_allow_html=True)
