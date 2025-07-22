
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def carregar_versiculos():
    df = pd.read_csv("versiculos_biblia.csv")
    return df

df_versiculos = carregar_versiculos()
versiculos_texto = df_versiculos["texto"].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(versiculos_texto)

def buscar_versiculos(entrada_usuario, top_n=5):
    entrada_vetor = vectorizer.transform([entrada_usuario])
    similaridades = cosine_similarity(entrada_vetor, tfidf_matrix).flatten()
    indices_mais_similares = similaridades.argsort()[::-1][:top_n]
    return df_versiculos.iloc[indices_mais_similares]

def gerar_contexto_ia(versiculo_texto):
    texto = versiculo_texto.lower()
    if "não temerás" in texto or "temor" in texto:
        return "Deus está dizendo que Ele protege e conforta nos momentos de medo."
    elif "confia" in texto or "fé" in texto:
        return "O versículo reforça a importância de confiar plenamente em Deus."
    elif "força" in texto or "fortalecer" in texto:
        return "Esse versículo lembra que nossa força vem do Senhor em momentos difíceis."
    elif "amor" in texto:
        return "Aqui, Deus demonstra Seu amor eterno por nós, mesmo em tempos de dor."
    elif "esperança" in texto or "futuro" in texto:
        return "Essa mensagem fala sobre manter a esperança no plano divino de Deus."
    else:
        return "Este versículo traz conforto e direção espiritual baseada na palavra de Deus."

st.set_page_config(page_title="Conselheiro de Fé", page_icon="📖", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #3b2f2f;
        color: #f0e6d2;
        font-family: 'Georgia', serif;
    }
    .title {
        color: #f7c873;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .verse {
        background-color: #5a4338;
        border-left: 6px solid #deb887;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    .contexto {
        font-size: 0.9rem;
        color: #f7e6c8;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📖 Conselheiro de Fé</div>', unsafe_allow_html=True)
st.markdown("### Digite seu sentimento, dor ou gratidão e receba uma resposta da Bíblia Sagrada.")

entrada = st.text_area("🙏 Escreva aqui seu momento:")

if st.button("🔍 Buscar versículos"):
    if entrada.strip() == "":
        st.warning("Por favor, escreva algo para receber orientação espiritual.")
    else:
        resultados = buscar_versiculos(entrada)
        st.success("Versículos encontrados para você meditar:")
        for _, row in resultados.iterrows():
            contexto = gerar_contexto_ia(row['texto'])
            st.markdown(f"""
            <div class="verse">
                <strong>{row['livro']} {row['capitulo']}:{row['versiculo']}</strong><br>
                {row['texto']}<br>
                <div class="contexto"><em>📌 Contexto: {contexto}</em></div>
            </div>
            """, unsafe_allow_html=True)
