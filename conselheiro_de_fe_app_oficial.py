
import streamlit as st
import pandas as pd
from datetime import datetime
import urllib.parse
import locale
import re

# Configurar data em português brasileiro
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except:
    locale.setlocale(locale.LC_TIME, '')

@st.cache_data
def carregar_versiculos():
    return pd.read_csv("versiculos_biblia.csv")

@st.cache_data
def carregar_versiculo_do_dia():
    return pd.read_csv("versiculo_do_dia.csv")

df_versiculos = carregar_versiculos()
df_versiculo_dia = carregar_versiculo_do_dia()

def versiculo_do_dia():
    hoje = datetime.now()
    dia_ano = hoje.timetuple().tm_yday
    v = df_versiculo_dia[df_versiculo_dia["dia_do_ano"] == dia_ano].iloc[0]
    return f"{v['livro']} {v['capitulo']}:{v['versiculo']}", v["texto"]

def buscar_versiculos_simples(texto_usuario, top_n=5):
    palavras = re.findall(r'\b\w+\b', texto_usuario.lower())
    df_versiculos['pontuacao'] = df_versiculos['texto'].apply(
        lambda v: sum(1 for palavra in palavras if palavra in v.lower())
    )
    return df_versiculos.sort_values(by="pontuacao", ascending=False).head(top_n)

def gerar_contexto_ia(versiculo_texto):
    texto = versiculo_texto.lower()
    if "não temerás" in texto or "temor" in texto or "medo" in texto:
        return "Deus está dizendo que Ele protege e conforta nos momentos de medo."
    elif "confia" in texto or "fé" in texto or "crê" in texto:
        return "O versículo reforça a importância de confiar plenamente em Deus."
    elif "força" in texto or "fortalecer" in texto:
        return "Esse versículo lembra que nossa força vem do Senhor em momentos difíceis."
    elif "amor" in texto:
        return "Aqui, Deus demonstra Seu amor eterno por nós, mesmo em tempos de dor."
    elif "esperança" in texto or "futuro" in texto:
        return "Essa mensagem fala sobre manter a esperança no plano divino de Deus."
    else:
        return "Este versículo traz conforto e direção espiritual baseada na palavra de Deus."

# Layout
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

# Versículo do Dia
ref, texto = versiculo_do_dia()
data_pt = datetime.now().strftime("%d de %B de %Y")
st.markdown(f"### 📆 Versículo do Dia ({data_pt}):")
st.markdown(f"""<div class="verse">
<strong>{ref}</strong><br>
{texto}
</div>""", unsafe_allow_html=True)

# Botão WhatsApp
mensagem = f"{ref} – {texto}\nEnviado via Conselheiro de Fé 🙏"
url_whatsapp = f"https://wa.me/?text={urllib.parse.quote(mensagem)}"
st.markdown(f'<a href="{url_whatsapp}" target="_blank"><button style="background-color:#25D366; color:white; padding:10px 16px; border:none; border-radius:5px; font-size:16px;">📲 Compartilhar no WhatsApp</button></a>', unsafe_allow_html=True)

# Campo de busca personalizada
st.markdown("### Digite seu sentimento, dor ou gratidão e receba uma resposta da Bíblia Sagrada.")
entrada = st.text_area("🙏 Escreva aqui seu momento:")

if st.button("🔍 Buscar versículos"):
    if entrada.strip() == "":
        st.warning("Por favor, escreva algo para receber orientação espiritual.")
    else:
        resultados = buscar_versiculos_simples(entrada)
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
