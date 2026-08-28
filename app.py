"""
Aplicação Streamlit — Previsão do preço de apartamentos em São Paulo
Checkpoint 4 — Data Science & Statistical Computing — FIAP 2026

Grupo: Enzo Augusto (RM562249), Rafaell Santiago (RM564386),
       Gustavo Neres (RM561785), Sebastian Iriarte (RM563619)

Como executar:
    streamlit run app.py

O modelo usado aqui é o mesmo treinado no notebook (modelo_imoveis_sp.pkl).
A função montar_imovel() é idêntica à do notebook, garantindo que a entrada do
usuário passe exatamente pelo mesmo preparo usado no treinamento.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Preço de apartamentos em SP", page_icon="🏢", layout="wide")


# ------------------------------------------------------------------
# Carregamento dos dados e do modelo (com cache, como pede o enunciado)
# ------------------------------------------------------------------
@st.cache_data
def carregar_base():
    return pd.read_csv("base_tratada.csv")


@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_imoveis_sp.pkl")


df = carregar_base()
pacote = carregar_modelo()

modelo = pacote["modelo"]
colunas_modelo = pacote["colunas"]
distritos = pacote["distritos"]
faixas = pacote["faixas"]
faixas_distrito = pacote["faixas_distrito"]


# ------------------------------------------------------------------
# Mesma função de preparo usada no notebook
# ------------------------------------------------------------------
def montar_imovel(area_m2, quartos, banheiros, vagas, distrito, colunas_modelo):
    linha = pd.DataFrame([{
        "area_m2": area_m2,
        "quartos": quartos,
        "banheiros": banheiros,
        "vagas": vagas,
        "distrito": distrito,
    }])
    linha = pd.get_dummies(linha, columns=["distrito"])
    linha["area_quadrado"] = linha["area_m2"] ** 2
    return linha.reindex(columns=colunas_modelo, fill_value=0)


# ------------------------------------------------------------------
# Refaz a separação treino/teste para mostrar métricas e gráficos
# (usa o MESMO random_state do notebook)
# ------------------------------------------------------------------
y = df["preco"]
X = df[["area_m2", "quartos", "banheiros", "vagas", "distrito"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_test_p = pd.get_dummies(X_test, columns=["distrito"], drop_first=True)
X_test_p["area_quadrado"] = X_test_p["area_m2"] ** 2
X_test_p = X_test_p.reindex(columns=colunas_modelo, fill_value=0)

previsoes = modelo.predict(X_test_p)
residuos = y_test.values - previsoes


# ------------------------------------------------------------------
# Cabeçalho
# ------------------------------------------------------------------
st.title("🏢 Quanto vale um apartamento em São Paulo?")

st.markdown(
    """
Esta aplicação estima o **preço de venda de um apartamento em São Paulo** a partir de suas
características, usando um modelo de **regressão polinomial**.

**Problema:** em que medida a área, o número de quartos, o número de banheiros, as vagas de
garagem e o distrito ajudam a explicar o preço de venda de um apartamento?

- **Variável resposta:** preço de venda, em **reais (R$)**
- **Variáveis usadas na previsão:** área (m²), quartos, banheiros, vagas e distrito
- **Fonte dos dados:** *São Paulo Real Estate — Sale/Rent — April 2019* (Kaggle), anúncios
  de imóveis da cidade de São Paulo, referentes a abril de 2019.

> Os dados são de **anúncios**, ou seja, refletem o preço pedido pelo vendedor.
"""
)

st.divider()

# ------------------------------------------------------------------
# 1. Base de dados
# ------------------------------------------------------------------
st.header("1. A base de dados")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Amostra da base tratada")
    st.dataframe(df.head(10), use_container_width=True)
with col_b:
    st.subheader("Estatísticas descritivas")
    st.dataframe(df.describe().round(0), use_container_width=True)

st.caption(f"Base tratada com {len(df):,} apartamentos à venda.".replace(",", "."))

st.divider()

# ------------------------------------------------------------------
# 2. Gráficos exploratórios
# ------------------------------------------------------------------
st.header("2. Exploração dos dados")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Preço vs. área")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["area_m2"], df["preco"], alpha=0.3)
    ax1.set_xlabel("Área (m²)")
    ax1.set_ylabel("Preço (R$)")
    ax1.set_title("Apartamentos maiores custam mais")
    ax1.grid(alpha=0.25)
    st.pyplot(fig1)
    st.caption("Relação positiva, com curvatura nos imóveis maiores.")

with col2:
    st.subheader("Preço mediano por distrito")
    top10 = df["distrito"].value_counts().head(10).index
    resumo = df[df["distrito"].isin(top10)].groupby("distrito")["preco"].median().sort_values()
    fig2, ax2 = plt.subplots()
    ax2.barh(resumo.index, resumo.values)
    ax2.set_xlabel("Preço mediano (R$)")
    ax2.set_title("A localização pesa muito no preço")
    ax2.grid(alpha=0.25)
    st.pyplot(fig2)
    st.caption("Dez distritos com mais imóveis na base.")

st.divider()

# ------------------------------------------------------------------
# 3. Desempenho do modelo
# ------------------------------------------------------------------
st.header("3. Desempenho do modelo final (conjunto de teste)")

mae = mean_absolute_error(y_test, previsoes)
rmse = np.sqrt(mean_squared_error(y_test, previsoes))
r2 = r2_score(y_test, previsoes)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"R$ {mae:,.0f}".replace(",", "."))
m2.metric("RMSE", f"R$ {rmse:,.0f}".replace(",", "."))
m3.metric("R²", f"{r2:.3f}")

st.caption(
    "O MAE indica o erro médio do modelo. O RMSE é maior porque penaliza mais os erros "
    "grandes, que ocorrem principalmente em imóveis de alto padrão."
)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Preço real vs. previsto")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, previsoes, alpha=0.3)
    limite = [y_test.min(), y_test.max()]
    ax3.plot(limite, limite, "r--", label="Previsão perfeita")
    ax3.set_xlabel("Preço real (R$)")
    ax3.set_ylabel("Preço previsto (R$)")
    ax3.legend()
    ax3.grid(alpha=0.25)
    st.pyplot(fig3)

with col4:
    st.subheader("Resíduos")
    fig4, ax4 = plt.subplots()
    ax4.scatter(previsoes, residuos, alpha=0.3)
    ax4.axhline(0, color="red", linestyle="--")
    ax4.set_xlabel("Preço previsto (R$)")
    ax4.set_ylabel("Resíduo (real - previsto)")
    ax4.grid(alpha=0.25)
    st.pyplot(fig4)

st.divider()

# ------------------------------------------------------------------
# 4. Formulário de previsão
# ------------------------------------------------------------------
st.header("4. Simule o preço de um apartamento")

with st.form("formulario"):
    c1, c2, c3 = st.columns(3)

    with c1:
        area = st.slider("Área (m²)", 30, 620, 70, step=5)
        quartos = st.selectbox("Quartos", [1, 2, 3, 4, 5, 6], index=1)

    with c2:
        banheiros = st.selectbox("Banheiros", [1, 2, 3, 4, 5, 6, 7], index=1)
        vagas = st.selectbox("Vagas de garagem", [0, 1, 2, 3, 4, 5, 6, 7, 8], index=1)

    with c3:
        distrito = st.selectbox("Distrito", distritos)

    enviar = st.form_submit_button("Prever preço")

if enviar:
    entrada = montar_imovel(area, quartos, banheiros, vagas, distrito, colunas_modelo)
    preco = float(modelo.predict(entrada)[0])

    st.success(f"### Preço estimado: **R$ {preco:,.0f}**".replace(",", "."))

    # Aviso quando a entrada está fora do intervalo observado
    valores = {"area_m2": area, "quartos": quartos, "banheiros": banheiros, "vagas": vagas}
    nomes = {"area_m2": "Área (m²)", "quartos": "Quartos", "banheiros": "Banheiros", "vagas": "Vagas"}

    # Nível 1: fora da faixa observada em TODA a base
    fora_global = []
    for variavel, (minimo, maximo) in faixas.items():
        v = valores[variavel]
        if v < minimo or v > maximo:
            fora_global.append(f"- **{nomes[variavel]}** = {v} (faixa da base: {minimo:.0f} a {maximo:.0f})")

    # Nível 2: dentro das faixas gerais, mas fora do que existe NAQUELE distrito
    fora_distrito = []
    for variavel, (minimo, maximo) in faixas_distrito.get(distrito, {}).items():
        v = valores[variavel]
        if v < minimo or v > maximo:
            fora_distrito.append(
                f"- **{nomes[variavel]}** = {v} (faixa em {distrito}: {minimo:.0f} a {maximo:.0f})"
            )

    if fora_global:
        st.error(
            "🚫 Uma ou mais entradas estão **fora da faixa observada em toda a base**. "
            "A previsão é uma **extrapolação** e não deve ser considerada confiável:\n\n"
            + "\n".join(fora_global)
        )
    elif fora_distrito:
        st.warning(
            f"⚠️ As entradas estão dentro das faixas gerais da base, mas **não existe nenhum "
            f"apartamento com essas características em {distrito}**. A previsão extrapola o "
            "que foi observado nesse distrito:\n\n"
            + "\n".join(fora_distrito)
            + "\n\nComo o modelo soma um valor fixo em reais para cada distrito, o resultado "
            "fica cada vez menos confiável conforme se afasta do porte típico do bairro."
        )
    else:
        st.info(f"✅ Todas as entradas estão dentro da faixa observada em {distrito}.")

    n_distrito = int((df["distrito"] == distrito).sum())
    if n_distrito < 40:
        st.caption(
            f"ℹ️ {distrito} tem apenas {n_distrito} imóveis na base. "
            "Estimativas para distritos com poucas observações são menos estáveis."
        )

    if preco < 0:
        st.error(
            "O modelo retornou um valor negativo, o que não faz sentido na prática. "
            "Isso acontece em combinações muito incomuns de características."
        )

st.divider()
st.caption(
    "Modelo: regressão polinomial de grau 2 (área elevada ao quadrado). "
    "Base: anúncios de imóveis de São Paulo, abril de 2019. "
    "Estudo observacional: associação não implica causalidade."
)
