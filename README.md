# O que explica o preço de um apartamento em São Paulo?

**Checkpoint 4 — Regressão Linear e Polinomial**
Data Science & Statistical Computing — FIAP 2026

**Grupo:** Enzo Augusto (RM562249) · Rafaell Santiago (RM564386) · Gustavo Neres (RM561785) · Sebastian Iriarte (RM563619)

---
## Link do aplicativo via Streamlit Community Cloud
[Aplicativo Streamlit](https://cp04datasciencegi-cew4whh9dp9ymqvozhu6pw.streamlit.app/)

## Objetivo

Investigar quais características ajudam a explicar o preço de venda de apartamentos na
cidade de São Paulo e construir um modelo capaz de estimar esse preço.

**Pergunta de pesquisa:** em que medida a área, o número de quartos, o número de banheiros,
as vagas de garagem e o distrito ajudam a explicar o preço de venda de um apartamento?

- **Variável resposta (y):** `preco` — preço de venda, em reais (R$)
- **Variáveis explicativas (X):** área (m²), quartos, banheiros, vagas de garagem e distrito

## Origem dos dados

- **Base:** *São Paulo Real Estate — Sale/Rent — April 2019*
- **Fonte:** Kaggle — https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019
- **Período de referência:** abril de 2019
- **Unidade de observação:** cada linha é um anúncio de imóvel
- **Dimensão original:** 13.640 anúncios e 16 colunas (venda e aluguel)
- **Uso:** base pública, disponibilizada para fins educacionais

O notebook lê os dados diretamente por URL, o que torna a análise reproduzível sem
necessidade de baixar arquivos manualmente.

> **Observação sobre a fonte:** os dados vêm de anúncios, ou seja, refletem o preço **pedido**
> pelo vendedor, que pode diferir do preço efetivamente praticado na venda.

## Estrutura dos arquivos

```
projeto/
├── app.py                    # Aplicação Streamlit
├── notebook.ipynb            # Análise completa (seções 1 a 10)
├── requirements.txt          # Dependências (com versões fixadas)
├── README.md                 # Este arquivo
├── base_tratada.csv          # Base após limpeza (gerada pelo notebook)
└── modelo_imoveis_sp.pkl     # Modelo final treinado (gerado pelo notebook)
```

## Instalação das dependências

```bash
pip install -r requirements.txt
```

## Execução do notebook

Abra `notebook.ipynb` no Google Colab ou no Jupyter e execute as células de cima para baixo.
Ao final, o notebook gera `base_tratada.csv` e `modelo_imoveis_sp.pkl`, usados pela aplicação.

## Execução da aplicação

Com `base_tratada.csv` e `modelo_imoveis_sp.pkl` na mesma pasta de `app.py`:

```bash
streamlit run app.py
```

A aplicação apresenta uma amostra da base, estatísticas descritivas, dois gráficos
exploratórios, as métricas do modelo final (MAE, RMSE e R²), os gráficos de preço real
versus previsto e de resíduos, e um formulário para simular o preço de um apartamento.

A entrada do usuário passa pelo **mesmo preparo** usado no treinamento (função
`montar_imovel`, idêntica à do notebook).

A aplicação emite **dois níveis de aviso de extrapolação**: um quando o valor informado está
fora da faixa de toda a base, e outro quando a combinação está dentro das faixas gerais mas
**não existe naquele distrito**. Por exemplo, o maior apartamento do Brás na base tem 196 m²
e 2 vagas: simular 400 m² com 5 vagas ali dispara o alerta, mesmo estando dentro da faixa
global de 30 a 620 m².

## Principais decisões de limpeza

| Problema encontrado | Investigação | Decisão | Linhas afetadas |
|---|---|---|---|
| Base mistura venda e aluguel | Coluna `Negotiation Type` | Manter apenas os imóveis à venda | 7.228 removidas |
| Colunas em inglês | — | Renomeadas para português | transformação |
| Distrito com sufixo "/São Paulo" | — | Texto padronizado | transformação |
| 215 linhas idênticas | Todas as colunas coincidem | Removidas (mesmo anúncio repetido) | 215 |
| Condomínio igual a zero | 1.247 imóveis (≈20%) sem a informação | Variável excluída do modelo | — |
| Preços muito altos | Verificamos que são imóveis de luxo em bairros nobres reais | **Mantidos** (fazem parte do mercado) | 0 |

**Dimensões:** de **13.640 × 16** (original) para **6.197 × 6** (tratada), sem valores ausentes.

## Resumo dos modelos (conjunto de teste)

| Modelo | MAE (R$) | RMSE (R$) | R² |
|---|---|---|---|
| 1. Referência (média) | 406.550 | 736.138 | −0,001 |
| 2. Linear simples (área) | 187.338 | 406.068 | 0,695 |
| 3. Linear múltipla | 162.471 | 320.744 | 0,810 |
| **4. Polinomial (área ao quadrado)** | **155.457** | **306.156** | **0,827** |

O modelo final escolhido foi o **polinomial de grau 2**. A escolha não se baseou apenas no
maior R²: a curvatura entre área e preço já havia sido observada na análise exploratória, e
o ganho (RMSE cerca de 4,5% menor) foi verificado no conjunto de **teste**, indicando que
não se trata de sobreajuste.

## Principais limitações conhecidas

- Os dados vêm de **anúncios**, refletindo o preço pedido, não necessariamente o de venda.
- São de **abril de 2019**; o mercado imobiliário mudou desde então.
- A base não inclui informações relevantes como andar, idade do prédio, estado de
  conservação, área de lazer e vista.
- O modelo é bem menos preciso para imóveis de **alto padrão**, onde ocorrem os maiores erros.
- O diagnóstico indicou **heteroscedasticidade** (a dispersão dos erros cresce com o preço).
- Alguns distritos têm poucos imóveis na base, tornando suas estimativas instáveis.
- **O efeito do distrito é aditivo, e não multiplicativo.** O distrito entra no modelo
  somando um valor fixo em reais, então a diferença entre dois bairros é sempre a mesma
  quantia, independentemente do tamanho do imóvel. Entre Brás e Brooklin, por exemplo, a
  diferença prevista é de cerca de R$ 205 mil nos dois casos — o que representa 44% num
  apartamento de 70 m², mas apenas 2,5% num de 620 m². No mercado real a localização
  multiplica o preço: o preço mediano por m² do Brooklin (R$ 10.618) é 75% maior que o do
  Brás (R$ 6.070) em qualquer tamanho.

**O que faríamos diferente.** Aplicar logaritmo na variável resposta resolveria os dois
últimos pontos de uma vez: estabilizaria a variância dos erros e faria o efeito dos distritos
multiplicar o preço em vez de somar um valor fixo. Essa transformação está fora do escopo
desta atividade, mas é o caminho natural para uma versão futura do modelo.
- Trata-se de um estudo **observacional**: as relações encontradas são associações, não
  relações de causa e efeito.
