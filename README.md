# O que explica o preço de um apartamento em São Paulo?

**Checkpoint 4 — Regressão Linear e Polinomial**
Data Science & Statistical Computing — FIAP 2026

**Grupo:** Enzo Augusto (RM562249) · Rafaell Santiago (RM564386) · Gustavo Neres (RM561785) · Sebastian Iriarte (RM563619)

---

## Link do aplicativo

Aplicação publicada no Streamlit Community Cloud: _(https://datasciencecp04-dffzyxea6xdvggxcvndaxj.streamlit.app/)_

## Objetivo

Investigar quais características ajudam a explicar o preço de venda de apartamentos na
cidade de São Paulo e construir um modelo capaz de estimar esse preço.

**Pergunta de pesquisa:** em que medida a área, o número de banheiros, as vagas de garagem e
o distrito ajudam a explicar o preço de venda de um apartamento?

- **Variável resposta (y):** `preco` — preço de venda, em reais (R$)
- **Variáveis explicativas (X):** área (m²), banheiros, vagas de garagem e distrito

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
├── requirements.txt          # Dependências
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
exploratórios, as métricas do modelo final (MAE, RMSE e R²), os gráficos de preço real versus
previsto e de resíduos, e um formulário para simular o preço de um apartamento.

A entrada do usuário passa pelo **mesmo preparo** usado no treinamento (função
`montar_imovel`, idêntica à do notebook), e a aplicação avisa quando algum valor informado
está fora da faixa observada — tanto na base inteira quanto no distrito selecionado.

## Principais decisões de limpeza

| Problema encontrado | Investigação | Decisão | Linhas afetadas |
|---|---|---|---|
| Base mistura venda e aluguel | Coluna `Negotiation Type` | Manter apenas os imóveis à venda | 7.228 removidas |
| Colunas em inglês | — | Renomeadas para português | transformação |
| Distrito com sufixo "/São Paulo" | — | Texto padronizado | transformação |
| 215 linhas idênticas | Todas as colunas coincidem | Removidas (mesmo anúncio repetido) | 215 |
| Condomínio igual a zero | 1.247 imóveis (≈20%) sem a informação | Variável excluída do modelo | — |
| Preços muito altos | Verificamos que são imóveis de luxo em bairros nobres reais | **Mantidos** | 0 |

**Dimensões:** de **13.640 × 16** (original) para **6.197 × 6** (tratada), sem valores ausentes.

## Por que o número de quartos não entra no modelo

O coeficiente de `quartos` aparecia **negativo** nos primeiros testes, sugerindo que mais
quartos reduziria o preço. Como o resultado é contraintuitivo, investigamos antes de aceitá-lo
(seção 5 do notebook). Três verificações:

1. **O coeficiente é instável.** Varia de −R$ 74.337 a −R$ 28.876 conforme as demais
   variáveis do modelo, chegando a ficar positivo em outras especificações.
2. **O efeito negativo é de composição.** Controlando bairro e faixa de área ao mesmo tempo,
   o efeito **se inverte** — em Casa Verde, Bom Retiro e Brás, apartamentos de 50 a 90 m² com
   3 quartos têm mediana **maior** que os de 2 quartos.
3. **Não acrescenta poder preditivo.** O R² permanece em 0,827 com ou sem a variável.

Pelo **princípio da parcimônia**, optamos pelo modelo mais simples. A informação que o número
de quartos traria já está contida na área, nos banheiros e nas vagas — variáveis com as quais
ele tem correlação de 0,55 a 0,68.

> Isso **não** significa que o número de quartos seja irrelevante para o preço: isoladamente,
> tem correlação de 0,49. Significa que ele não acrescenta informação além do que as demais
> variáveis já explicam.

## Resumo dos modelos (conjunto de teste)

| Modelo | MAE (R$) | RMSE (R$) | R² |
|---|---|---|---|
| 1. Referência (média) | 406.550 | 736.138 | −0,001 |
| 2. Linear simples (área) | 187.338 | 406.068 | 0,695 |
| 3. Linear múltipla | 164.095 | 324.763 | 0,805 |
| **4. Polinomial (área ao quadrado)** | **155.386** | **306.041** | **0,827** |

O modelo final escolhido foi o **polinomial de grau 2**. A escolha não se baseou apenas no
maior R²: a curvatura entre área e preço já havia sido observada na análise exploratória, e o
ganho (RMSE cerca de 5,8% menor) foi verificado no conjunto de **teste**, indicando que não se
trata de sobreajuste.

**Coeficientes do modelo final:** área +R$ 3.532 por m² (mais o termo quadrático),
banheiros +R$ 74.611, vagas +R$ 239.590 — todos com sinal positivo.

## Principais limitações conhecidas

- Os dados vêm de **anúncios**, refletindo o preço pedido, não necessariamente o de venda.
- São de **abril de 2019**; o mercado imobiliário mudou desde então.
- A base não inclui informações relevantes como andar, idade do prédio, estado de
  conservação, área de lazer e vista.
- O modelo é bem menos preciso para imóveis de **alto padrão**, onde ocorrem os maiores erros.
- O diagnóstico indicou **heteroscedasticidade** (a dispersão dos erros cresce com o preço).
  Uma alternativa de tratamento seria aplicar logaritmo na variável resposta.
- O **efeito do distrito é aditivo, não multiplicativo**: a diferença entre dois bairros é
  sempre a mesma quantia em reais, independentemente do tamanho do imóvel.
- O modelo **não distingue** dois apartamentos de mesma área e mesmos banheiros com
  distribuições internas diferentes, já que o número de quartos foi excluído.
- Alguns distritos têm poucos imóveis na base, tornando suas estimativas instáveis.
- Trata-se de um estudo **observacional**: as relações encontradas são associações, não
  relações de causa e efeito.
