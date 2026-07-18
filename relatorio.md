# EEL891 — 2026.01 — Trabalho 2
## Regressão: estimativa de preço de imóveis a partir de suas características

**Aluno(a):** _[seu nome]_ — **Id no Kaggle:** _[seu id]_

---

## 1. O problema

O objetivo é estimar o preço de imóveis (Recife) a partir de características físicas (tipo, quartos, suítes, vagas, áreas), localização (bairro) e amenidades (piscina, churrasqueira, vista para o mar etc.). O conjunto de treinamento tem 4.683 anúncios com preço; o de teste, 2.000 sem preço. A métrica da competição é o **RMSPE** (raiz do erro percentual quadrático médio):

RMSPE = √( (1/n) · Σᵢ ((yᵢ − ŷᵢ)/yᵢ)² )

Duas propriedades da métrica orientaram todo o trabalho: (i) o erro é **relativo**, então errar R$100 mil num imóvel de R$200 mil custa 25× mais do que num de R$1 milhão; (ii) a métrica é **assimétrica** — dado um mesmo erro absoluto, superestimar custa mais que subestimar, e o preditor ótimo de um grupo de preços y₁…y_k não é a média, e sim **p\* = Σ(1/yᵢ) / Σ(1/yᵢ²)** (obtido derivando Σ((yᵢ−p)/yᵢ)² em p e igualando a zero), que é sempre menor que a média. Essas duas propriedades aparecem adiante no objetivo de treino customizado, no "match de duplicatas" e na calibração final.

## 2. Análise exploratória — três descobertas que definiram a solução

**(a) Erros de digitação no alvo e nas features.** O preço varia de R$750 a R$630.000.000 — ambos absurdos (o máximo é um apartamento de 98 m² em Casa Forte). Há também erros na área: apartamentos de 2 quartos com 569 m², 618 m², 1.019 m² (com preços típicos de apartamentos de ~60–100 m²), claramente área com dígito a mais. O preço por m² varia de R$13/m² a R$6,4 milhões/m².

**(b) Anúncios duplicados — inclusive entre treino e teste.** 1.061 linhas do treino (22,7%) têm ao menos uma outra linha com features idênticas (o mesmo imóvel anunciado mais de uma vez, às vezes com preços um pouco diferentes: a razão máx/mín mediana dentro dos grupos é 1,11). Mais importante: **433 das 2.000 linhas de teste (21,7%) têm match exato de todas as features no treino**, e 988 (49,4%) têm match numa chave reduzida (tipo, bairro, quartos, suítes, vagas, área útil, área extra). Ou seja, para metade do teste existe informação quase direta do preço no treino.

**(c) Bairro é o único sinal de localização — mas as amenidades identificam o prédio.** 66 bairros no treino; 3 bairros do teste não existem no treino. Dois anúncios do mesmo prédio compartilham bairro e o mesmo conjunto de amenidades de condomínio, mesmo sendo unidades diferentes — isso permite um encoding de "assinatura de prédio" mais fino que o bairro.

## 3. Pré-processamento

- **Filtro de preço:** mantive apenas linhas com preço em [R$10 mil, R$30 milhões] — remove 4 linhas com preço-typo flagrante (R$750, R$65M, R$340M, R$630M), cujo alvo é lixo irrecuperável.
- **Typos de área foram mantidos de propósito.** Testei removê-los (z-score robusto do log-preço/m² por bairro, vários limiares) e o resultado **piorou** na avaliação comparável (OOF avaliado sempre nas mesmas linhas, removendo outliers apenas do treino de cada fold): RMSPE 0,219 → 0,239–0,259. Motivo: os typos de área são *previsíveis* — muitos têm anúncios irmãos idênticos no treino — e o teste contém os mesmos typos, então o modelo precisa aprendê-los, não ignorá-los.
- **Categóricas:** `tipo` e `bairro` com dtype nativo de categoria (LightGBM/XGBoost) ou string (CatBoost); categorias fixadas sobre treino+teste.
- **`diferenciais`:** parsing da string em 8 flags binárias adicionais (copa, esquina, hidromassagem etc.) e contagem de diferenciais.

## 4. Engenharia de atributos (~46 features)

| Grupo | Features |
|---|---|
| Físicas | quartos, suítes, vagas, área útil, área extra, área total, log-áreas, razões (área/quarto, vagas/quarto, suítes/quarto, quartos/área, fração de área extra) |
| Amenidades | 10 flags originais + 8 extraídas de `diferenciais` + contagens |
| Vendedor | pessoa física vs. imobiliária (binária) |
| Bairro | target encoding (TE) do log-preço; TE do log-preço/m²; frequência do bairro |
| **Duplicatas** | Para 3 chaves (loose: tipo+bairro+quartos+área; red: + suítes, vagas, área extra; full: todas as features): TE do log-preço com suavização mínima (k=1) + contagem do grupo em treino+teste |
| **Prédio** | TE do log-preço/m² da assinatura (bairro + conjunto de amenidades) e de (bairro + string `diferenciais`) + contagem |

**Target encoding sem vazamento:** todos os TEs usam esquema K-fold — o valor de cada linha de treino é calculado só com as demais folds — com suavização m-estimate (média do grupo puxada para a média global proporcionalmente a k/(n+k)). Detalhe importante: o KFold interno do TE usa **o mesmo número de folds e a mesma seed do CV principal**, senão o TE de uma linha de treino contém alvo de linhas do fold de validação e o OOF fica otimista (bug sutil que detectei ao migrar de 5 para 10 folds).

## 5. Modelos

| Modelo | Alvo/objetivo | Observações |
|---|---|---|
| LightGBM ×3 (seeds 42/2027/777) | **Objetivo RMSPE customizado**: L = ((p−y)/y)², grad = 2(p−y)/y², hess = 2/y², reescalado por ŷ² médio para não esbarrar nos limiares de hessiana mínima | num_leaves=63, lr=0.02, colsample=0.6, min_child=15 (sweep confirmou como ótimo) |
| XGBoost | mesmo objetivo customizado | max_depth=6, hist, categóricas nativas |
| CatBoost | RMSE em preço cru com **sample_weight = 1/y²** — matematicamente idêntico a otimizar MSPE, e muito mais rápido que objetivo custom em Python | Antes usava RMSE em log-preço e rendia 0,230; com o peso caiu para ≈0,215 e ganhou ~19% do ensemble |
| ExtraTrees | log1p(preço) | membro de diversidade (~9–15% do blend) |

## 6. Validação

- **KFold com shuffle**, evoluindo 5 → 10 → 20 folds; OOF RMSPE como métrica de desenvolvimento. Todos os experimentos de features/hiperparâmetros foram comparados no mesmo esquema de folds.
- **Protocolo "honesto" para o pós-processamento:** qualquer hiperparâmetro ajustado sobre o próprio OOF (pesos do ensemble, pesos do match, calibração) infla o score. Para cada configuração eu reporto também o valor com *tuning na metade A das linhas e avaliação na metade B* (e vice-versa). A diferença chegou a 0,0015 — e o leaderboard confirmou sistematicamente o valor honesto, não o otimista.
- **Gap OOF→LB:** as linhas de teste incluem preços-typo imprevisíveis (como os R$630M do treino), que adicionam um erro fixo para todos os competidores. O gap ficou estável em ~+0,005/+0,008 ao longo de todas as submissões.

## 7. Ensemble e pós-processamento

1. **Média geométrica entre 9 seeds de pipeline** (CV/TE com seeds 42–50) para cada modelo — reduz variância de partição.
2. **Blend dos 6 modelos com pesos SLSQP separados por segmento** (linhas com match × sem match; simplex: pesos ≥0, soma 1, otimizados sobre o log das previsões OOF). O mix ótimo é bem diferente nos dois regimes: sem match domina o CatBoost com peso 1/y² (~0,54); com match domina o trio LightGBM — faz sentido, pois nas linhas com match a previsão do modelo será corrigida pelo preço dos irmãos, e o que importa é o comportamento *relativo* do resíduo.
3. **Match de duplicatas hierárquico (4 níveis):** para linhas de teste com grupo correspondente no treino (prioridade: full > red > loose > **noarea**), a previsão vira média geométrica ponderada entre o modelo e o estimador ótimo do grupo p\* = Σ(1/y)/Σ(1/y²) (robusto: em grupos ≥3, descarta o menor preço se < 0,45× a mediana — proteção contra irmão com typo). A chave noarea usa todas as features *exceto a área útil* — casa exatamente os anúncios cuja área tem dígito trocado (o typo mais comum do dataset) — e elevou a cobertura de match para 1.705 das 2.000 linhas de teste (85%). O peso do blend depende do nível da chave, do nº de irmãos em **6 buckets** (1 / 2 / 3+ irmãos × grupo apertado/disperso) — o tuning aprendeu peso ~zero para grupos dispersos (se os irmãos discordam entre si, confie no modelo).
4. **Calibração multiplicativa segmentada:** como o ótimo de RMSPE é menor que a média da distribuição preditiva, encolher as previsões reduz o erro esperado — mais onde a incerteza é maior. Fatores ajustados no OOF: ×0,988 para linhas com match, ×0,932 para linhas sem match.

## 8. Resultados

| Configuração | OOF honesto | LB público |
|---|---|---|
| Ensemble 4 modelos, 5 folds (baseline inicial) | 0,2175* | ≈0,225 |
| + TE de duplicatas, match, calibração (5 folds) | 0,2124 | 0,2181 |
| + assinatura de prédio, CatBoost 1/y², match hierárquico, 10 folds | 0,2095 | 0,2147 |
| + média de 3 seeds de pipeline | 0,2082 | 0,2138 |
| + 20 folds e calibração segmentada | 0,2060 | 0,2136 |
| + match de 3 níveis (loose) e 6 seeds | 0,2055 | 0,2136 |
| + 9 seeds | 0,2054 | 0,2136 |
| + pesos por segmento, chave noarea, 6 buckets (**final**) | 0,2041 | **0,2115** |

\* OOF simples (protocolo honesto adotado a partir da segunda linha).

O leaderboard público usa ~50% do teste (resolução ≈ ±0,001); a classificação final é calculada na outra metade. As submissões finais foram selecionadas pelo **OOF honesto** (melhor estimador do desempenho no teste completo), não pelo score público — escolher pelo público seleciona a submissão mais sortuda naquela metade, não a melhor. Na última linha o ganho público (−0,0021) veio maior que o ganho honesto (−0,0013): parte é sorte da metade pública, e o critério de seleção continua sendo o honesto.

**O que não funcionou** (todos avaliados no mesmo protocolo e descartados): remoção de outliers de área do treino; correção global de áreas-typo (÷10); features de vizinho-mais-próximo (kNN); stacking com meta-modelo GBM; LightGBM dart; regressão quantílica em log; piso no denominador do gradiente RMSPE; HistGradientBoosting como 7º membro; TEs de granularidade média (bairro×tipo×quartos); calibração segmentada por discordância entre modelos ou entre seeds; match de "prédio" via preço/m² da assinatura de amenidades como nível extra; harmonização de previsões entre duplicatas internas do teste (dispersão já pequena, ganho ≈ 0); teto de peso 1,0 no match (0,95 melhor); LightGBM com alvo preço/m² (o RMSPE é idêntico nos dois espaços, mas o modelo individual saiu ~0,002 pior). A lição dos folds: aumentar de 10→20 melhorou o OOF (−0,0022) muito mais que o LB (−0,0002) — com mais folds os TEs do OOF ficam mais fortes e a *medição* melhora, mas os TEs do teste sempre usam 100% do treino e não mudam. E a lição do sweep de hiperparâmetros: uma config que parecia −0,0013 num esquema de folds (colsample 0,35) **não replicou** em outros dois esquemas — diferenças de ±0,001 num único esquema de folds são ruído de partição; só a média sobre ≥3 esquemas separa sinal de sorte.

## 9. Reprodução

```
# 1. treina o ensemble e gera um dump de previsões por seed de CV
#    (20 folds; 9 seeds no total, ~30 min cada):
python scripts/train.py 42 20
python scripts/train.py 43 20
...
python scripts/train.py 50 20

# 2. combina os dumps, aplica match/calibração e gera a submissão final:
python scripts/posproc.py
# -> output/submissao_final.csv
```

Dependências: pandas, numpy, scipy, scikit-learn, lightgbm, xgboost, catboost.

## 10. Conclusões

O maior ganho do trabalho não veio de modelos maiores, e sim de **ler os dados**: perceber que metade do teste tem o mesmo imóvel anunciado no treino transformou o problema em parte em *matching*, e tratar a métrica RMSPE com rigor (objetivo de treino exato, estimador ótimo de grupo, calibração por encolhimento) rendeu ganhos consistentes que transferiram para o leaderboard. A segunda lição foi metodológica: com muitos knobs ajustados sobre a validação, só o protocolo de tuning-em-metade separou ganho real de ilusão — em todas as submissões, o leaderboard confirmou o número honesto.
