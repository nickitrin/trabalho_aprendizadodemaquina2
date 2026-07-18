"""
EEL891 2026.01 - Trabalho 2 - Pos-processamento e submissao final

Consome os dumps de previsoes gerados por train.py (um por seed de CV) e:
  1. Otimiza os pesos do ensemble (SLSQP em log-preco) sobre o OOF
  2. Aplica o "match de duplicatas" hierarquico (full-key > red-key)
  3. Aplica calibracao multiplicativa segmentada (com match / sem match)
  4. Gera output/submissao_final.csv

Os hiperparametros do pos-processamento sao validados com protocolo
"honesto": ajusta na metade A das linhas, avalia na metade B e vice-versa.
O valor reportado como honesto e a media das duas avaliacoes; a submissao
final usa os parametros ajustados em todas as linhas.

Uso:
    python scripts/posproc.py                      (usa os 3 dumps 20-fold)
    python scripts/posproc.py dump1.npz dump2.npz  (dumps especificos)
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from train import (
    engineer_features, add_bairro_target_encoding, add_duplicate_features,
    add_building_features, rmspe, TRAIN_PATH, TEST_PATH,
)

DEFAULT_DUMPS = ["output/preds_dump_f20.npz"] + [
    f"output/preds_dump_s{s}_f20.npz" for s in range(43, 51)
]
OUTPUT = "output/submissao_final.csv"
# match hierarquico: full > red > loose > noarea (loose cobre "mesmo
# predio/area, vagas ou suites divergentes"; noarea casa linhas cuja
# area_util tem typo, ja que todas as demais features batem)
MATCH_KEYS = ["k_full", "k_red", "k_loose", "k_noarea"]
NOAREA_COLS = ["tipo", "bairro", "quartos", "suites", "vagas", "area_extra"]
TIGHT_THR = 0.10  # dispersao (std do log-preco) que separa grupo "apertado"


# ----------------------------------------------------------------- dados
def build_keys():
    """Reconstroi as chaves de duplicata (identicas as do train.py)."""
    tr = pd.read_csv(TRAIN_PATH)
    te = pd.read_csv(TEST_PATH)
    tr = tr[(tr["preco"] >= 10_000) & (tr["preco"] <= 30_000_000)].reset_index(drop=True)
    train_fe = engineer_features(tr)
    test_fe = engineer_features(te)
    train_fe["log_preco"] = np.log1p(train_fe["preco"])
    train_fe, test_fe = add_bairro_target_encoding(train_fe, test_fe, "log_preco")
    train_fe, test_fe = add_duplicate_features(train_fe, test_fe)
    train_fe, test_fe = add_building_features(train_fe, test_fe)
    for df in (train_fe, test_fe):
        df["k_noarea"] = df[NOAREA_COLS].astype(str).agg("|".join, axis=1)
    return train_fe, test_fe


# ----------------------------------------------------------------- match
def robust_group_stats(prices):
    """Estimador otimo de RMSPE de um grupo: p* = sum(1/y)/sum(1/y^2).
    Robusto a typo: em grupos >=3, descarta o menor preco se < 0.45x mediana."""
    p = np.asarray(prices, dtype=float)
    if len(p) >= 3:
        i_min = int(np.argmin(p))
        if p[i_min] < 0.45 * np.median(p):
            p = np.delete(p, i_min)
    s1 = (1.0 / p).sum()
    s2 = (1.0 / p ** 2).sum()
    return s1 / s2, len(p), float(np.std(np.log(p)))


def build_match(train_fe, test_fe, y, key):
    """Match LOO para as linhas de treino e match completo para o teste."""
    n = len(y)
    df = pd.DataFrame({"k": train_fe[key].values, "y": y})
    groups = df.groupby("k")["y"].apply(list)
    idx_by_key = df.groupby("k").indices

    m_oof = np.full(n, np.nan)
    sib_oof = np.zeros(n)
    std_oof = np.zeros(n)
    for k, idxs in idx_by_key.items():
        prices = np.asarray(groups[k], dtype=float)
        if len(prices) < 2:
            continue
        for j, i in enumerate(idxs):
            p, cnt, sd = robust_group_stats(np.delete(prices, j))
            m_oof[i], sib_oof[i], std_oof[i] = p, cnt, sd

    kte = test_fe[key].values
    m_te = np.full(len(kte), np.nan)
    sib_te = np.zeros(len(kte))
    std_te = np.zeros(len(kte))
    gstats = {k: robust_group_stats(v) for k, v in groups.items()}
    for i, k in enumerate(kte):
        if k in gstats:
            m_te[i], sib_te[i], std_te[i] = gstats[k]
    return (m_oof, sib_oof, std_oof), (m_te, sib_te, std_te)


def bucket(sib, std):
    """(1 / 2 / 3+ irmaos) x (apertado / disperso) -> 6 buckets"""
    sb = np.where(sib >= 3, 2, np.where(sib >= 2, 1, 0))
    return (sb * 2 + np.where(std > TIGHT_THR, 1, 0)).astype(int)


def apply_match(pred, stats_by_key, W):
    out = np.log(pred.copy())
    done = np.zeros(len(pred), dtype=bool)
    for key in MATCH_KEYS:  # full tem prioridade sobre red
        m, sib, std = stats_by_key[key]
        has = ~np.isnan(m) & ~done
        if has.any():
            w = W[key][bucket(sib[has], std[has])]
            out[has] = w * np.log(m[has]) + (1 - w) * out[has]
            done |= has
    return np.exp(out)


def tune_match(pred, y, match_oof, mask, n_rounds=3):
    W = {k: np.full(6, 0.25) for k in MATCH_KEYS}
    grid = np.linspace(0, 0.95, 20)
    for _ in range(n_rounds):
        for key in MATCH_KEYS:
            for bi in range(6):
                best_r, best_w = np.inf, W[key][bi]
                for w in grid:
                    W[key][bi] = w
                    r = rmspe(y[mask], apply_match(pred, match_oof, W)[mask])
                    if r < best_r:
                        best_r, best_w = r, w
                W[key][bi] = best_w
    return W


# ----------------------------------------------------------------- chain
def slsqp_weights(oof_members, y, mask):
    names = list(oof_members)
    P = np.column_stack([np.log(oof_members[m]) for m in names])

    def obj(w):
        return rmspe(y[mask], np.exp(P[mask] @ w))

    w0 = np.full(len(names), 1 / len(names))
    res = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)] * len(names),
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"maxiter": 300})
    return res.x


def chain(oof_m, test_m, y, match_oof, match_te, seg_oof, seg_te, mask):
    # pesos SLSQP separados para linhas com match e sem match: o mix otimo
    # de modelos e diferente quando a previsao vai ser corrigida pelo match
    names = list(oof_m)
    P_oof = np.column_stack([np.log(oof_m[m]) for m in names])
    P_te = np.column_stack([np.log(test_m[m]) for m in names])
    p_oof = np.zeros(len(y))
    p_te = np.zeros(P_te.shape[0])
    wb = []
    for so, st in [(seg_oof, seg_te), (~seg_oof, ~seg_te)]:
        w = slsqp_weights(oof_m, y, mask & so)
        p_oof[so] = np.exp(P_oof[so] @ w)
        p_te[st] = np.exp(P_te[st] @ w)
        wb.append(w)
    W = tune_match(p_oof, y, match_oof, mask)
    p_oof = apply_match(p_oof, match_oof, W)
    p_te = apply_match(p_te, match_te, W)
    ss = np.linspace(0.93, 1.03, 41)
    svals = []
    for seg in (seg_oof, ~seg_oof):
        m = seg & mask
        svals.append(ss[int(np.argmin([rmspe(y[m], s * p_oof[m]) for s in ss]))])
    for so, st, s in zip((seg_oof, ~seg_oof), (seg_te, ~seg_te), svals):
        p_oof[so] *= s
        p_te[st] *= s
    return p_oof, p_te, wb, W, svals


def main():
    dump_paths = sys.argv[1:] or DEFAULT_DUMPS
    dumps = [np.load(p, allow_pickle=True) for p in dump_paths]
    y = dumps[0]["y"]
    models = [str(m) for m in dumps[0]["model_names"]]
    for d in dumps[1:]:
        assert np.allclose(d["y"], y), "dumps com filtros de treino diferentes"

    # media geometrica das seeds, por modelo
    oof_m = {m: np.exp(np.mean([np.log(d[f"oof_{m}"]) for d in dumps], axis=0)) for m in models}
    test_m = {m: np.exp(np.mean([np.log(d[f"test_{m}"]) for d in dumps], axis=0)) for m in models}
    for m in models:
        print(f"OOF {m:<5} (media {len(dumps)} seeds): {rmspe(y, oof_m[m]):.4f}")

    print("reconstruindo chaves de duplicata...")
    train_fe, test_fe = build_keys()
    match_oof = {}
    match_te = {}
    for key in MATCH_KEYS:
        match_oof[key], match_te[key] = build_match(train_fe, test_fe, y, key)
    seg_oof = np.any([~np.isnan(match_oof[k][0]) for k in MATCH_KEYS], axis=0)
    seg_te = np.any([~np.isnan(match_te[k][0]) for k in MATCH_KEYS], axis=0)

    # protocolo honesto (2-fold nos hiperparametros do pos-processamento)
    n = len(y)
    half = np.random.RandomState(123).permutation(n) < n // 2
    rs = []
    for mt in (half, ~half):
        po, _, _, _, _ = chain(oof_m, test_m, y, match_oof, match_te, seg_oof, seg_te, mt)
        rs.append(rmspe(y[~mt], po[~mt]))
    print(f"OOF honesto (2-fold): {np.mean(rs):.4f}")

    full = np.ones(n, dtype=bool)
    p_oof, p_te, wb, W, svals = chain(oof_m, test_m, y, match_oof, match_te, seg_oof, seg_te, full)
    print(f"OOF tuned-on-all: {rmspe(y, p_oof):.4f}")
    for label, w in zip(("com match", "sem match"), wb):
        print(f"pesos do ensemble ({label}):", {m: round(float(x), 3) for m, x in zip(models, w)})
    for k in MATCH_KEYS:
        print(f"pesos match {k} (1s-ap, 1s-di, 2-ap, 2-di, 3+ap, 3+di): {np.round(W[k], 2)}")
    print(f"calibracao (com match, sem match): {np.round(svals, 3)}")

    p_te = np.clip(p_te, 10_000, None)
    pd.DataFrame({"Id": dumps[0]["ids_test"], "preco": p_te}).to_csv(OUTPUT, index=False)
    print(f"submissao salva em {OUTPUT}")


if __name__ == "__main__":
    main()
