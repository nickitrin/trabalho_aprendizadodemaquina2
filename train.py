"""
EEL891 2026.01 - Trabalho 2 - Regressao de precos de imoveis
Metrica: RMSPE (raiz do erro percentual quadratico medio)

Pipeline:
  1. Leitura dos dados
  2. Remocao de outliers grosseiros (erros de digitacao evidentes no preco)
  3. Engenharia de atributos (diferenciais, areas, encoding de bairro)
  4. Validacao cruzada (KFold) medindo RMSPE de verdade
  5. Ensemble de CatBoost + LightGBM + XGBoost treinados em log1p(preco)
  6. Treino final com todos os dados e geracao do arquivo de submissao
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
N_SPLITS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
# seed do CV/TE via CLI (padrao 42); os TEs precisam seguir a MESMA seed
# do KFold principal para nao vazar alvo de fold de validacao
CV_SEED = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else RANDOM_STATE

TRAIN_PATH = "conjunto_de_treinamento.csv"
TEST_PATH = "conjunto_de_teste.csv"
OUTPUT_PATH = "output/submissao.csv"

DIFERENCIAIS_EXTRA = {
    "copa": "copa",
    "esquina": "esquina",
    "children care": "children_care",
    "hidromassagem": "hidromassagem",
    "quadra de squash": "quadra_squash",
    "vestiario": "vestiario",
    "campo de futebol": "campo_futebol",
    "quadra poliesportiva": "quadra_poliesportiva",
}


def print_progress(done, total, label, start_time, bar_len=30):
    frac = done / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.time() - start_time
    eta = (elapsed / done * (total - done)) if done > 0 else 0
    sys.stdout.write(
        f"\r[{bar}] {done}/{total} ({frac*100:5.1f}%) {label:<28} "
        f"decorrido={elapsed:6.1f}s  eta={eta:6.1f}s   "
    )
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for token, col in DIFERENCIAIS_EXTRA.items():
        df[col] = df["diferenciais"].str.contains(token, regex=False).astype(int)

    df["n_diferenciais"] = df["diferenciais"].apply(
        lambda s: 0 if s == "nenhum" else len(s.split(" e "))
    )

    df["area_total"] = df["area_util"] + df["area_extra"]
    df["area_extra_ratio"] = df["area_extra"] / (df["area_total"] + 1)
    df["log_area_util"] = np.log1p(df["area_util"])
    df["log_area_total"] = np.log1p(df["area_total"])

    df["quartos_por_area"] = df["quartos"] / (df["area_util"] + 1)
    df["vagas_por_quarto"] = df["vagas"] / (df["quartos"] + 1)
    df["suites_por_quarto"] = df["suites"] / (df["quartos"] + 1)
    df["area_por_quarto"] = df["area_util"] / (df["quartos"] + 1)

    df["n_amenidades"] = df[
        [
            "churrasqueira",
            "estacionamento",
            "piscina",
            "playground",
            "quadra",
            "s_festas",
            "s_jogos",
            "s_ginastica",
            "sauna",
            "vista_mar",
        ]
    ].sum(axis=1)

    df["tipo_vendedor_bin"] = (df["tipo_vendedor"] == "Pessoa Fisica").astype(int)

    return df


def _smoothed_group_mean(sub_df, group_col, target_col, global_mean, k=15):
    stats = sub_df.groupby(group_col)[target_col].agg(["mean", "count"])
    return (stats["mean"] * stats["count"] + global_mean * k) / (stats["count"] + k)


def add_target_encoding(train_df, test_df, group_col, target_col, out_col, n_splits=None, seed=RANDOM_STATE, k=15):
    n_splits = n_splits or N_SPLITS
    """K-Fold target encoding com suavizacao (m-estimate) para reduzir
    ruido em categorias com poucas observacoes, sem vazamento."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_df))
    for tr_idx, val_idx in kf.split(train_df):
        fold_means = _smoothed_group_mean(train_df.iloc[tr_idx], group_col, target_col, global_mean, k)
        oof[val_idx] = train_df.iloc[val_idx][group_col].map(fold_means).fillna(global_mean)
    train_df[out_col] = oof

    full_means = _smoothed_group_mean(train_df, group_col, target_col, global_mean, k)
    test_df[out_col] = test_df[group_col].map(full_means).fillna(global_mean)

    return train_df, test_df


def add_bairro_target_encoding(train_df, test_df, target_col, n_splits=None, seed=RANDOM_STATE):
    n_splits = n_splits or N_SPLITS
    train_df, test_df = add_target_encoding(
        train_df, test_df, "bairro", target_col, "bairro_te", n_splits=n_splits, seed=seed, k=15
    )

    bairro_count = train_df["bairro"].value_counts()
    train_df["bairro_freq"] = train_df["bairro"].map(bairro_count).fillna(0)
    test_df["bairro_freq"] = test_df["bairro"].map(bairro_count).fillna(0)

    # Encoding suavizado do preco por m2 (log) do bairro: captura o "premio
    # de localizacao" de forma mais independente da area do imovel.
    train_df["log_ppm2"] = np.log1p(train_df["preco"] / train_df["area_util"].clip(lower=1))
    train_df, test_df = add_target_encoding(
        train_df, test_df, "bairro", "log_ppm2", "bairro_ppm2_te", n_splits=n_splits, seed=seed, k=15
    )
    train_df = train_df.drop(columns=["log_ppm2"])

    return train_df, test_df


NUMERIC_FEATURES = [
    "quartos", "suites", "vagas", "area_util", "area_extra",
    "churrasqueira", "estacionamento", "piscina", "playground", "quadra",
    "s_festas", "s_jogos", "s_ginastica", "sauna", "vista_mar",
    "copa", "esquina", "children_care", "hidromassagem", "quadra_squash",
    "vestiario", "campo_futebol", "quadra_poliesportiva",
    "n_diferenciais", "area_total", "area_extra_ratio",
    "log_area_util", "log_area_total",
    "quartos_por_area", "vagas_por_quarto", "suites_por_quarto", "area_por_quarto",
    "n_amenidades", "tipo_vendedor_bin", "bairro_te", "bairro_freq", "bairro_ppm2_te",
    "te_loose", "cnt_loose", "te_red", "cnt_red", "te_full", "cnt_full",
    "bld_ppm2_te", "bld_cnt", "blddif_ppm2_te",
]
CAT_FEATURES = ["tipo", "bairro"]

# Chaves de "anuncio duplicado": muitos imoveis aparecem varias vezes no
# dataset (e ~metade do teste tem match no treino). O TE dessas chaves com
# suavizacao minima injeta o preco dos anuncios irmaos como feature.
DUP_KEYS = {
    "loose": ["tipo", "bairro", "quartos", "area_util"],
    "red": ["tipo", "bairro", "quartos", "suites", "vagas", "area_util", "area_extra"],
}
DUP_KEYS["full"] = DUP_KEYS["red"] + [
    "tipo_vendedor", "diferenciais", "churrasqueira", "estacionamento",
    "piscina", "playground", "quadra", "s_festas", "s_jogos",
    "s_ginastica", "sauna", "vista_mar",
]


def add_duplicate_features(train_df, test_df, seed=RANDOM_STATE):
    for name, cols in DUP_KEYS.items():
        kcol, tecol, ccol = f"k_{name}", f"te_{name}", f"cnt_{name}"
        train_df[kcol] = train_df[cols].astype(str).agg("|".join, axis=1)
        test_df[kcol] = test_df[cols].astype(str).agg("|".join, axis=1)
        vc = pd.concat([train_df[kcol], test_df[kcol]]).value_counts()
        train_df[ccol] = train_df[kcol].map(vc)
        test_df[ccol] = test_df[kcol].map(vc)
        train_df, test_df = add_target_encoding(
            train_df, test_df, kcol, "log_preco", tecol, k=1, seed=seed
        )
    return train_df, test_df


AMENIDADES = ["churrasqueira", "estacionamento", "piscina", "playground", "quadra",
              "s_festas", "s_jogos", "s_ginastica", "sauna", "vista_mar"]


def add_building_features(train_df, test_df, seed=RANDOM_STATE):
    """Assinatura de predio: anuncios do mesmo predio compartilham bairro e
    o conjunto de amenidades, mesmo sendo unidades diferentes. O TE do
    preco/m2 (log) nessa chave da sinal fino para linhas sem match exato."""
    for df in (train_df, test_df):
        df["bld_key"] = df["bairro"].astype(str) + "|" + df[AMENIDADES].astype(str).agg("".join, axis=1)
        df["bld_dif_key"] = df["bairro"].astype(str) + "|" + df["diferenciais"].astype(str)
    train_df["log_ppm2_t"] = np.log1p(train_df["preco"] / train_df["area_util"].clip(lower=1))
    train_df, test_df = add_target_encoding(
        train_df, test_df, "bld_key", "log_ppm2_t", "bld_ppm2_te", k=5, seed=seed)
    train_df, test_df = add_target_encoding(
        train_df, test_df, "bld_dif_key", "log_ppm2_t", "blddif_ppm2_te", k=3, seed=seed)
    vc = pd.concat([train_df["bld_key"], test_df["bld_key"]]).value_counts()
    train_df["bld_cnt"] = train_df["bld_key"].map(vc)
    test_df["bld_cnt"] = test_df["bld_key"].map(vc)
    train_df = train_df.drop(columns=["log_ppm2_t"])
    return train_df, test_df


def build_matrices(train_df, test_df):
    X_train = train_df[NUMERIC_FEATURES + CAT_FEATURES].copy()
    X_test = test_df[NUMERIC_FEATURES + CAT_FEATURES].copy()
    # Categorias fixadas com base em train+test para evitar categoria
    # desconhecida em algum fold do CV (o XGBoost exige categorias de
    # validacao subconjunto das de treino quando usa dtype categorico).
    for c in CAT_FEATURES:
        categories = sorted(set(X_train[c].astype(str)) | set(X_test[c].astype(str)))
        cat_dtype = pd.CategoricalDtype(categories=categories)
        X_train[c] = X_train[c].astype(str).astype(cat_dtype)
        X_test[c] = X_test[c].astype(str).astype(cat_dtype)
    return X_train, X_test


def train_catboost(X_tr, price_tr, X_val, price_val, X_test, cat_features):
    # RMSE em preco cru com sample_weight 1/y^2 e exatamente MSPE: otimiza a
    # metrica da competicao de forma nativa (e muito mais rapida que um
    # objetivo custom em Python).
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    for c in cat_features:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    w_tr = (np.mean(price_tr) / price_tr) ** 2

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=150,
        verbose=False,
    )
    model.fit(
        X_tr, price_tr,
        sample_weight=w_tr,
        eval_set=(X_val, price_val),
        cat_features=cat_features,
        use_best_model=True,
        verbose=False,
    )
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    return pred_val, pred_test, model


def _make_rmspe_objective(scale):
    """Objetivo customizado que otimiza RMSPE diretamente (nao uma aproximacao
    via log): L = ((pred-y)/y)^2 => grad = 2(pred-y)/y^2, hess = 2/y^2.
    `scale` normaliza a magnitude de grad/hess (~O(1)) para nao esbarrar nos
    limiares padrao de min_child_weight/min_sum_hessian_in_leaf dos GBMs."""
    def objective(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        grad = scale * 2.0 * (y_pred - y_true) / (y_true ** 2)
        hess = scale * 2.0 / (y_true ** 2) * np.ones_like(y_true)
        return grad, hess
    return objective


def train_lgbm(X_tr, price_tr, X_val, price_val, X_test, cat_features, seed=RANDOM_STATE, **overrides):
    # X_tr/X_val/X_test ja chegam com dtype category (categorias fixadas
    # em build_matrices com base em train+test) - nao recastar aqui.
    scale = float(np.mean(price_tr) ** 2)
    objective = _make_rmspe_objective(scale)

    def eval_rmspe(y_true, y_pred):
        return "rmspe", rmspe(y_true, y_pred), False

    params = dict(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=seed,
        verbose=-1,
    )
    params.update(overrides)
    model = lgb.LGBMRegressor(objective=objective, **params)
    model.fit(
        X_tr, price_tr,
        eval_set=[(X_val, price_val)],
        eval_metric=eval_rmspe,
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    return pred_val, pred_test, model


def train_xgb(X_tr, price_tr, X_val, price_val, X_test, cat_features):
    # X_tr/X_val/X_test ja chegam com dtype category (categorias fixadas
    # em build_matrices com base em train+test) - nao recastar aqui.
    scale = float(np.mean(price_tr) ** 2)
    objective = _make_rmspe_objective(scale)

    def eval_rmspe(y_true, y_pred):
        return rmspe(y_true, y_pred)

    model = xgb.XGBRegressor(
        objective=objective,
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=1e-6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
        early_stopping_rounds=150,
        eval_metric=eval_rmspe,
        base_score=float(np.mean(price_tr)),
    )
    model.fit(X_tr, price_tr, eval_set=[(X_val, price_val)], verbose=False)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    return pred_val, pred_test, model


def train_extratrees(X_tr, price_tr, X_val, price_val, X_test, cat_features):
    # ExtraTrees nao aceita dtype categorico do pandas: usa codigos inteiros
    # das categorias (fixadas globalmente em build_matrices) como numericos.
    # Treina em log1p(preco) e devolve previsoes na escala de preco.
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    for c in cat_features:
        X_tr[c] = X_tr[c].cat.codes
        X_val[c] = X_val[c].cat.codes
        X_test[c] = X_test[c].cat.codes

    y_tr = np.log1p(price_tr)

    model = ExtraTreesRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    pred_val = np.expm1(model.predict(X_val))
    pred_test = np.expm1(model.predict(X_test))
    return pred_val, pred_test, model


def main():
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    n_before = len(train_raw)
    train_raw = train_raw[(train_raw["preco"] >= 10_000) & (train_raw["preco"] <= 30_000_000)].reset_index(drop=True)
    print(f"Outliers removidos: {n_before - len(train_raw)} (de {n_before})")

    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)

    train_fe["log_preco"] = np.log1p(train_fe["preco"])
    train_fe, test_fe = add_bairro_target_encoding(train_fe, test_fe, "log_preco", seed=CV_SEED)
    train_fe, test_fe = add_duplicate_features(train_fe, test_fe, seed=CV_SEED)
    train_fe, test_fe = add_building_features(train_fe, test_fe, seed=CV_SEED)

    X_all, X_test_all = build_matrices(train_fe, test_fe)
    preco_all = train_fe["preco"].values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    model_names = ["cat", "lgb", "lgb2", "lgb3", "xgb", "et"]
    trainers = {
        "cat": train_catboost,
        "lgb": train_lgbm,
        "lgb2": lambda *a: train_lgbm(*a, seed=2027),
        "lgb3": lambda *a: train_lgbm(*a, seed=777),
        "xgb": train_xgb,
        "et": train_extratrees,
    }
    oof = {m: np.zeros(len(X_all)) for m in model_names}
    test_preds = {m: [] for m in model_names}

    total_steps = N_SPLITS * len(model_names)
    step = 0
    t0 = time.time()
    print(f"Treinando {total_steps} modelos ({N_SPLITS} folds x {len(model_names)} algoritmos)...")
    print_progress(0, total_steps, "iniciando", t0)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        price_tr, price_val = preco_all[tr_idx], preco_all[val_idx]

        fold_scores = {}
        fold_preds = {}
        for m in model_names:
            pv, pt, _ = trainers[m](X_tr, price_tr, X_val, price_val, X_test_all, CAT_FEATURES)
            step += 1
            print_progress(step, total_steps, f"fold {fold} {m}", t0)
            oof[m][val_idx] = pv
            test_preds[m].append(pt)
            fold_preds[m] = pv
            fold_scores[m] = rmspe(price_val, pv)

        blend_fold = np.mean([fold_preds[m] for m in model_names], axis=0)
        r_blend = rmspe(price_val, blend_fold)
        scores_str = "  ".join(f"{m}={fold_scores[m]:.4f}" for m in model_names)
        print(f"Fold {fold}: {scores_str}  Blend={r_blend:.4f}")

    print()
    for m in model_names:
        r = rmspe(preco_all, oof[m])
        print(f"OOF RMSPE {m:<4}: {r:.4f}")
    oof_blend_equal = np.mean([oof[m] for m in model_names], axis=0)
    print(f"OOF RMSPE blend (media simples): {rmspe(preco_all, oof_blend_equal):.4f}")

    # otimizacao dos pesos do blend (simplex: soma=1, pesos>=0) via SLSQP
    from scipy.optimize import minimize

    def neg_obj(w):
        pred = sum(wi * oof[m] for wi, m in zip(w, model_names))
        return rmspe(preco_all, pred)

    n_m = len(model_names)
    w0 = np.full(n_m, 1.0 / n_m)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_m
    res = minimize(neg_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    best_w = res.x
    best_r = res.fun
    weights_str = ", ".join(f"{m}={w:.3f}" for m, w in zip(model_names, best_w))
    print(f"Melhor blend OOF (SLSQP): {weights_str}  RMSPE={best_r:.4f}")

    test_final = {m: np.mean(test_preds[m], axis=0) for m in model_names}

    # dump para iterar o pos-processamento offline sem retreinar
    np.savez(
        "output/preds_dump"
        + ("" if CV_SEED == RANDOM_STATE else f"_s{CV_SEED}")
        + ("" if N_SPLITS == 10 else f"_f{N_SPLITS}")
        + ".npz",
        y=preco_all,
        ids_test=test_fe["Id"].values,
        k_red_tr=train_fe["k_red"].values.astype(str),
        k_red_te=test_fe["k_red"].values.astype(str),
        model_names=np.array(model_names),
        **{f"oof_{m}": oof[m] for m in model_names},
        **{f"test_{m}": test_final[m] for m in model_names},
    )

    test_pred = sum(w * test_final[m] for w, m in zip(best_w, model_names))
    oof_pred = sum(w * oof[m] for w, m in zip(best_w, model_names))

    # ------------------------------------------------------------------
    # Pos-processamento 1: blend com o "match de duplicatas" (chave red).
    # Para linhas com anuncios irmaos, o preco dos irmaos e um estimador
    # forte. Previsao otima de RMSPE dado um grupo: p* = sum(1/y)/sum(1/y2).
    # O peso do blend e ajustado no OOF (simulacao leave-one-out), com
    # pesos separados para 1 irmao e 2+ irmaos.
    # ------------------------------------------------------------------
    key_tr = train_fe["k_red"]
    key_te = test_fe["k_red"]
    y = preco_all

    g = pd.DataFrame({"k": key_tr, "y": y}).groupby("k")["y"]
    cnt = g.transform("count").values
    s1 = g.transform(lambda s: (1 / s).sum()).values - 1 / y
    s2 = g.transform(lambda s: (1 / s ** 2).sum()).values - 1 / y ** 2
    match_oof = np.where(cnt >= 2, s1 / np.maximum(s2, 1e-30), np.nan)
    sib_oof = cnt - 1

    stats = pd.DataFrame({"k": key_tr, "y": y}).groupby("k")["y"].agg(
        s1=lambda s: (1 / s).sum(), s2=lambda s: (1 / s ** 2).sum(), c="count")
    stats["p"] = stats["s1"] / stats["s2"]
    match_test = key_te.map(stats["p"]).values
    sib_test = key_te.map(stats["c"]).fillna(0).values

    def blend_match(pred, match, sib, w1, w2):
        out = pred.copy()
        for w, m in [(w1, ~np.isnan(match) & (sib == 1)),
                     (w2, ~np.isnan(match) & (sib >= 2))]:
            out[m] = np.exp(w * np.log(match[m]) + (1 - w) * np.log(pred[m]))
        return out

    grid = np.linspace(0, 1, 21)
    best_w1, best_w2, best_rm = 0.0, 0.0, np.inf
    for w1 in grid:
        for w2 in grid:
            r = rmspe(y, blend_match(oof_pred, match_oof, sib_oof, w1, w2))
            if r < best_rm:
                best_w1, best_w2, best_rm = w1, w2, r
    oof_pred = blend_match(oof_pred, match_oof, sib_oof, best_w1, best_w2)
    test_pred = blend_match(test_pred, match_test, sib_test, best_w1, best_w2)
    print(f"Blend com match de duplicatas: w1={best_w1:.2f} (1 irmao), "
          f"w2={best_w2:.2f} (2+ irmaos)  OOF RMSPE={best_rm:.4f}")

    # ------------------------------------------------------------------
    # Pos-processamento 2: calibracao multiplicativa global. O otimo de
    # RMSPE nao e a media da distribuicao preditiva (p* = E[1/y]/E[1/y2]),
    # entao um fator global < 1 costuma reduzir o erro.
    # ------------------------------------------------------------------
    ss = np.linspace(0.90, 1.05, 61)
    rs = [rmspe(y, s * oof_pred) for s in ss]
    s_best = ss[int(np.argmin(rs))]
    oof_pred = s_best * oof_pred
    test_pred = s_best * test_pred
    print(f"Calibracao global: s={s_best:.3f}  OOF RMSPE final={rmspe(y, oof_pred):.4f}")

    test_pred = np.clip(test_pred, 10_000, None)

    submission = pd.DataFrame({"Id": test_fe["Id"], "preco": test_pred})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmissao salva em {OUTPUT_PATH}")
    print(submission.head())
    diag = train_fe[["Id", "tipo", "bairro", "area_util", "area_extra", "preco"]].copy()
    diag["pred"] = oof_pred
    diag["erro_pct"] = (diag["preco"] - diag["pred"]) / diag["preco"]
    diag.to_csv("output/oof_diagnostico.csv", index=False)
    print("\nPiores previsoes OOF (maior erro percentual absoluto):")
    print(diag.reindex(diag["erro_pct"].abs().sort_values(ascending=False).index).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
