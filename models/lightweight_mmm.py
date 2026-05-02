"""LightweightMMM — Google's JAX-based MMM.

Uses geometric adstock + Hill saturation internally.
Requires: pip install lightweight_mmm

Falls back to a numpy-based MMM if JAX/lightweight_mmm unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mape(y_true, y_pred) -> float:
    mask = np.array(y_true) != 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


_LWMMM_TIMEOUT_SEC = 120


def _subprocess_target(train_dict, test_dict, cfg, queue):
    """Runs in a child process so we can hard-kill it if JAX hangs."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # avoid Metal/GPU init hang on macOS
    try:
        import pandas as pd
        train_df = pd.DataFrame(train_dict)
        test_df = pd.DataFrame(test_dict)
        from lightweight_mmm import lightweight_mmm, preprocessing
        import jax.numpy as jnp
        result = _run_lightweight(train_df, test_df, cfg, lightweight_mmm, preprocessing, jnp)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", str(e)))


def run(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_subprocess_target,
        args=(train_df.to_dict("list"), test_df.to_dict("list"), cfg, queue),
        daemon=True,
    )
    p.start()
    p.join(timeout=_LWMMM_TIMEOUT_SEC)
    if p.is_alive():
        p.kill()
        p.join()
        note = f"LightweightMMM/JAX timed out after {_LWMMM_TIMEOUT_SEC}s — used BayesianRidge fallback"
        result = _run_numpy_fallback(train_df, test_df, cfg)
        result["note"] = note
        return result
    status, payload = queue.get()
    if status == "ok":
        return payload
    note = f"LightweightMMM/JAX error ({payload}) — used BayesianRidge fallback"
    result = _run_numpy_fallback(train_df, test_df, cfg)
    result["note"] = note
    return result


def _run_lightweight(train_df, test_df, cfg, lightweight_mmm, preprocessing, jnp) -> dict:
    channels = cfg["media_channels"]
    feat_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
    valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]

    media = train_df[feat_cols].values.astype(np.float32)
    target = train_df["kpi"].values.astype(np.float32)

    scaler_media = preprocessing.CustomScaler(divide_operation=jnp.mean)
    scaler_target = preprocessing.CustomScaler(divide_operation=jnp.mean)
    media_s = scaler_media.fit_transform(media)
    target_s = scaler_target.fit_transform(target)

    mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
    mmm.fit(
        media=media_s,
        media_prior=jnp.ones(len(feat_cols)),
        target=target_s,
        number_warmup=cfg.get("pymc_tune", 500),
        number_samples=cfg.get("pymc_samples", 1000),
        number_chains=2,
        seed=42,
    )

    media_contribution, _ = mmm.get_contribution_decomposition(media_s)
    # media_contribution is in scaled target space — multiply by target mean to recover original scale
    # (scaler_target divides by mean, so inverse is multiply by mean)
    target_mean = float(target.mean()) if float(target.mean()) != 0 else 1.0
    media_contribution_original = media_contribution * target_mean

    total_kpi = float(target.sum())
    channel_contribs = {}
    for i, ch in enumerate(valid_channels):
        channel_contribs[ch] = float(media_contribution_original[:, i].sum())

    roi = {}
    for ch in valid_channels:
        spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 1
        roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

    y_pred_train = scaler_target.inverse_transform(mmm.trace["mu"].mean(axis=0))
    train_r2 = 1 - np.var(target - y_pred_train) / np.var(target)
    train_mape = _mape(target, y_pred_train)

    # Test
    media_test = test_df[feat_cols].values.astype(np.float32)
    media_test_s = scaler_media.transform(media_test)
    y_pred_test = scaler_target.inverse_transform(
        mmm.predict(media=media_test_s, extra_features=None).mean(axis=0)
    )
    test_mape = _mape(test_df["kpi"].values, y_pred_test)

    return {
        "model": "lightweight_mmm",
        "train_r2": round(float(train_r2), 4),
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "channel_contributions": {k: round(v, 2) for k, v in channel_contribs.items()},
        "channel_contribution_pct": {
            k: round(v / total_kpi * 100, 2) if total_kpi else 0
            for k, v in channel_contribs.items()
        },
        "roi": {k: round(v, 4) for k, v in roi.items()},
        "y_pred_train": y_pred_train.tolist(),
        "y_actual_train": target.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": test_df["kpi"].tolist(),
    }


def _run_numpy_fallback(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict) -> dict:
    """BayesianRidge MMM fallback — used when JAX/LightweightMMM is unavailable.

    Uses sklearn BayesianRidge with automatic hyperparameter tuning via evidence
    maximisation. Allows negative coefficients (unlike NNLS), giving meaningful
    attribution for negative-ROI channels. Provides a statistically independent
    third estimate for the CV ensemble.
    """
    from sklearn.linear_model import BayesianRidge

    channels = cfg["media_channels"]
    feat_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
    valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]
    ctrl_cols = [c for c in cfg["control_variables"] if c in train_df.columns]

    all_cols = feat_cols + ctrl_cols + ["trend"]
    present_cols = [c for c in all_cols if c in train_df.columns]

    X_train = np.column_stack([train_df[c].values for c in present_cols])
    X_test  = np.column_stack([test_df[c].values  for c in present_cols if c in test_df.columns])
    y_train = train_df["kpi"].values.astype(float)
    y_test  = test_df["kpi"].values.astype(float)

    # Standardise y so BayesianRidge's automatic regularisation isn't overwhelmed
    # by the scale difference between X (Hill-saturated, 0–1) and y (raw revenue, millions)
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    ys_train = (y_train - y_mean) / y_std

    model = BayesianRidge(max_iter=500, tol=1e-4)
    model.fit(X_train, ys_train)
    coefs = model.coef_ * y_std  # rescale to original revenue units

    y_pred_train = model.predict(X_train) * y_std + y_mean
    y_pred_test  = model.predict(X_test)  * y_std + y_mean

    train_r2   = 1 - np.var(y_train - y_pred_train) / (np.var(y_train) + 1e-10)
    train_mape = _mape(y_train, y_pred_train)
    test_mape  = _mape(y_test, y_pred_test)

    total_kpi = float(y_train.sum())
    channel_contribs = {}
    for ch in valid_channels:
        col = f"{ch}_saturated"
        if col not in present_cols:
            channel_contribs[ch] = 0.0
            continue
        col_idx = present_cols.index(col)
        channel_contribs[ch] = float((coefs[col_idx] * X_train[:, col_idx]).sum())

    # Cap positive media attribution at a plausible ceiling (default 65% of total KPI).
    # BayesianRidge can over-attribute when negative-coefficient channels offset each
    # other, inflating positive channel contributions beyond physically possible levels.
    max_media_pct = float(cfg.get("bayesian_ridge_max_media_pct", 0.65))
    pos_total = sum(v for v in channel_contribs.values() if v > 0)
    cap = max_media_pct * total_kpi
    if pos_total > cap:
        scale = cap / pos_total
        channel_contribs = {
            ch: v * scale if v > 0 else v
            for ch, v in channel_contribs.items()
        }

    roi = {}
    for ch in valid_channels:
        spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 1
        roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

    return {
        "model": "lightweight_mmm",
        "note": "LightweightMMM/JAX unavailable — used BayesianRidge fallback",
        "train_r2": round(float(train_r2), 4),
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "channel_contributions": {k: round(v, 2) for k, v in channel_contribs.items()},
        "channel_contribution_pct": {
            k: round(v / total_kpi * 100, 2) if total_kpi else 0
            for k, v in channel_contribs.items()
        },
        "roi": {k: round(v, 4) for k, v in roi.items()},
        "y_pred_train": y_pred_train.tolist(),
        "y_actual_train": y_train.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": y_test.tolist(),
    }
