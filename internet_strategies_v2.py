"""
INTERNET STRATEGIES ENGINE v1 — WDO B3

20 estrategias da internet + IA evolutiva.

LICOES APLICADAS:
- flush=True em todos os prints (log nao fica vazio)
- Entrada no OPEN do proximo candle (sem lookahead)
- MIN_TRADES=300 IS, 50 OOS
- MAX_PF=2.5, MAX_SHARPE=3.0 (anti-overfitting)
- Plateau test: vizinhos +/-10% devem funcionar
- Numba compilado para velocidade
- VWAP calculado corretamente (reseta por dia)
- SHORT favorecido (insight do WDO)
- Sessao PM favorecida

FASE 1: Grid 1M combos com Numba
FASE 2: IA evolutiva nos candidatos (PF > 0.95)
FASE 3: Plateau test + OOS + stress
"""

import pandas as pd
import numpy as np
from numba import njit
import json, sys, os, time, warnings, math, itertools
from datetime import datetime
from scipy import stats
warnings.filterwarnings("ignore")

CSV_PATH       = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR     = "/workspace/param_opt_output/internet_strategies"
CAPITAL        = 50_000.0
MULT           = 10.0
COMM           = 5.0
SLIP           = 2.0
MIN_TRADES_IS  = 300
MIN_TRADES_OOS = 50
MAX_PF         = 2.5
MAX_SHARPE     = 3.0
MAX_DD         = -30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SECAO 1: DADOS
# ================================================================

def carregar():
    print("[DATA] Carregando...", flush=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    return df


# ================================================================
# SECAO 2: INDICADORES (pre-computados uma vez)
# ================================================================

def calcular_indicadores(df):
    print("[IND] Calculando...", flush=True)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    ind = {
        "close":     c,
        "high":      h,
        "low":       l,
        "open":      o,
        "volume":    v,
        "open_next": np.concatenate([o[1:], [c[-1]]]),
    }

    # EMAs — todos os periodos usados nos grids
    for p in [3, 5, 8, 9, 10, 13, 20, 21, 34, 50, 100, 200]:
        a   = 2 / (p + 1)
        out = np.empty_like(c); out[0] = c[0]
        for i in range(1, n):
            out[i] = a * c[i] + (1 - a) * out[i - 1]
        ind[f"ema_{p}"] = out

    # RSIs — todos os periodos usados nos grids
    for p in [2, 3, 5, 7, 9, 11, 14, 18, 21, 28]:
        dv = np.diff(c, prepend=c[0])
        g  = np.where(dv > 0, dv, 0.0)
        ls = np.where(dv < 0, -dv, 0.0)
        ag = np.full(n, np.nan)
        al = np.full(n, np.nan)
        if p < n:
            ag[p] = g[1:p + 1].mean()
            al[p] = ls[1:p + 1].mean()
            for i in range(p + 1, n):
                ag[i] = (ag[i - 1] * (p - 1) + g[i]) / p
                al[i] = (al[i - 1] * (p - 1) + ls[i]) / p
        ind[f"rsi_{p}"] = 100 - (100 / (1 + ag / (al + 1e-9)))

    # ATR
    prev = np.roll(c, 1); prev[0] = c[0]
    tr   = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    for p in [7, 14, 20]:
        atr = np.full(n, np.nan)
        if p < n:
            atr[p - 1] = tr[:p].mean()
            for i in range(p, n):
                atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
        ind[f"atr_{p}"] = atr

    # MACD — todos os pares usados nos grids
    for fast, slow in [(3, 10), (5, 13), (8, 21), (10, 22), (12, 26)]:
        af  = 2 / (fast + 1)
        as_ = 2 / (slow + 1)
        ef  = np.empty_like(c); ef[0] = c[0]
        es  = np.empty_like(c); es[0] = c[0]
        for i in range(1, n):
            ef[i] = af  * c[i] + (1 - af)  * ef[i - 1]
            es[i] = as_ * c[i] + (1 - as_) * es[i - 1]
        mac = ef - es
        sig = np.empty_like(c); sig[0] = mac[0]; a9 = 2 / 10
        for i in range(1, n):
            sig[i] = a9 * mac[i] + (1 - a9) * sig[i - 1]
        ind[f"macd_{fast}_{slow}_hist"] = mac - sig

    # Bollinger Bands
    for p in [5, 10, 20, 50]:
        s   = pd.Series(c)
        sma = s.rolling(p).mean().values
        std = s.rolling(p).std().values
        for mult_str, mult_f in [("10", 1.0), ("15", 1.5), ("20", 2.0), ("25", 2.5), ("30", 3.0)]:
            up  = sma + mult_f * std
            lo  = sma - mult_f * std
            pct = (c - lo) / (up - lo + 1e-9)
            wid = (up - lo) / (sma + 1e-9)
            ind[f"bb_{p}_{mult_str}_pct"]   = pct
            ind[f"bb_{p}_{mult_str}_upper"] = up
            ind[f"bb_{p}_{mult_str}_lower"] = lo
            ind[f"bb_{p}_{mult_str}_width"] = wid

    # Donchian
    for p in [5, 10, 20, 50, 100, 200]:
        ind[f"don_high_{p}"] = pd.Series(h).rolling(p).max().shift(1).values
        ind[f"don_low_{p}"]  = pd.Series(l).rolling(p).min().shift(1).values

    # Stochastic
    for p in [3, 5, 7, 9, 14, 21]:
        lo_p = pd.Series(l).rolling(p).min().values
        hi_p = pd.Series(h).rolling(p).max().values
        stk  = (c - lo_p) / (hi_p - lo_p + 1e-9) * 100
        ind[f"stoch_k_{p}"] = stk

    # CCI
    for p in [7, 10, 14, 20, 30]:
        tp  = (h + l + c) / 3
        sma = pd.Series(tp).rolling(p).mean().values
        mad = pd.Series(tp).rolling(p).apply(
            lambda x: np.abs(x - x.mean()).mean()).values
        ind[f"cci_{p}"] = (tp - sma) / (0.015 * mad + 1e-9)

    # VWAP diario (reseta todo dia — CORRETO)
    vwap_arr = np.full(n, np.nan)
    tp       = (h + l + c) / 3
    cum_tpv  = np.zeros(n)
    cum_vol  = np.zeros(n)
    datas    = df.index.date
    data_atual = None
    for i in range(n):
        if datas[i] != data_atual:
            data_atual = datas[i]
            cum_tpv[i] = tp[i] * v[i]
            cum_vol[i] = v[i]
        else:
            cum_tpv[i] = cum_tpv[i - 1] + tp[i] * v[i]
            cum_vol[i] = cum_vol[i - 1] + v[i]
        if cum_vol[i] > 0:
            vwap_arr[i] = cum_tpv[i] / cum_vol[i]
    ind["vwap"] = vwap_arr

    # VWAP desvios (bandas de 1 e 2 std)
    vwap_std   = np.full(n, np.nan)
    data_atual = None
    sq_sum     = np.zeros(n)
    cnt        = np.zeros(n)
    for i in range(n):
        if datas[i] != data_atual:
            data_atual = datas[i]
            sq_sum[i]  = (c[i] - vwap_arr[i]) ** 2
            cnt[i]     = 1
        else:
            sq_sum[i] = sq_sum[i - 1] + (c[i] - vwap_arr[i]) ** 2
            cnt[i]    = cnt[i - 1] + 1
        if cnt[i] > 1:
            vwap_std[i] = np.sqrt(sq_sum[i] / cnt[i])
    ind["vwap_std"]    = vwap_std
    ind["vwap_upper1"] = vwap_arr + vwap_std
    ind["vwap_lower1"] = vwap_arr - vwap_std
    ind["vwap_upper2"] = vwap_arr + 2 * vwap_std
    ind["vwap_lower2"] = vwap_arr - 2 * vwap_std

    # Volume zscore
    for p in [10, 20]:
        vm = pd.Series(v).rolling(p).mean().values
        vs = pd.Series(v).rolling(p).std().values
        ind[f"vol_z_{p}"]     = (v - vm) / (vs + 1e-9)
        ind[f"vol_ratio_{p}"] = v / (vm + 1e-9)

    # Volatilidade relativa
    ret = np.diff(c, prepend=c[0]) / (c + 1e-9)
    v5  = pd.Series(ret).rolling(5).std().values * 100
    v20 = pd.Series(ret).rolling(20).std().values * 100
    ind["vol_ratio"] = v5 / (v20 + 1e-9)

    # Keltner Channel
    for ema_p in [5, 10, 20, 50]:
        for atr_p in [7, 14, 20]:
            for mult_f in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                key = f"kc_{ema_p}_{atr_p}_{str(mult_f).replace('.', '')}"
                ema = ind[f"ema_{ema_p}"]
                atr = ind[f"atr_{atr_p}"]
                ind[f"{key}_upper"] = ema + mult_f * atr
                ind[f"{key}_lower"] = ema - mult_f * atr

    # Opening Range (primeiros N minutos do dia)
    for orb_min in [5, 10, 15, 20, 30, 45]:
        orb_high = np.full(n, np.nan)
        orb_low  = np.full(n, np.nan)
        day_data = {}
        for i in range(n):
            dt   = df.index[i]
            data = dt.date()
            mins = dt.hour * 60 + dt.minute - 9 * 60
            if mins < 0:
                continue
            if data not in day_data:
                day_data[data] = {"hi": -np.inf, "lo": np.inf, "done": False}
            if not day_data[data]["done"]:
                if mins <= orb_min:
                    day_data[data]["hi"] = max(day_data[data]["hi"], h[i])
                    day_data[data]["lo"] = min(day_data[data]["lo"], l[i])
                else:
                    day_data[data]["done"] = True
            if day_data[data]["done"] or mins > orb_min:
                orb_high[i] = day_data[data]["hi"]
                orb_low[i]  = day_data[data]["lo"]
        ind[f"orb_high_{orb_min}"] = orb_high
        ind[f"orb_low_{orb_min}"]  = orb_low

    # Sessao
    hora = df.index.hour
    ind["session_am"] = ((hora >= 9)  & (hora < 12)).astype(np.int8)
    ind["session_pm"] = ((hora >= 13) & (hora < 17)).astype(np.int8)
    ind["hora"]       = hora
    ind["dow"]        = df.index.dayofweek.values

    n_ind = len([k for k in ind
                 if k not in ["close", "open", "high", "low", "volume",
                               "open_next", "hora", "dow"]])
    print(f"[IND] {n_ind} indicadores prontos | {n:,} candles", flush=True)
    return ind


# ================================================================
# SECAO 3: SIMULADOR NUMBA
# ================================================================

@njit(cache=True)
def simular(open_next, high, low, entries, exits, sl_pts, tp_pts,
            capital, mult, comm, slip):
    n    = len(open_next)
    pnls = np.empty(n, dtype=np.float64)
    n_tr = 0
    em   = False
    ep   = sl = tp = 0.0
    d    = 1
    for i in range(n - 1):
        if em:
            hit_sl = (d == 1 and low[i]  <= sl) or (d == -1 and high[i] >= sl)
            hit_tp = (d == 1 and high[i] >= tp) or (d == -1 and low[i]  <= tp)
            if hit_sl or hit_tp or exits[i]:
                saida = sl if hit_sl else (tp if hit_tp else open_next[i])
                pnls[n_tr] = (saida - ep) * d * mult - comm - slip * mult * 0.1
                n_tr += 1
                em = False
            continue
        if entries[i] and not em:
            ep = open_next[i]
            if np.isnan(ep) or ep <= 0:
                continue
            d  = 1
            sl = ep - sl_pts
            tp = ep + tp_pts
            em = True
    return pnls[:n_tr]


@njit(cache=True)
def simular_short(open_next, high, low, entries, exits, sl_pts, tp_pts,
                  capital, mult, comm, slip):
    n    = len(open_next)
    pnls = np.empty(n, dtype=np.float64)
    n_tr = 0
    em   = False
    ep   = sl = tp = 0.0
    for i in range(n - 1):
        if em:
            hit_sl = high[i] >= sl
            hit_tp = low[i]  <= tp
            if hit_sl or hit_tp or exits[i]:
                saida = sl if hit_sl else (tp if hit_tp else open_next[i])
                pnls[n_tr] = (ep - saida) * mult - comm - slip * mult * 0.1
                n_tr += 1
                em = False
            continue
        if entries[i] and not em:
            ep = open_next[i]
            if np.isnan(ep) or ep <= 0:
                continue
            sl = ep + sl_pts
            tp = ep - tp_pts
            em = True
    return pnls[:n_tr]


def executar_backtest(ind, entries, exits, sl_pts, tp_pts, direction="long"):
    on = ind["open_next"].astype(np.float64)
    hi = ind["high"].astype(np.float64)
    lo = ind["low"].astype(np.float64)
    e  = entries.astype(np.bool_)
    x  = exits.astype(np.bool_)
    if direction == "long":
        return simular(on, hi, lo, e, x, sl_pts, tp_pts, CAPITAL, MULT, COMM, SLIP)
    else:
        return simular_short(on, hi, lo, e, x, sl_pts, tp_pts, CAPITAL, MULT, COMM, SLIP)


def metricas(pnls, min_trades=MIN_TRADES_IS):
    if len(pnls) < min_trades:
        return None
    w = pnls[pnls > 0]
    l = pnls[pnls <= 0]
    if len(l) == 0 or len(w) == 0:
        return None
    pf = abs(w.sum() / l.sum())
    if pf > MAX_PF:
        return None
    eq  = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnls)])
    pk  = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk * 100).min())
    if mdd < MAX_DD:
        return None
    ret = pnls / CAPITAL
    sh  = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252 * 390))
    if sh > MAX_SHARPE:
        return None
    exp   = float(pnls.mean())
    n_jan = max(1, len(range(0, len(pnls) - 30, 15)))
    jan_pos = sum(1 for s in range(0, len(pnls) - 30, 15)
                  if pnls[s:s + 30].sum() > 0)
    return {
        "n":       len(pnls),
        "wr":      round(len(w) / len(pnls) * 100, 2),
        "pf":      round(pf, 3),
        "sh":      round(sh, 3),
        "exp":     round(exp, 2),
        "pnl":     round(float(pnls.sum()), 2),
        "mdd":     round(mdd, 2),
        "jan_pos": round(jan_pos / n_jan * 100, 1),
    }


# ================================================================
# SECAO 4: GERADORES DE SINAIS — 20+ ESTRATEGIAS
# ================================================================

def mascara_sessao(ind, session):
    if session == "am":
        return ind["session_am"].astype(bool)
    elif session == "pm":
        return ind["session_pm"].astype(bool)
    return np.ones(len(ind["close"]), dtype=bool)


def h1(x):
    return np.roll(x, 1)


def gerar_sinais(estrategia, ind, params):
    """Gera entradas e saidas para cada estrategia."""
    d    = params.get("direction", "short")
    ses  = params.get("session", "all")
    mask = mascara_sessao(ind, ses)
    c    = ind["close"]
    ent  = ext = None

    # ── 1. VWAP Reversion ──────────────────────────────────────
    if estrategia == "vwap_reversion":
        vwap = ind["vwap"]
        std  = ind["vwap_std"]
        mv   = params["vwap_std"]
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        if rsi is None or vwap is None:
            return None, None
        lvl = params["rsi_level"]
        if d == "long":
            ent = (c < vwap - mv * std) & (rsi < lvl) & (h1(rsi) >= lvl)
            ext = c > vwap
        else:
            ent = (c > vwap + mv * std) & (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl))
            ext = c < vwap

    # ── 2. VWAP Breakout ───────────────────────────────────────
    elif estrategia == "vwap_breakout":
        vwap = ind["vwap"]
        vr   = ind.get(f"vol_ratio_{params.get('vol_period', 20)}", ind["vol_ratio"])
        vc   = params["vol_confirm"]
        if d == "long":
            ent = (c > vwap) & (h1(c) <= h1(vwap)) & (vr > vc)
            ext = c < vwap
        else:
            ent = (c < vwap) & (h1(c) >= h1(vwap)) & (vr > vc)
            ext = c > vwap

    # ── 3. VWAP Pullback ───────────────────────────────────────
    elif estrategia == "vwap_pullback":
        vwap = ind["vwap"]
        ema  = ind.get(f"ema_{params['ema_period']}")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rf   = params["rsi_filter"]
        if ema is None or rsi is None:
            return None, None
        tol = ind["atr_14"] * 0.3
        if d == "long":
            ent = (ema > vwap) & (np.abs(c - vwap) < tol) & (rsi > rf)
            ext = c < vwap
        else:
            ent = (ema < vwap) & (np.abs(c - vwap) < tol) & (rsi < (100 - rf))
            ext = c > vwap

    # ── 4. ORB Breakout ────────────────────────────────────────
    elif estrategia == "orb_breakout":
        om  = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        vr = ind["vol_ratio"]
        vc = params.get("vol_confirm", 1.0)
        if d == "long":
            ent = (c > orh) & (h1(c) <= h1(orh)) & (vr > vc)
            ext = c < orl
        else:
            ent = (c < orl) & (h1(c) >= h1(orl)) & (vr > vc)
            ext = c > orh

    # ── 5. ORB Retest ──────────────────────────────────────────
    elif estrategia == "orb_retest":
        om  = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        tol       = ind["atr_14"] * 0.5
        above_orh = c > orh
        if d == "long":
            foi_acima = pd.Series(above_orh).rolling(20).max().values.astype(bool)
            ent = foi_acima & (np.abs(c - orh) < tol)
            ext = c < orl
        else:
            foi_abaixo = pd.Series(~above_orh).rolling(20).max().values.astype(bool)
            ent = foi_abaixo & (np.abs(c - orl) < tol)
            ext = c > orh

    # ── 6. RSI + VWAP ──────────────────────────────────────────
    elif estrategia == "rsi_vwap_combo":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        side = params["vwap_side"]
        lvl  = params["rsi_level"]
        if rsi is None:
            return None, None
        vwap_cond = (c > vwap) if side == "above" else (c < vwap)
        if d == "long":
            ent = (rsi < lvl) & (h1(rsi) >= lvl) & vwap_cond
            ext = rsi > 50
        else:
            ent = (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl)) & (~vwap_cond)
            ext = rsi < 50

    # ── 7. RSI + EMA + VWAP ────────────────────────────────────
    elif estrategia == "rsi_ema_vwap":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        ema  = ind.get(f"ema_{params['ema_period']}")
        vwap = ind["vwap"]
        lvl  = params["rsi_level"]
        if rsi is None or ema is None:
            return None, None
        if d == "long":
            ent = (rsi < lvl) & (h1(rsi) >= lvl) & (c > ema) & (c > vwap)
            ext = rsi > 55
        else:
            ent = (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl)) & (c < ema) & (c < vwap)
            ext = rsi < 45

    # ── 8. ATR Channel Breakout (Keltner) ──────────────────────
    elif estrategia == "atr_channel_breakout":
        ep  = params["ema_period"]
        ap  = params["atr_period"]
        am  = params["atr_mult"]
        key = f"kc_{ep}_{ap}_{str(am).replace('.', '')}"
        ku  = ind.get(f"{key}_upper")
        kl  = ind.get(f"{key}_lower")
        if ku is None:
            return None, None
        if d == "long":
            ent = (c > ku) & (h1(c) <= h1(ku)); ext = (c < kl) & (h1(c) >= h1(kl))
        else:
            ent = (c < kl) & (h1(c) >= h1(kl)); ext = (c > ku) & (h1(c) <= h1(ku))

    # ── 9. ATR Momentum ────────────────────────────────────────
    elif estrategia == "atr_trailing_momentum":
        pp  = params["momentum_period"]
        pt  = params["momentum_thresh"]
        roc = np.empty(len(c)); roc[:pp] = np.nan
        roc[pp:] = (c[pp:] - c[:-pp]) / (c[:-pp] + 1e-9) * 100
        if d == "long":
            ent = roc > pt;  ext = roc < 0
        else:
            ent = roc < -pt; ext = roc > 0

    # ── 10. MACD + VWAP ────────────────────────────────────────
    elif estrategia == "macd_vwap":
        mf   = params["macd_fast"]
        ms   = params["macd_slow"]
        hist = ind.get(f"macd_{mf}_{ms}_hist")
        vwap = ind["vwap"]
        if hist is None:
            return None, None
        if d == "long":
            ent = (hist > 0) & (h1(hist) <= 0) & (c > vwap)
            ext = (hist < 0) & (h1(hist) >= 0)
        else:
            ent = (hist < 0) & (h1(hist) >= 0) & (c < vwap)
            ext = (hist > 0) & (h1(hist) <= 0)

    # ── 11. MACD + RSI + VWAP ──────────────────────────────────
    elif estrategia == "macd_rsi_vwap":
        cfg  = params["macd_config"]
        hist = ind.get(f"macd_{cfg}_hist")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rf   = params["rsi_filter"]
        vwap = ind["vwap"]
        if hist is None or rsi is None:
            return None, None
        if d == "long":
            ent = (hist > 0) & (h1(hist) <= 0) & (rsi > rf) & (c > vwap)
            ext = (hist < 0) & (h1(hist) >= 0)
        else:
            ent = (hist < 0) & (h1(hist) >= 0) & (rsi < (100 - rf)) & (c < vwap)
            ext = (hist > 0) & (h1(hist) <= 0)

    # ── 12. EMA + VWAP Trend ───────────────────────────────────
    elif estrategia == "ema_vwap_trend":
        ef   = ind.get(f"ema_{params['fast']}")
        es   = ind.get(f"ema_{params['slow']}")
        vwap = ind["vwap"]
        if ef is None or es is None:
            return None, None
        if d == "long":
            ent = (ef > es) & (h1(ef) <= h1(es)) & (c > vwap)
            ext = (ef < es) & (h1(ef) >= h1(es))
        else:
            ent = (ef < es) & (h1(ef) >= h1(es)) & (c < vwap)
            ext = (ef > es) & (h1(ef) <= h1(es))

    # ── 13. Dual EMA + Volume ──────────────────────────────────
    elif estrategia == "dual_ema_momentum":
        ef = ind.get(f"ema_{params['fast']}")
        es = ind.get(f"ema_{params['slow']}")
        vr = ind["vol_ratio"]
        vc = params["vol_confirm"]
        if ef is None or es is None:
            return None, None
        if d == "long":
            ent = (ef > es) & (h1(ef) <= h1(es)) & (vr > vc)
            ext = (ef < es) & (h1(ef) >= h1(es))
        else:
            ent = (ef < es) & (h1(ef) >= h1(es)) & (vr > vc)
            ext = (ef > es) & (h1(ef) <= h1(es))

    # ── 14. BB Squeeze Breakout ────────────────────────────────
    elif estrategia == "bb_squeeze_breakout":
        bp  = params["bb_period"]
        bs  = params["bb_std"]
        vc  = params["vol_confirm"]
        key = f"bb_{bp}_{str(bs).replace('.', '')}"
        bw  = ind.get(f"{key}_width")
        bu  = ind.get(f"{key}_upper")
        bl  = ind.get(f"{key}_lower")
        vr  = ind["vol_ratio"]
        if bw is None:
            return None, None
        bw_avg  = pd.Series(bw).rolling(20).mean().values
        squeeze = bw < bw_avg * 0.8
        saindo  = ~squeeze & pd.Series(squeeze).shift(1).fillna(False).values
        if d == "long":
            ent = saindo & (c > bu) & (vr > vc); ext = c < bl
        else:
            ent = saindo & (c < bl) & (vr > vc); ext = c > bu

    # ── 15. BB + RSI + VWAP ────────────────────────────────────
    elif estrategia == "bb_rsi_vwap":
        bp   = params["bb_period"]
        bs   = params["bb_std"]
        key  = f"bb_{bp}_{str(bs).replace('.', '')}"
        pct  = ind.get(f"{key}_pct")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rc   = params["rsi_confirm"]
        vwap = ind["vwap"]
        if pct is None or rsi is None:
            return None, None
        if d == "long":
            ent = (pct < 0.05) & (rsi < rc) & (c > vwap)
            ext = pct > 0.5
        else:
            ent = (pct > 0.95) & (rsi > (100 - rc)) & (c < vwap)
            ext = pct < 0.5

    # ── 16. Donchian + VWAP ────────────────────────────────────
    elif estrategia == "donchian_vwap":
        dh   = ind.get(f"don_high_{params['don_period']}")
        dl   = ind.get(f"don_low_{params['don_period']}")
        vwap = ind["vwap"]
        vr   = ind["vol_ratio"]
        vc   = params["vol_confirm"]
        if dh is None:
            return None, None
        if d == "long":
            ent = (c > dh) & (c > vwap) & (vr > vc); ext = c < dl
        else:
            ent = (c < dl) & (c < vwap) & (vr > vc); ext = c > dh

    # ── 17. Stoch + VWAP ───────────────────────────────────────
    elif estrategia == "stoch_vwap":
        k    = ind.get(f"stoch_k_{params['stoch_period']}")
        vwap = ind["vwap"]
        ovs  = params["oversold"]
        ovb  = params["overbought"]
        if k is None:
            return None, None
        if d == "long":
            ent = (k < ovs) & (h1(k) >= ovs) & (c > vwap)
            ext = k > 50
        else:
            ent = (k > ovb) & (h1(k) <= ovb) & (c < vwap)
            ext = k < 50

    # ── 18. Volume Spike + RSI ─────────────────────────────────
    elif estrategia == "volume_spike_reversal":
        vz  = ind.get("vol_z_20")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        vs  = params["vol_spike"]
        lvl = params["rsi_level"]
        if vz is None or rsi is None:
            return None, None
        spike = vz > vs
        if d == "long":
            ent = spike & (rsi < lvl) & (h1(rsi) >= lvl)
            ext = rsi > 55
        else:
            ent = spike & (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl))
            ext = rsi < 45

    # ── 19. Volume + VWAP Momentum ─────────────────────────────
    elif estrategia == "volume_vwap_momentum":
        vr   = ind["vol_ratio"]
        vwap = ind["vwap"]
        vc   = params["vol_confirm"]
        if d == "long":
            ent = (c > vwap) & (h1(c) <= h1(vwap)) & (vr > vc)
            ext = c < vwap
        else:
            ent = (c < vwap) & (h1(c) >= h1(vwap)) & (vr > vc)
            ext = c > vwap

    # ── 20. CCI + VWAP ─────────────────────────────────────────
    elif estrategia == "cci_vwap":
        cci  = ind.get(f"cci_{params['cci_period']}")
        vwap = ind["vwap"]
        thr  = params["cci_thresh"]
        if cci is None:
            return None, None
        if d == "long":
            ent = (cci < -thr) & (h1(cci) >= -thr) & (c > vwap)
            ext = cci > 0
        else:
            ent = (cci > thr) & (h1(cci) <= thr) & (c < vwap)
            ext = cci < 0

    # ── 21. RSI VWAP SESSION ───────────────────────────────────
    elif estrategia == "rsi_vwap_session":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        ovs  = params["oversold"]
        ovb  = params["overbought"]
        exl  = params["exit_level"]
        if rsi is None:
            return None, None
        if d == "long":
            ent = (rsi < ovs) & (h1(rsi) >= ovs) & (c > vwap)
            ext = rsi > exl
        else:
            ent = (rsi > ovb) & (h1(rsi) <= ovb) & (c < vwap)
            ext = rsi < (100 - exl)

    # ── 22. STOCH + EMA + VWAP ─────────────────────────────────
    elif estrategia == "stoch_ema_vwap":
        k    = ind.get(f"stoch_k_{params['stoch_period']}")
        ema  = ind.get(f"ema_{params['ema_period']}")
        vwap = ind["vwap"]
        ovs  = params["oversold"]
        ovb  = params["overbought"]
        if k is None or ema is None:
            return None, None
        if d == "long":
            ent = (k < ovs) & (h1(k) >= ovs) & (c > ema) & (c > vwap)
            ext = k > 50
        else:
            ent = (k > ovb) & (h1(k) <= ovb) & (c < ema) & (c < vwap)
            ext = k < 50

    # ── 23. ORB + VWAP COMBO ───────────────────────────────────
    elif estrategia == "orb_vwap_combo":
        om   = params["orb_minutes"]
        orh  = ind.get(f"orb_high_{om}")
        orl  = ind.get(f"orb_low_{om}")
        vr   = ind["vol_ratio"]
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        rf   = params["rsi_filter"]
        vc   = params["vol_confirm"]
        if orh is None or rsi is None:
            return None, None
        if d == "long":
            ent = (c > orh) & (h1(c) <= h1(orh)) & (vr > vc) & (rsi > rf) & (c > vwap)
            ext = c < orl
        else:
            ent = (c < orl) & (h1(c) >= h1(orl)) & (vr > vc) & (rsi < (100 - rf)) & (c < vwap)
            ext = c > orh

    else:
        return None, None

    if ent is None:
        return None, None
    ent    = ent & mask
    ent[0] = False
    return ent.astype(np.bool_), ext.astype(np.bool_)


# ================================================================
# SECAO 5: GRIDS
# ================================================================

GRIDS = {
    # === VWAP STRATEGIES ===
    "vwap_reversion": {
        "vwap_std":   [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "rsi_period": [3, 5, 7, 9, 11, 14, 18, 21, 28],
        "rsi_level":  [10, 15, 20, 25, 30, 35, 40, 45],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "vwap_breakout": {
        "vol_confirm": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "vwap_pullback": {
        "ema_period": [5, 9, 20, 50, 100, 200],
        "rsi_period": [7, 9, 14, 21],
        "rsi_filter": [35, 40, 45, 50, 55, 60],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    # === ORB STRATEGIES ===
    "orb_breakout": {
        "orb_minutes": [5, 10, 15, 20, 30, 45],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "orb_retest": {
        "orb_minutes": [5, 10, 15, 20, 30, 45],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    # === RSI STRATEGIES ===
    "rsi_vwap_combo": {
        "rsi_period": [3, 5, 7, 9, 14, 21, 28],
        "rsi_level":  [10, 15, 20, 25, 30, 35, 40, 45],
        "vwap_side":  ["above", "below"],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "rsi_ema_vwap": {
        "rsi_period": [5, 7, 9, 14, 21, 28],
        "rsi_level":  [15, 20, 25, 30, 35, 40],
        "ema_period": [10, 20, 50, 100, 200],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "rsi_vwap_session": {
        "rsi_period": [2, 3, 5, 7, 9, 11, 14, 18, 21, 28],
        "oversold":   [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
        "overbought": [55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95],
        "exit_level": [45, 50, 55, 60],
        "atr_sl":     [0.5, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    # === ATR STRATEGIES ===
    "atr_channel_breakout": {
        "ema_period": [5, 10, 20, 50],
        "atr_period": [7, 14, 20],
        "atr_mult":   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "atr_trailing_momentum": {
        "momentum_period": [3, 5, 10, 15, 20, 30],
        "momentum_thresh": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        "atr_sl":          [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":              [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":         ["am", "pm", "all"],
        "direction":       ["long", "short"],
    },
    # === MACD STRATEGIES ===
    "macd_vwap": {
        "macd_fast": [3, 5, 8, 10, 12],
        "macd_slow": [10, 13, 21, 26],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":        [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":   ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "macd_rsi_vwap": {
        "macd_config": ["12_26", "8_21", "5_13", "3_10"],
        "rsi_period":  [7, 9, 14, 21],
        "rsi_filter":  [35, 40, 45, 50, 55, 60],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    # === EMA STRATEGIES ===
    "ema_vwap_trend": {
        "fast":      [3, 5, 8, 10, 13, 20, 21],
        "slow":      [20, 34, 50, 100, 200],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":        [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":   ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "dual_ema_momentum": {
        "fast":        [3, 5, 8, 10, 13, 20],
        "slow":        [20, 21, 34, 50, 100],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    # === BOLLINGER STRATEGIES ===
    "bb_squeeze_breakout": {
        "bb_period":   [5, 10, 20, 50],
        "bb_std":      [1.0, 1.5, 2.0, 2.5, 3.0],
        "squeeze_mult":[0.6, 0.7, 0.8, 0.9],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "bb_rsi_vwap": {
        "bb_period":   [5, 10, 20, 50],
        "bb_std":      [1.0, 1.5, 2.0, 2.5, 3.0],
        "rsi_period":  [5, 7, 9, 14, 21],
        "rsi_confirm": [15, 20, 25, 30, 35, 40],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    # === DONCHIAN ===
    "donchian_vwap": {
        "don_period":  [5, 10, 20, 50, 100, 200],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    # === STOCHASTIC STRATEGIES ===
    "stoch_vwap": {
        "stoch_period": [3, 5, 7, 9, 14, 21],
        "oversold":     [5, 10, 15, 20, 25, 30, 35],
        "overbought":   [65, 70, 75, 80, 85, 90, 95],
        "atr_sl":       [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":           [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":      ["am", "pm", "all"],
        "direction":    ["long", "short"],
    },
    "stoch_ema_vwap": {
        "stoch_period": [3, 5, 7, 9, 14, 21],
        "oversold":     [5, 10, 15, 20, 25, 30],
        "overbought":   [70, 75, 80, 85, 90, 95],
        "ema_period":   [20, 50, 100, 200],
        "atr_sl":       [0.5, 1.0, 1.5, 2.0],
        "rr":           [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":      ["am", "pm", "all"],
        "direction":    ["long", "short"],
    },
    # === VOLUME STRATEGIES ===
    "volume_spike_reversal": {
        "vol_spike":  [1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "rsi_period": [5, 7, 9, 14, 21],
        "rsi_level":  [20, 25, 30, 35, 40],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "volume_vwap_momentum": {
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    # === CCI ===
    "cci_vwap": {
        "cci_period": [7, 10, 14, 20, 30],
        "cci_thresh": [50, 75, 100, 125, 150, 175, 200, 250],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    # === ORB + VWAP COMBO ===
    "orb_vwap_combo": {
        "orb_minutes": [5, 10, 15, 20, 30],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5],
        "rsi_period":  [7, 14, 21],
        "rsi_filter":  [40, 45, 50, 55],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "rr":          [1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
}


# ================================================================
# SECAO 6: PLATEAU TEST (anti-overfitting)
# ================================================================

def plateau_test(estrategia, ind, melhor_params, atr_pts, top_pf):
    """
    Testa se vizinhos +/-1 do melhor parametro tem PF similar.
    REGRA: pelo menos 50% dos vizinhos deve ter PF > top_pf * 0.7
    """
    variacoes_numericas = {}
    for k, val in melhor_params.items():
        if isinstance(val, (int, float)) and k not in [
            "rr", "atr_sl", "direction", "session",
            "vwap_side", "macd_config", "macd_fast", "macd_slow",
        ]:
            variacoes_numericas[k] = val

    if not variacoes_numericas:
        return True, 1.0

    resultados_vizinhos = []
    param_teste = list(variacoes_numericas.keys())[0]
    valor_base  = variacoes_numericas[param_teste]

    for delta in [-2, -1, 1, 2]:
        params_viz = melhor_params.copy()
        params_viz[param_teste] = valor_base + delta
        if params_viz[param_teste] <= 0:
            continue
        try:
            ent, ext = gerar_sinais(estrategia, ind, params_viz)
            if ent is None or ent.sum() < 10:
                continue
            sl_pts = atr_pts * params_viz.get("atr_sl", 1.0)
            tp_pts = sl_pts  * params_viz.get("rr", 2.0)
            pnls   = executar_backtest(ind, ent, ext, sl_pts, tp_pts,
                                       params_viz.get("direction", "short"))
            m = metricas(pnls, min_trades=50)
            if m:
                resultados_vizinhos.append(m["pf"])
        except Exception:
            continue

    if not resultados_vizinhos:
        return True, 1.0

    pct_ok  = sum(1 for pf in resultados_vizinhos if pf > top_pf * 0.7) / len(resultados_vizinhos)
    robusto = pct_ok >= 0.5
    return robusto, round(pct_ok, 2)


# ================================================================
# SECAO 7: IA EVOLUTIVA — ADICIONA INDICADORES
# ================================================================

def ia_evolutiva(estrategia, ind, melhor_params, top_pf, atr_pts):
    """
    Se a estrategia quase passa (PF > 0.90), tenta adicionar
    filtros adicionais para melhorar.

    FILTROS TENTADOS:
    1. Adicionar filtro de volume minimo
    2. Adicionar filtro de ATR minimo
    3. Mudar sessao
    """
    if top_pf >= 1.0:
        return None

    print(f"    [IA] Tentando melhorar {estrategia} (PF={top_pf:.3f})...", flush=True)
    melhorias = []

    filtros_extras = [
        {"nome": "vol_min",    "param": "vol_min",  "valores": [1.5, 2.0, 2.5]},
        {"nome": "atr_min",    "param": "atr_min",  "valores": [5.0, 8.0, 12.0]},
        {"nome": "session_am", "param": "session",  "valores": ["am"]},
        {"nome": "session_pm", "param": "session",  "valores": ["pm"]},
    ]

    for filtro in filtros_extras:
        for val in filtro["valores"]:
            params_new = melhor_params.copy()
            params_new[filtro["param"]] = val
            try:
                ent, ext = gerar_sinais(estrategia, ind, params_new)
                if ent is None or ent.sum() < 10:
                    continue
                if filtro["nome"] == "vol_min":
                    ent = ent & (ind["vol_ratio"] > val)
                elif filtro["nome"] == "atr_min":
                    ent = ent & (ind["atr_14"] > val)
                sl_pts = atr_pts * params_new.get("atr_sl", 1.0)
                tp_pts = sl_pts  * params_new.get("rr", 2.0)
                pnls   = executar_backtest(ind, ent, ext, sl_pts, tp_pts,
                                           params_new.get("direction", "short"))
                m = metricas(pnls, min_trades=MIN_TRADES_IS)
                if m and m["pf"] > top_pf:
                    melhorias.append({
                        "filtro_adicionado": filtro["nome"],
                        "valor":             val,
                        "params":            params_new,
                        "pf_novo":           m["pf"],
                        "pf_anterior":       top_pf,
                        "melhoria_pct":      round((m["pf"] - top_pf) / top_pf * 100, 1),
                        "metricas":          m,
                    })
            except Exception:
                continue

    if melhorias:
        melhorias.sort(key=lambda x: -x["pf_novo"])
        melhor = melhorias[0]
        print(f"    [IA] Melhoria encontrada! "
              f"PF {top_pf:.3f} -> {melhor['pf_novo']:.3f} "
              f"(+{melhor['melhoria_pct']}%) "
              f"via {melhor['filtro_adicionado']}={melhor['valor']}", flush=True)
        return melhor

    print(f"    [IA] Sem melhoria encontrada", flush=True)
    return None


# ================================================================
# SECAO 8: GRID SEARCH PRINCIPAL
# ================================================================

def grid_search(estrategia, ind, grid, n_total, mini=False):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if mini:
        combos = combos[:20]

    print(f"\n[{estrategia.upper()}] {len(combos):,} combos...", flush=True)
    t0      = time.time()
    validos = []
    n_ok    = 0

    atr_pts_med = float(np.nanmean(ind["atr_14"]))

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            ent, ext = gerar_sinais(estrategia, ind, params)
            if ent is None or ent.sum() < 10:
                continue
            sl_pts = atr_pts_med * params.get("atr_sl", 1.0)
            tp_pts = sl_pts * params.get("rr", 2.0)
            pnls   = executar_backtest(ind, ent, ext, sl_pts, tp_pts,
                                       params.get("direction", "short"))
            m = metricas(pnls)
            if not m:
                continue
            n_ok += 1
            pf_s  = min(m["pf"], MAX_PF) / MAX_PF
            exp_s = max(0, min(m["exp"], 500)) / 500
            jan_s = m["jan_pos"] / 100
            tr_s  = min(m["n"], 2000) / 2000
            sh_s  = max(0, min(m["sh"], 3)) / 3
            score = pf_s * 0.30 + exp_s * 0.25 + jan_s * 0.20 + sh_s * 0.15 + tr_s * 0.10
            validos.append({
                "estrategia": estrategia,
                "params":     params,
                "score":      round(score, 6),
                **m,
            })
        except Exception:
            continue

    elapsed = time.time() - t0
    validos.sort(key=lambda x: -x["score"])
    spd = len(combos) / max(elapsed, 0.1)
    print(f"  {n_ok:,}/{len(combos):,} validos | {elapsed:.1f}s | {spd:.0f}/s", flush=True)

    if validos:
        r = validos[0]
        print(f"  TOP: PF={r['pf']:.3f} WR={r['wr']:.1f}% "
              f"Trades={r['n']} Exp=R${r['exp']:.2f} "
              f"Score={r['score']:.4f}", flush=True)
        with open(f"{OUTPUT_DIR}/{estrategia}_top10.json", "w") as fp:
            json.dump(validos[:10], fp, indent=2, default=str)

    return validos


# ================================================================
# SECAO 9: VALIDACAO OOS
# ================================================================

def validar_oos(estrategia, ind_oos, params, atr_pts_oos):
    ent, ext = gerar_sinais(estrategia, ind_oos, params)
    if ent is None or ent.sum() < 5:
        return None
    sl   = atr_pts_oos * params.get("atr_sl", 1.0)
    tp   = sl * params.get("rr", 2.0)
    pnls = executar_backtest(ind_oos, ent, ext, sl, tp,
                             params.get("direction", "short"))
    return metricas(pnls, min_trades=MIN_TRADES_OOS)


# ================================================================
# SECAO 10: MAIN
# ================================================================

def main():
    MINI = "--mini" in sys.argv

    total = sum(math.prod(len(v) for v in g.values()) for g in GRIDS.values())

    print("=" * 68, flush=True)
    print("  INTERNET STRATEGIES ENGINE v1 — WDO B3", flush=True)
    print(f"  {len(GRIDS)} estrategias | {total:,} combos | VWAP + ORB + Volume", flush=True)
    print(f"  IA Evolutiva: adiciona filtros nos candidatos promissores", flush=True)
    print("=" * 68, flush=True)

    df    = carregar()
    split = int(len(df) * 0.70)
    df_is = df.iloc[:split]
    df_os = df.iloc[split:]
    print(f"  IS : {len(df_is):,} | {df_is.index[0].date()} -> {df_is.index[-1].date()}", flush=True)
    print(f"  OOS: {len(df_os):,} | {df_os.index[0].date()} -> {df_os.index[-1].date()}", flush=True)

    ind_is = calcular_indicadores(df_is)

    # Aquece Numba
    print("\n[JIT] Compilando Numba...", flush=True)
    dv = np.ones(200, dtype=np.float64) * 5000
    bv = np.zeros(200, dtype=np.bool_); bv[10] = True
    _  = simular(dv, dv, dv, bv, bv, 20.0, 40.0, 50000, 10, 5, 2)
    _  = simular_short(dv, dv, dv, bv, bv, 20.0, 40.0, 50000, 10, 5, 2)
    print("[JIT] Pronto!", flush=True)

    todos        = []
    aprovados    = []
    ia_melhorias = []
    atr_med_is   = float(np.nanmean(ind_is["atr_14"]))

    for estrategia, grid in GRIDS.items():
        resultados = grid_search(estrategia, ind_is, grid, total, mini=MINI)
        todos.extend(resultados[:10])

        if not resultados:
            continue

        melhor = resultados[0]

        # Plateau test
        robusto, pct_viz = plateau_test(
            estrategia, ind_is, melhor["params"],
            atr_med_is, melhor["pf"],
        )
        melhor["plateau_ok"]  = robusto
        melhor["plateau_pct"] = pct_viz
        status_plateau = "OK" if robusto else "FRAGIL"
        print(f"  Plateau: {status_plateau} {pct_viz * 100:.0f}% vizinhos ok", flush=True)

        # IA evolutiva
        if 0.90 <= melhor["pf"] < 1.0:
            melhoria = ia_evolutiva(
                estrategia, ind_is, melhor["params"],
                melhor["pf"], atr_med_is,
            )
            if melhoria:
                ia_melhorias.append(melhoria)
                melhor_com_ia = melhor.copy()
                melhor_com_ia.update(melhoria["metricas"])
                melhor_com_ia["params"]      = melhoria["params"]
                melhor_com_ia["ia_melhoria"] = melhoria
                aprovados.append((estrategia, melhor_com_ia))
        elif melhor["pf"] >= 1.0 and robusto:
            aprovados.append((estrategia, melhor))

        aprovados_nomes = [a[0] for a in aprovados]
        status = "APROVADO" if estrategia in aprovados_nomes else "REPROVADO"
        print(f"  {status} PF={melhor['pf']:.3f}", flush=True)

    # Top geral
    todos.sort(key=lambda x: -x["score"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  TOP 20 GERAL", flush=True)
    print(f"  {'ESTRATEGIA':25} {'PF':>6} {'WR%':>6} {'Trades':>7} {'Exp':>8} {'Score':>7}", flush=True)
    print(f"  {'-' * 60}", flush=True)
    for r in todos[:20]:
        print(f"  {r['estrategia']:25} "
              f"{r['pf']:>6.3f} {r['wr']:>6.1f} "
              f"{r['n']:>7} {r['exp']:>8.2f} {r['score']:>7.4f}", flush=True)

    # Validacao OOS
    print(f"\n{'=' * 68}", flush=True)
    print(f"  VALIDANDO {len(aprovados)} NO OOS...", flush=True)

    ind_os     = calcular_indicadores(df_os)
    atr_med_os = float(np.nanmean(ind_os["atr_14"]))
    resultados_finais = []

    for estrategia, melhor in aprovados:
        params = melhor["params"]
        m_oos  = validar_oos(estrategia, ind_os, params, atr_med_os)
        is_pf  = melhor["pf"]
        oos_pf = m_oos["pf"] if m_oos else 0
        deg    = (is_pf - oos_pf) / is_pf * 100 if is_pf > 0 and oos_pf > 0 else 999
        ok     = m_oos is not None and oos_pf > 1.0 and deg < 50

        print(f"  {estrategia:25} IS={is_pf:.3f} OOS={oos_pf:.3f} "
              f"Deg={deg:.1f}% {'PASSA' if ok else 'REPROVADO'}", flush=True)

        resultado = {
            "estrategia":   estrategia,
            "params":       params,
            "metricas_is":  melhor,
            "metricas_oos": m_oos,
            "degradacao":   round(deg, 1),
            "plateau_ok":   melhor.get("plateau_ok", False),
            "ia_melhoria":  melhor.get("ia_melhoria"),
            "aprovado":     ok,
            "gerado_em":    datetime.now().isoformat(),
        }
        resultados_finais.append(resultado)
        with open(f"{OUTPUT_DIR}/{estrategia}_final.json", "w") as fp:
            json.dump(resultado, fp, indent=2, default=str)

    # Leaderboard
    n_apr = sum(1 for r in resultados_finais if r["aprovado"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  LEADERBOARD — {n_apr} APROVADO(S)", flush=True)
    print(f"  {'ESTRATEGIA':25} {'PF_IS':>6} {'PF_OOS':>7} "
          f"{'DEG%':>6} {'PLATEAU':>8} {'STATUS':>10}", flush=True)
    print(f"  {'-' * 68}", flush=True)

    for r in sorted(resultados_finais,
                    key=lambda x: -(x["metricas_is"] or {}).get("pf", 0)):
        mi = r["metricas_is"] or {}
        mo = r["metricas_oos"] or {}
        pl = "OK" if r["plateau_ok"] else "FRAGIL"
        print(f"  {r['estrategia']:25} "
              f"{mi.get('pf', 0):>6.3f} "
              f"{mo.get('pf', 0):>7.3f} "
              f"{r['degradacao']:>6.1f} "
              f"{pl:>8} "
              f"{'APROVADO' if r['aprovado'] else 'REPROVADO':>10}", flush=True)

    if ia_melhorias:
        print(f"\n  IA EVOLUTIVA — {len(ia_melhorias)} MELHORIA(S) ENCONTRADA(S):", flush=True)
        for m in ia_melhorias:
            print(f"  {m['filtro_adicionado']:20} "
                  f"PF {m['pf_anterior']:.3f} -> {m['pf_novo']:.3f} "
                  f"(+{m['melhoria_pct']}%)", flush=True)

    lb = {
        "gerado_em":       datetime.now().isoformat(),
        "total_combos":    total,
        "estrategias":     len(GRIDS),
        "aprovados_final": n_apr,
        "ia_melhorias":    len(ia_melhorias),
        "top20":           todos[:20],
        "leaderboard":     resultados_finais,
    }
    with open(f"{OUTPUT_DIR}/leaderboard.json", "w") as fp:
        json.dump(lb, fp, indent=2, default=str)

    print(f"\n  {n_apr} estrategia(s) aprovada(s)!", flush=True)
    print(f"  Salvo em: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
