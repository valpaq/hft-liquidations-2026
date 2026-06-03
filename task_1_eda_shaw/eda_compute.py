"""Memory-safe compute for the liquidation EDA. Heavy tables are never fully loaded: DuckDB
out-of-core for markout, streaming GROUP BY for 1s bins, the BBO mid series in numpy queried
with searchsorted, a reservoir sample for feature IC. Results cache to eda_results/.

CLI: python eda_compute.py {fast|markout|quirks|micro|conditional|leadlag|all}
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path
import numpy as np
import polars as pl


_HERE = Path(__file__).resolve().parent


def _data_root() -> str:
    """Find liquidation_task/data by walking up from this file (independent of the CWD)."""
    for d in (_HERE, *_HERE.parents):
        cand = d / "liquidation_task" / "data"
        if cand.is_dir():
            return str(cand)
    return "liquidation_task/data"


DATA = _data_root()
OUT = str(_HERE / "eda_results")
TMP = str(_HERE / ".eda_scratch" / "duck_tmp")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

SYMS = ["btc", "eth"]
TAUS = [30, 120, 300]
VAL_START_US = 1_769_904_000_000_000
EXCHANGE_2_LAG_US = 200_000
N_TRAIN_DAYS, N_VAL_DAYS = 62, 28

def trades_path(sym): return f"{DATA}/exchange_1_trades/perp_{sym}usdt.parquet"
def bbo_path(sym):    return f"{DATA}/exchange_1_booktickers/perp_{sym}usdt.parquet"
def binliq_path(sym): return f"{DATA}/exchange_1_liquidations/perp_{sym}usdt.parquet"
def bybliq_path(sym): return f"{DATA}/exchange_2_liquidations/{sym}usdt.parquet"

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def combined_liq_stream(sym: str) -> pl.DataFrame:
    """Exchange 1 + Exchange 2 liquidations for one symbol with Exchange 2 shifted +200 ms.

    The shift is applied before the streams are merged, so a feature at trade time t only sees
    Exchange 2 events whose true time is at most t - 200 ms (their 200 ms availability lag has passed).
    Sorted by available time.
    """
    bn = (pl.read_parquet(binliq_path(sym), columns=["timestamp", "side", "price", "amount"])
            .with_columns(ex=pl.lit("exchange_1")))
    bb = (pl.read_parquet(bybliq_path(sym), columns=["timestamp", "side", "price", "amount"])
            .with_columns((pl.col("timestamp") + EXCHANGE_2_LAG_US).alias("timestamp"), ex=pl.lit("exchange_2")))
    return (pl.concat([bn, bb]).sort("timestamp")
              .with_columns(notional=pl.col("price") * pl.col("amount")))

def causal_window_sum(src_ts: np.ndarray, prefix: np.ndarray, q: np.ndarray, w_us: int) -> np.ndarray:
    """Sum of a value over the half-open window [q-w, q) for each query time q.

    `prefix` is the cumulative sum of the (time-sorted) source value with a
    leading 0 (len = len(src_ts)+1). `searchsorted(..., 'left')` makes the upper
    bound STRICTLY < q (exclusive of the trade's own timestamp -> no leakage).
    """
    hi = np.searchsorted(src_ts, q, side="left")
    lo = np.searchsorted(src_ts, q - w_us, side="left")
    return prefix[hi] - prefix[lo]

def causal_count(src_ts: np.ndarray, q: np.ndarray, w_us: int) -> np.ndarray:
    return (np.searchsorted(src_ts, q, "left") - np.searchsorted(src_ts, q - w_us, "left")).astype(np.float64)

def asof_backward(src_ts: np.ndarray, src_val: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Forward-filled value: last src_val with src_ts <= q. NaN beyond coverage."""
    if len(src_ts) == 0:
        return np.full(len(q), np.nan)
    i = np.searchsorted(src_ts, q, side="right") - 1
    out = np.where(i >= 0, src_val[np.clip(i, 0, len(src_val) - 1)], np.nan)
    out = np.asarray(out, dtype=np.float64)
    out[(q < src_ts[0]) | (q > src_ts[-1])] = np.nan
    return out

def validate_feature(name: str, values: np.ndarray, n_trades: int, allow_nan: bool = True) -> dict:
    """Part-E validators: alignment, no inf, finite share, (optional) no NaN."""
    v = np.asarray(values, dtype=np.float64)
    n_nan = int(np.isnan(v).sum()); n_inf = int(np.isinf(v).sum())
    rep = {"feature": name, "aligned": len(v) == n_trades, "n_inf": n_inf,
           "n_nan": n_nan, "nan_%": round(100 * n_nan / max(len(v), 1), 3),
           "finite_min": float(np.nanmin(v)) if np.isfinite(v).any() else None,
           "finite_max": float(np.nanmax(v)) if np.isfinite(v).any() else None}
    rep["ok"] = rep["aligned"] and n_inf == 0 and (allow_nan or n_nan == 0)
    return rep

def compute_markout_baseline(batch_rows=20_000_000):
    """Full-data PnL_all(tau) by streaming trades in batches against an in-RAM BBO mid
    series (vectorised searchsorted). No sort, no disk spill: memory = BBO arrays (~1.6GB)
    + one trade batch. One pass over trades covers all 3 horizons."""
    art = f"{OUT}/markout_baseline.parquet"
    if os.path.exists(art):
        log(f"skip markout (exists: {art})"); return
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    rows = []
    for sym in SYMS:
        bbo = pl.read_parquet(bbo_path(sym), columns=["timestamp", "bid_price", "ask_price"])
        if not bbo["timestamp"].is_sorted():
            bbo = bbo.sort("timestamp")
        Bts = bbo["timestamp"].to_numpy()
        mid = ((bbo["bid_price"] + bbo["ask_price"]) / 2).to_numpy()
        del bbo
        acc = {(sp, tau): [0.0, 0.0, 0] for sp in ("train", "val") for tau in TAUS}
        t0 = time.perf_counter()
        pf = pq.ParquetFile(trades_path(sym))
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["timestamp", "side", "price", "amount"]):
            t = batch.column("timestamp").to_numpy(zero_copy_only=False)
            price = batch.column("price").to_numpy(zero_copy_only=False)
            amount = batch.column("amount").to_numpy(zero_copy_only=False)
            s = np.where(np.asarray(pc.equal(batch.column("side"), "buy")), 1.0, -1.0)
            w = np.minimum(price * amount, 100_000.0)
            is_train = t < VAL_START_US
            for tau in TAUS:
                mt = asof_backward(Bts, mid, t + tau * 1_000_000)
                pnl = -s * (mt - price) / price * 1e4 + 0.5
                ok = ~np.isnan(pnl)
                for sp, mask in (("train", is_train & ok), ("val", (~is_train) & ok)):
                    a = acc[(sp, tau)]
                    a[0] += float(w[mask].sum())
                    a[1] += float((w[mask] * pnl[mask]).sum())
                    a[2] += int(mask.sum())
        for (sp, tau), (wsum, wpnl, n) in acc.items():
            rows.append({"symbol": sym, "tau": tau, "split": sp, "n": n,
                         "wsum": wsum, "PnL_all_bps": wpnl / wsum})
        del Bts, mid
        log(f"markout {sym}: {time.perf_counter()-t0:.0f}s | "
            + " ".join(f"{tau}s val={acc[('val',tau)][1]/acc[('val',tau)][0]:+.3f}" for tau in TAUS))
    pl.DataFrame(rows).write_parquet(art)
    log(f"wrote {art}")

def _bbo_mid_arrays(sym):
    bbo = pl.read_parquet(bbo_path(sym), columns=["timestamp", "bid_price", "ask_price"])
    if not bbo["timestamp"].is_sorted():
        bbo = bbo.sort("timestamp")
    return bbo["timestamp"].to_numpy(), ((bbo["bid_price"] + bbo["ask_price"]) / 2).to_numpy()

def compute_event_study():
    art = f"{OUT}/event_study.parquet"
    if os.path.exists(art):
        log(f"skip event_study (exists)"); return
    out = []
    estaus = [5, 30, 120, 300]
    for sym in SYMS:
        df = combined_liq_stream(sym)
        new = (df["timestamp"] - df["timestamp"].shift(1)).fill_null(10**12) > 500_000
        df = df.with_columns(cid=new.cum_sum())
        cl = df.group_by("cid").agg(cnotl=pl.col("notional").sum(), csize=pl.len())
        ev = df.join(cl, on="cid").with_columns(
            s=pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0))
        bts, mid = _bbo_mid_arrays(sym)
        anchor = ev["timestamp"].to_numpy(); s = ev["s"].to_numpy()
        m0 = asof_backward(bts, mid, anchor)
        R = ev.select("cid", "ex", "cnotl", "csize",
                      split=pl.when(pl.col("timestamp") < VAL_START_US).then(pl.lit("train")).otherwise(pl.lit("val")))
        for tau in estaus:
            sg = s * (asof_backward(bts, mid, anchor + tau * 1_000_000) - m0) / m0 * 1e4
            R = R.with_columns(pl.Series(f"sg{tau}", sg))
        R = R.with_columns(bucket=pl.col("cnotl").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], allow_duplicates=True))
        agg = (R.group_by("bucket")
                .agg(n=pl.len(), cnotl_med=pl.col("cnotl").median(),
                     **{f"signed_{t}s": pl.col(f"sg{t}").fill_nan(None).mean() for t in estaus})
                .with_columns(symbol=pl.lit(sym), cut=pl.lit("cluster_notional_quintile"))
                .rename({"bucket": "group"}))
        out.append(agg)
        clx = (df.group_by("cid").agg(nb=(pl.col("ex") == "exchange_2").sum(), nn=(pl.col("ex") == "exchange_1").sum()))
        ex_map = clx.with_columns(klass=pl.when((pl.col("nb") > 0) & (pl.col("nn") > 0)).then(pl.lit("both"))
                                  .when(pl.col("nb") > 0).then(pl.lit("exchange_2_only")).otherwise(pl.lit("exchange_1_only")))
        Rx = R.join(ex_map.select("cid", "klass"), on="cid")
        aggx = (Rx.group_by("klass")
                  .agg(n=pl.len(), cnotl_med=pl.col("cnotl").median(),
                       **{f"signed_{t}s": pl.col(f"sg{t}").fill_nan(None).mean() for t in estaus})
                  .with_columns(symbol=pl.lit(sym), cut=pl.lit("exchange_membership"))
                  .rename({"klass": "group"}))
        out.append(aggx)
        log(f"event_study {sym}: {ev.height:,} events")
    pl.concat(out, how="diagonal_relaxed").write_parquet(art)
    log(f"wrote {art}")

def compute_adverse_selection():
    art = f"{OUT}/adverse_selection.json"
    if os.path.exists(art):
        log("skip adverse_selection (exists)"); return
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='6GB'; SET threads=4; SET temp_directory='{TMP}';")
    res = {}
    for sym in SYMS:
        bins = con.execute(f"""SELECT (timestamp//1000000) AS sec, sum(price*amount) AS notl
                               FROM read_parquet('{trades_path(sym)}') GROUP BY 1 ORDER BY 1""").pl()
        sec0 = int(bins["sec"].min()); span = int(bins["sec"].max()) - sec0 + 1
        C = np.zeros(span + 10)
        C[(bins["sec"] - sec0).to_numpy()] = bins["notl"].to_numpy()
        cum = np.concatenate([[0.0], np.cumsum(C)])
        baseline5 = C.sum() / span * 5
        sym_res = {"baseline_5s_usd": baseline5}
        for src, path in [("exchange_2", bybliq_path(sym)), ("exchange_1", binliq_path(sym))]:
            ts = pl.read_parquet(path, columns=["timestamp"])["timestamp"]
            ls = (ts // 1_000_000 - sec0).to_numpy()
            ls = ls[(ls >= 0) & (ls + 5 < len(cum))]
            post5 = cum[ls + 5] - cum[ls]
            sym_res[f"{src}_post5s_mean_usd"] = float(post5.mean())
            sym_res[f"{src}_post5s_median_usd"] = float(np.median(post5))
            sym_res[f"{src}_mult_mean"] = float(post5.mean() / baseline5)
            sym_res[f"{src}_mult_median"] = float(np.median(post5) / baseline5)
        res[sym] = sym_res
        log(f"adverse_selection {sym}: exchange_2 mult mean {sym_res['exchange_2_mult_mean']:.1f}x")
    json.dump(res, open(art, "w"), indent=2)
    log(f"wrote {art}")

FEATURES = ["liq_notional_30s", "liq_notional_120s", "liq_event_count_30s",
            "liq_side_imbalance_30s", "exchange_2_liq_notional_30s", "liq_velocity",
            "time_since_liq_s", "bbo_spread_bps", "bbo_imbalance", "mid_ret_5s"]

def build_feature_sample(sym, n=3_000_000, seed=42):
    art = f"{OUT}/feature_sample_{sym}.parquet"
    if os.path.exists(art):
        log(f"skip sample {sym} (exists)"); return pl.read_parquet(art)
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='6GB'; SET threads=4; SET temp_directory='{TMP}';")
    t0 = time.perf_counter()
    smp = con.execute(f"""SELECT timestamp, side, price, price*amount AS notional
                          FROM read_parquet('{trades_path(sym)}')
                          USING SAMPLE {n} ROWS (reservoir, {seed})""").pl().sort("timestamp")
    log(f"sample {sym}: {smp.height:,} trades in {time.perf_counter()-t0:.0f}s")
    t = smp["timestamp"].to_numpy()
    price = smp["price"].to_numpy()
    s = np.where(smp["side"].to_numpy() == "buy", 1.0, -1.0)
    w = np.minimum(smp["notional"].to_numpy(), 100_000)

    cl = combined_liq_stream(sym)
    lts = cl["timestamp"].to_numpy()
    notl = cl["notional"].to_numpy()
    pre_notl = np.concatenate([[0.0], np.cumsum(notl)])
    buy = np.where(cl["side"].to_numpy() == "buy", notl, 0.0)
    sel = np.where(cl["side"].to_numpy() == "sell", notl, 0.0)
    pre_buy = np.concatenate([[0.0], np.cumsum(buy)]); pre_sel = np.concatenate([[0.0], np.cumsum(sel)])
    bb = cl.filter(pl.col("ex") == "exchange_2")
    bts_l = bb["timestamp"].to_numpy(); pre_bb = np.concatenate([[0.0], np.cumsum(bb["notional"].to_numpy())])

    W30, W120 = 30_000_000, 120_000_000
    f = {}
    f["liq_notional_30s"]  = causal_window_sum(lts, pre_notl, t, W30)
    f["liq_notional_120s"] = causal_window_sum(lts, pre_notl, t, W120)
    f["liq_event_count_30s"] = causal_count(lts, t, W30)
    b30 = causal_window_sum(lts, pre_buy, t, W30); s30 = causal_window_sum(lts, pre_sel, t, W30)
    f["liq_side_imbalance_30s"] = np.where((b30 + s30) > 0, (b30 - s30) / (b30 + s30), 0.0)
    f["exchange_2_liq_notional_30s"] = causal_window_sum(bts_l, pre_bb, t, W30)
    f["liq_velocity"] = f["liq_notional_30s"] / (f["liq_notional_120s"] + 1.0)
    last_i = np.searchsorted(lts, t, "left") - 1
    f["time_since_liq_s"] = np.where(last_i >= 0, (t - lts[np.clip(last_i, 0, len(lts)-1)]) / 1e6, 1e9)

    bbo = pl.read_parquet(bbo_path(sym))
    if not bbo["timestamp"].is_sorted():
        bbo = bbo.sort("timestamp")
    Bts = bbo["timestamp"].to_numpy()
    mid = ((bbo["bid_price"] + bbo["ask_price"]) / 2).to_numpy()
    spread = ((bbo["ask_price"] - bbo["bid_price"]) / ((bbo["bid_price"] + bbo["ask_price"]) / 2) * 1e4).to_numpy()
    imb = ((bbo["bid_amount"] - bbo["ask_amount"]) / (bbo["bid_amount"] + bbo["ask_amount"])).to_numpy()
    f["bbo_spread_bps"] = asof_backward(Bts, spread, t)
    f["bbo_imbalance"] = asof_backward(Bts, imb, t)
    mid_now = asof_backward(Bts, mid, t); mid_5ago = asof_backward(Bts, mid, t - 5_000_000)
    f["mid_ret_5s"] = (mid_now - mid_5ago) / mid_5ago * 1e4
    del spread, imb

    maxbb = Bts[-1]
    targets = {}
    for tau in TAUS:
        mt = asof_backward(Bts, mid, t + tau * 1_000_000)
        pnl = -s * (mt - price) / price * 1e4 + 0.5
        pnl[(t + tau * 1_000_000) > maxbb] = np.nan
        targets[f"pnl_{tau}"] = pnl

    cols = {"timestamp": t, "w": w, "s": s, **{k: v for k, v in f.items()}, **targets,
            "split": np.where(t < VAL_START_US, "train", "val")}
    out = pl.DataFrame(cols)
    out.write_parquet(art)
    log(f"wrote {art}")
    return out

def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100: return np.nan
    ra = pl.Series(a[m]).rank(method="average").to_numpy().astype(np.float64)
    rb = pl.Series(b[m]).rank(method="average").to_numpy().astype(np.float64)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else np.nan

def _psi(train, val, bins=10):
    t = train[np.isfinite(train)]; v = val[np.isfinite(val)]
    if len(t) < 100 or len(v) < 100: return np.nan
    edges = np.quantile(t, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    tp = np.histogram(t, edges)[0] / len(t) + 1e-6
    vp = np.histogram(v, edges)[0] / len(v) + 1e-6
    return float(((vp - tp) * np.log(vp / tp)).sum())

def compute_ic_psi():
    ic_rows, psi_rows, dec_rows, val_reports, rule_rows, corr_payload = [], [], [], [], [], {}
    for sym in SYMS:
        smp = build_feature_sample(sym)
        n = smp.height
        train = smp.filter(pl.col("split") == "train"); val = smp.filter(pl.col("split") == "val")
        for feat in FEATURES:
            val_reports.append({"symbol": sym, **validate_feature(feat, smp[feat].to_numpy(), n)})
        for feat in FEATURES:
            psi_rows.append({"symbol": sym, "feature": feat,
                             "psi": _psi(train[feat].to_numpy(), val[feat].to_numpy())})
            for split, part in [("train", train), ("val", val)]:
                fv = part[feat].to_numpy()
                for tau in TAUS:
                    ic_rows.append({"symbol": sym, "feature": feat, "split": split, "tau": tau,
                                    "IC": _spearman(fv, part[f"pnl_{tau}"].to_numpy())})
            dpart = val.with_columns(dec=pl.col(feat).qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True))
            dd = dpart.group_by("dec").agg(pnl120=pl.col("pnl_120").fill_nan(None).mean(),
                                           fmed=pl.col(feat).median(), n=pl.len()).sort("dec")
            for r in dd.iter_rows(named=True):
                dec_rows.append({"symbol": sym, "feature": feat, **r})
        M = np.zeros((len(FEATURES), len(FEATURES)))
        arrs = [smp[fz].to_numpy() for fz in FEATURES]
        for i in range(len(FEATURES)):
            for j in range(len(FEATURES)):
                M[i, j] = 1.0 if i == j else _spearman(arrs[i], arrs[j])
        corr_payload[sym] = {"features": FEATURES, "matrix": M.tolist()}
        def score_rule(rule, param, filt_expr, tau):
            d = smp.select(w=pl.col("w"), pnl=pl.col(f"pnl_{tau}"), filt=filt_expr,
                           split=pl.col("split")).filter(pl.col("pnl").is_not_nan())
            for split in ["train", "val"]:
                ds = d.filter(pl.col("split") == split)
                wpa = (ds["w"] * ds["pnl"]).sum() / ds["w"].sum()
                kept = ds.filter(~pl.col("filt"))
                wpk = ((kept["w"] * kept["pnl"]).sum() / kept["w"].sum()) if kept.height else float("nan")
                rule_rows.append({"symbol": sym, "rule": rule, "param": param, "tau": tau, "split": split,
                                  "PnL_all": wpa, "PnL_kept": wpk, "Score": wpk - wpa,
                                  "kept_%": round(100 * kept.height / ds.height, 1)})
        for tau in TAUS:
            for thr in [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]:
                score_rule("symmetric_notional", float(thr), pl.col("liq_notional_30s") >= thr, tau)
            for thr in [50_000, 100_000, 200_000, 500_000]:
                adverse = (pl.col("s") * pl.col("liq_side_imbalance_30s") > 0) & (pl.col("liq_notional_30s") >= thr)
                score_rule("directional_adverse", float(thr), adverse, tau)
        log(f"ic/psi {sym} done")
    pl.DataFrame(ic_rows).write_parquet(f"{OUT}/feature_ic.parquet")
    pl.DataFrame(psi_rows).write_parquet(f"{OUT}/feature_psi.parquet")
    pl.DataFrame(dec_rows).write_parquet(f"{OUT}/feature_decile.parquet")
    pl.DataFrame(val_reports).write_parquet(f"{OUT}/feature_validation.parquet")
    pl.DataFrame(rule_rows).write_parquet(f"{OUT}/baseline_rule.parquet")
    json.dump(corr_payload, open(f"{OUT}/feature_corr.json", "w"))
    log("wrote feature_ic / psi / decile / validation / baseline_rule / corr")

def compute_quirks():
    art = f"{OUT}/quirks_summary.json"
    if os.path.exists(art):
        log("skip quirks (exists)"); return
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='5GB'; SET threads=4; SET temp_directory='{TMP}';")
    summ = {"liquidations": {}, "bbo": {}, "trades": {}}
    FUND = np.array([0, 28800, 57600, 86400])

    for ex, pf in [("exchange_1", binliq_path), ("exchange_2", bybliq_path)]:
        for sym in SYMS:
            d = pl.read_parquet(pf(sym))
            ts = d["timestamp"].sort().to_numpy()
            gaps = np.diff(ts) / 1e6
            sid = (ts // 1_000_000) % 86400
            dmin = np.min(np.abs(sid[:, None] - FUND[None, :]), axis=1)
            dow = (ts // 86_400_000_000 + 4) % 7
            summ["liquidations"][f"{ex}_{sym}"] = {
                "rows": d.height,
                "pct_ms_granular": float((d["timestamp"] % 1000 == 0).mean() * 100),
                "dup_ts_pct": float(d["timestamp"].is_duplicated().mean() * 100),
                "exact_dup_rows": int(d.is_duplicated().sum()),
                "min_gap_s": float(gaps.min()), "p1_gap_s": float(np.quantile(gaps, 0.01)),
                "median_gap_s": float(np.median(gaps)),
                "sell_pct": float((d["side"] == "sell").mean() * 100),
                "funding_within60_pct": float((dmin < 60).mean() * 100),
                "weekend_pct": float(np.isin(dow, [5, 6]).mean() * 100),
            }

    gap_rows = []
    for sym in SYMS:
        p = bbo_path(sym)
        r = con.execute(f"""SELECT count(*) n, sum((bid_price>ask_price)::INT) crossed,
            sum((bid_price=ask_price)::INT) lck, sum((bid_amount<=0 OR ask_amount<=0)::INT) zero_sz,
            min(ask_price-bid_price) tick, avg((timestamp%1000=0)::INT)*100 ms
            FROM read_parquet('{p}')""").fetchone()
        g = con.execute(f"""WITH q AS (SELECT (timestamp-lag(timestamp) OVER(ORDER BY timestamp))/1000.0 ms
            FROM read_parquet('{p}')) SELECT max(ms)/1000 max_gap_s,
            count(*) FILTER(WHERE ms>=10000) n_gap_10s_plus FROM q WHERE ms IS NOT NULL""").fetchone()
        summ["bbo"][sym] = {"rows": r[0], "crossed": int(r[1]), "locked": int(r[2]),
                            "zero_size": int(r[3]), "tick": float(r[4]), "pct_ms_granular": float(r[5]),
                            "max_gap_s": float(g[0]), "n_gaps_ge_10s": int(g[1]), "distinct_ts_pct": 100.0}
        h = con.execute(f"""WITH q AS (SELECT (timestamp-lag(timestamp) OVER(ORDER BY timestamp))/1000.0 ms
            FROM read_parquet('{p}'))
            SELECT CASE WHEN ms<1 THEN '0 <1ms' WHEN ms<10 THEN '1 1-10ms' WHEN ms<100 THEN '2 10-100ms'
                WHEN ms<1000 THEN '3 0.1-1s' WHEN ms<10000 THEN '4 1-10s' ELSE '5 >10s' END bucket,
                count(*) c FROM q WHERE ms IS NOT NULL GROUP BY 1""").pl().with_columns(symbol=pl.lit(sym))
        gap_rows.append(h)
    pl.concat(gap_rows).write_parquet(f"{OUT}/quirks_bbo_gap_hist.parquet")

    burst_rows, price_rows = [], []
    for sym in SYMS:
        p = trades_path(sym)
        b = con.execute(f"""WITH t AS (SELECT timestamp, count(*) c FROM read_parquet('{p}') GROUP BY 1)
            SELECT count(*) distinct_ts, max(c) max_fills,
                   sum((c>=50)::INT) ms_ge50 FROM t""").fetchone()
        n = con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
        g = con.execute(f"""WITH q AS (SELECT (timestamp-lag(timestamp) OVER(ORDER BY timestamp))/1e6 s
            FROM read_parquet('{p}')) SELECT max(s) mx, count(*) FILTER(WHERE s>10) n10,
            count(*) FILTER(WHERE s>5) n5 FROM q WHERE s IS NOT NULL""").fetchone()
        summ["trades"][sym] = {"rows": n, "distinct_ts": int(b[0]),
                               "distinct_ts_pct": float(100 * b[0] / n), "max_fills_per_ms": int(b[1]),
                               "ms_with_ge50_fills": int(b[2]), "max_gap_s": float(g[0]),
                               "n_gaps_gt_10s": int(g[1]), "n_gaps_gt_5s": int(g[2])}
        bh = con.execute(f"""WITH t AS (SELECT timestamp, count(*) c FROM read_parquet('{p}') GROUP BY 1)
            SELECT least(c,60) fills, count(*) n FROM t GROUP BY 1 ORDER BY 1""").pl().with_columns(symbol=pl.lit(sym))
        burst_rows.append(bh)
        pr = con.execute(f"""SELECT (timestamp//86400000000) day_idx, arg_max(price,timestamp) last_price,
            min(price) lo, max(price) hi, sum(price*amount) notional, count(*) n
            FROM read_parquet('{p}') GROUP BY 1 ORDER BY 1""").pl().with_columns(symbol=pl.lit(sym))
        price_rows.append(pr)
        log(f"quirks trades {sym} done")
    pl.concat(burst_rows).write_parquet(f"{OUT}/quirks_burst_hist.parquet")
    pl.concat(price_rows).write_parquet(f"{OUT}/quirks_price_daily.parquet")
    json.dump(summ, open(art, "w"), indent=2)
    log(f"wrote {art} + quirks_bbo_gap_hist / quirks_burst_hist / quirks_price_daily")


def _spearman_full(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    rx = pl.Series(x[m]).rank(method="average").to_numpy().astype(np.float64)
    ry = pl.Series(y[m]).rank(method="average").to_numpy().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def _us(s):
    import datetime as dt
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() * 1e6)


def compute_micro():
    """Microstructure: order-flow long memory, microprice/imbalance predictiveness,
    return autocorrelation, and the Exchange 1 liquidation-price offset convention."""
    art = f"{OUT}/micro_summary.json"
    if os.path.exists(art):
        log("skip micro (exists)"); return
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='5GB'; SET threads=4; SET temp_directory='{TMP}';")
    WINDOWS = {"calm (2026-01-10)": ("2026-01-10T00:00", "2026-01-10T06:00"),
               "stress (2026-02-04)": ("2026-02-04T00:00", "2026-02-04T06:00")}
    summary = {}

    acf_rows, ret_rows = [], []
    for lab, (t0, t1) in WINDOWS.items():
        tr = con.execute(f"""SELECT timestamp//1000 ms, sum(CASE WHEN side='buy' THEN amount ELSE -amount END) netv,
            arg_max(price,timestamp) px FROM read_parquet('{trades_path('btc')}')
            WHERE timestamp>={_us(t0)} AND timestamp<{_us(t1)} GROUP BY 1 ORDER BY 1""").pl()
        sign = np.sign(tr["netv"].to_numpy()); sign = sign[sign != 0]
        sc = sign - sign.mean(); denom = float((sc * sc).sum())
        for k in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
            acf_rows.append({"window": lab, "lag": k,
                             "acf": float((sc[:-k] * sc[k:]).sum() / denom)})
        r = np.diff(np.log(tr["px"].to_numpy())); r = r[np.isfinite(r) & (r != 0)]
        ret_rows.append({"window": lab, "series": "trade_price",
                         "lag1_autocorr": float(np.corrcoef(r[:-1], r[1:])[0, 1])})
    pl.DataFrame(acf_rows).write_parquet(f"{OUT}/micro_orderflow_acf.parquet")
    pl.DataFrame(ret_rows).write_parquet(f"{OUT}/micro_return_acf.parquet")

    bbo = pl.read_parquet(bbo_path("btc"))
    if not bbo["timestamp"].is_sorted():
        bbo = bbo.sort("timestamp")
    bts = bbo["timestamp"].to_numpy()
    bid, ask = bbo["bid_price"].to_numpy(), bbo["ask_price"].to_numpy()
    bsz, asz = bbo["bid_amount"].to_numpy(), bbo["ask_amount"].to_numpy()
    mid = (bid + ask) / 2
    del bbo
    imb = (bsz - asz) / (bsz + asz)
    micro_dev = (ask * bsz + bid * asz) / (bsz + asz) - mid
    n = len(bts); idx = np.arange(0, n, max(1, n // 4_000_000))
    ic_rows = []
    for h in [1, 5, 30, 120]:
        tgt = bts[idx] + h * 1_000_000
        j = np.clip(np.searchsorted(bts, tgt, "right") - 1, 0, n - 1)
        fwd = (mid[j] - mid[idx]) / mid[idx] * 1e4
        fwd[tgt > bts[-1]] = np.nan
        ic_rows.append({"horizon_s": h, "ic_imbalance": _spearman_full(imb[idx], fwd),
                        "ic_microprice": _spearman_full(micro_dev[idx], fwd)})
    pl.DataFrame(ic_rows).write_parquet(f"{OUT}/micro_imbalance_ic.parquet")
    del imb, micro_dev, bid, ask, bsz, asz

    off_rows = []
    for sym, bts_, mid_ in [("btc", bts, mid), ("eth", None, None)]:
        if sym == "eth":
            b = pl.read_parquet(bbo_path("eth"))
            if not b["timestamp"].is_sorted():
                b = b.sort("timestamp")
            bts_ = b["timestamp"].to_numpy(); mid_ = ((b["bid_price"] + b["ask_price"]) / 2).to_numpy(); del b
        lq = pl.read_parquet(binliq_path(sym))
        i = np.clip(np.searchsorted(bts_, lq["timestamp"].to_numpy(), "right") - 1, 0, len(bts_) - 1)
        off = (lq["price"].to_numpy() - mid_[i]) / mid_[i] * 1e4
        for side, m in [("buy", (lq["side"] == "buy").to_numpy()), ("sell", (lq["side"] == "sell").to_numpy())]:
            for v in off[m]:
                off_rows.append({"symbol": sym, "side": side, "offset_bps": float(v)})
    pl.DataFrame(off_rows).write_parquet(f"{OUT}/micro_liq_offset.parquet")
    summary["liq_offset_median_abs_bps"] = float(np.median(np.abs(pl.DataFrame(off_rows)["offset_bps"].to_numpy())))
    json.dump(summary, open(art, "w"), indent=2)
    log(f"wrote micro_* artifacts")


def compute_conditional():
    """Does the liquidation signal add information beyond queue imbalance? Compares, on the cached
    feature sample: short-horizon imbalance adverse-selection vs a directional liquidation
    signal `liq_against = s * liq_side_imbalance_30s` (>0 = liquidation flow pushed against the
    maker's filled side). Residualises markout within imbalance deciles to test orthogonality,
    and reports the maker-markout spread (filled against vs with the cascade) in active-liq windows."""
    art = f"{OUT}/conditional_means.parquet"
    if os.path.exists(art):
        log("skip conditional (exists)"); return
    ic_rows, mean_rows = [], []
    for sym in SYMS:
        full = pl.read_parquet(f"{OUT}/feature_sample_{sym}.parquet")
        for split in ("train", "val"):
            d = full.filter(pl.col("split") == split)
            s = d["s"].to_numpy()
            imb_against = s * d["bbo_imbalance"].to_numpy()
            liq_against = s * d["liq_side_imbalance_30s"].to_numpy()
            liqn = d["liq_notional_30s"].to_numpy()
            pos = liqn[liqn > 0]
            active = liqn >= np.quantile(pos, 0.5) if len(pos) else liqn > 0
            dec = pl.Series(imb_against).rank(method="ordinal").to_numpy()
            dec = np.clip((dec / (len(dec) + 1) * 10).astype(int), 0, 9)
            for tau in TAUS:
                pnl = d[f"pnl_{tau}"].to_numpy()
                resid = pnl.astype(float).copy()
                for k in range(10):
                    m = (dec == k) & np.isfinite(pnl)
                    if m.any():
                        resid[m] = pnl[m] - np.nanmean(pnl[m])
                ic_rows.append({"symbol": sym, "split": split, "tau": tau,
                                "ic_imb_against": _spearman_full(imb_against, pnl),
                                "ic_liq_against_raw": _spearman_full(liq_against, pnl),
                                "ic_liq_against_resid": _spearman_full(liq_against, resid),
                                "ic_liqn_raw": _spearman_full(liqn, pnl),
                                "ic_liqn_resid": _spearman_full(liqn, resid)})
                ag = pnl[active & (liq_against > 0.2)]; wi = pnl[active & (liq_against < -0.2)]
                mean_rows.append({"symbol": sym, "split": split, "tau": tau,
                                  "baseline": float(np.nanmean(pnl)),
                                  "against": float(np.nanmean(ag)), "with_push": float(np.nanmean(wi)),
                                  "spread": float(np.nanmean(ag) - np.nanmean(wi)),
                                  "n_against": int(len(ag)), "n_with": int(len(wi))})
        log(f"conditional {sym} done")
    pl.DataFrame(ic_rows).write_parquet(f"{OUT}/conditional_ic.parquet")
    pl.DataFrame(mean_rows).write_parquet(art)
    log("wrote conditional_ic / conditional_means")


def compute_leadlag():
    """Impulse response of the Exchange 1 mid to a liquidation print, vs lag relative to the print
    time (real time, not +200ms-shifted: we measure the market, not availability). Shows whether
    the print leads or lags the move."""
    art = f"{OUT}/micro_leadlag.parquet"
    if os.path.exists(art):
        log("skip leadlag (exists)"); return
    sym = "btc"
    bbo = pl.read_parquet(bbo_path(sym), columns=["timestamp", "bid_price", "ask_price"])
    if not bbo["timestamp"].is_sorted():
        bbo = bbo.sort("timestamp")
    bts = bbo["timestamp"].to_numpy()
    mid = ((bbo["bid_price"] + bbo["ask_price"]) / 2).to_numpy()
    del bbo

    def asof(q):
        i = np.clip(np.searchsorted(bts, q, "right") - 1, 0, len(bts) - 1)
        return np.where((q >= bts[0]) & (q <= bts[-1]), mid[i], np.nan)

    lags = [-2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 5]
    rows = []
    for ex, pf in [("exchange_2", bybliq_path), ("exchange_1", binliq_path)]:
        lq = pl.read_parquet(pf(sym))
        L = lq["timestamp"].to_numpy()
        s = np.where(lq["side"].to_numpy() == "buy", 1.0, -1.0)
        notl = (lq["price"] * lq["amount"]).to_numpy()
        m0 = asof(L)
        for bucket, mask in [("all", np.ones(len(L), bool)),
                             ("top_decile", notl >= np.quantile(notl, 0.9))]:
            for lag in lags:
                r = s * (asof(L + int(lag * 1e6)) - m0) / m0 * 1e4
                rows.append({"exchange": ex, "bucket": bucket, "lag_s": lag,
                             "response_bps": float(np.nanmean(r[mask])), "n": int(mask.sum())})
        log(f"leadlag {ex} done")
    pl.DataFrame(rows).write_parquet(art)
    log(f"wrote {art}")


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("conditional", "all"):
        compute_conditional()
    if part in ("leadlag", "all"):
        compute_leadlag()
    if part in ("fast", "all"):
        compute_event_study(); compute_adverse_selection(); compute_ic_psi()
    if part in ("quirks", "all"):
        compute_quirks()
    if part in ("micro", "all"):
        compute_micro()
    if part in ("markout", "all"):
        compute_markout_baseline()
    log("DONE")

if __name__ == "__main__":
    main()
