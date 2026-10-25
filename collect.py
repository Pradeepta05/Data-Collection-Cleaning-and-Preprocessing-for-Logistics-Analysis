import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output2")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
R = {}   # results that I will quote in the report

# =============================================================================
# PART A - SIMULATED DATA COLLECTION
# =============================================================================
N = 5000
modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
sched = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
base = {"Standard Class": 3.5, "Second Class": 2.2, "First Class": 1.4, "Same Day": 0.3}
cats = ["Electronics", "Apparel", "Grocery", "Furniture", "Sports"]
price_med = {"Electronics": 120, "Apparel": 40, "Grocery": 25, "Furniture": 300, "Sports": 60}
wt_med = {"Electronics": 2.0, "Apparel": 0.5, "Grocery": 3.0, "Furniture": 18.0, "Sports": 4.0}
regions = ["Central", "East", "North", "South", "West"]

true = pd.DataFrame({"order_id": np.arange(100001, 100001 + N)})
true["order_date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D")
true["shipping_mode"] = rng.choice(modes, N, p=[0.55, 0.20, 0.15, 0.10])
true["region"] = rng.choice(regions, N)
true["product_category"] = rng.choice(cats, N)
true["quantity"] = rng.integers(1, 6, N)
true["product_price"] = [round(rng.lognormal(np.log(price_med[c]), 0.35), 2) for c in true.product_category]
true["discount_rate"] = rng.choice([0, 0.05, 0.10, 0.15, 0.20, 0.25], N)
true["weight_kg"] = [round(rng.lognormal(np.log(wt_med[c]), 0.3), 2) for c in true.product_category]
true["distance_km"] = rng.gamma(3, 90, N).clip(5, 900).round(1)
true["scheduled_days"] = true.shipping_mode.map(sched)
true["actual_days"] = (true.shipping_mode.map(base) + true.distance_km / 150
                       + rng.normal(0, 0.8, N)).round().clip(0, 12).astype(int)
true["sales"] = (true.product_price * true.quantity * (1 - true.discount_rate)).round(2)
true["delivery_status"] = np.select(
    [true.actual_days > true.scheduled_days, true.actual_days < true.scheduled_days],
    ["Late delivery", "Advance shipping"], "Shipping on time")

# ---- now make a messy copy, the way real exports look ------------------------
raw = true.copy()

# dates saved in three different formats (as text)
fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"], N, p=[0.6, 0.3, 0.1])
raw["order_date"] = [d.strftime(f) for d, f in zip(true.order_date, fmt)]

# shipping mode typed in many different ways
def messy_mode(m):
    r = rng.random()
    if r < 0.10: return m.lower()
    if r < 0.17: return m.upper()
    if r < 0.24: return m + " "
    if r < 0.27: return {"Standard Class": "Std Class", "Second Class": "2nd Class",
                         "First Class": "1st Class", "Same Day": "Sameday"}[m]
    return m
raw["shipping_mode"] = raw.shipping_mode.map(messy_mode)

# some prices saved as text with $ and commas
raw["product_price"] = raw.product_price.astype(object)
txt = rng.choice(N, int(0.03 * N), replace=False)
raw.loc[txt, "product_price"] = raw.loc[txt, "product_price"].map(lambda v: f"${v:,.2f}")

# outliers and impossible values
price_typo = rng.choice(N, 30, replace=False)       # decimal point in wrong place
for i in price_typo:
    v = raw.at[i, "product_price"]
    if not isinstance(v, str):
        raw.at[i, "product_price"] = v * 100
raw.loc[rng.choice(N, 25, replace=False), "actual_days"] = rng.integers(45, 91, 25)
raw.loc[rng.choice(N, 15, replace=False), "weight_kg"] = -999          # sentinel code
raw.loc[rng.choice(N, 15, replace=False), "weight_kg"] *= 1000         # grams typed as kg
raw.loc[rng.choice(N, 10, replace=False), "distance_km"] = 99999
raw.loc[rng.choice(N, 8, replace=False), "quantity"] = 0
raw.loc[rng.choice(N, 6, replace=False), "quantity"] = 500

# wrong status labels (does not agree with the days)
bad = rng.choice(N, int(0.03 * N), replace=False)
raw.loc[bad, "delivery_status"] = rng.choice(["Late delivery", "Advance shipping", "Shipping on time"], len(bad))

# missing values (grocery weights go missing more often -> not purely random)
miss_w = np.where(raw.product_category == "Grocery", 0.15, 0.06)
raw.loc[rng.random(N) < miss_w, "weight_kg"] = np.nan
for col, p in [("distance_km", 0.04), ("discount_rate", 0.03), ("shipping_mode", 0.02),
               ("actual_days", 0.03), ("product_category", 0.01), ("sales", 0.02)]:
    raw.loc[rng.random(N) < p, col] = np.nan

# useless columns
raw["coupon_code"] = pd.Series("SAVE" + rng.integers(10, 99, N).astype(str)).where(rng.random(N) >= 0.72)
raw["warehouse_id"] = "WH-01"

# duplicated rows (same record exported twice)
raw = pd.concat([raw, raw.sample(120, random_state=3)]).sample(frac=1, random_state=5).reset_index(drop=True)
raw.columns = ["Order ID", "Order Date", "Shipping Mode ", "Region", "Product Category", "Quantity",
               "Product Price", "Discount Rate", "Weight (kg)", "Distance km", "Scheduled Days",
               "Actual Days", "Sales", "Delivery Status", "Coupon Code", "Warehouse ID"]
raw.to_csv(f"{OUT}/shipments_raw.csv", index=False)
R["raw_shape"] = list(raw.shape)

# =============================================================================
# PART B - THE PIPELINE
# =============================================================================
df = pd.read_csv(f"{OUT}/shipments_raw.csv")

# ---- STEP 1: first look ------------------------------------------------------
before_missing = (df.isna().mean() * 100).round(1)
R["step1"] = {"rows": len(df), "cols": df.shape[1],
              "non_numeric_cols": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
              "missing_pct": before_missing.to_dict(), "missing_total_cells": int(df.isna().sum().sum())}

# ---- STEP 2: tidy names and text --------------------------------------------
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
df = df.rename(columns={"weight_kg": "weight_kg", "distance_km": "distance_km"})
mode_map = {"standard class": "Standard Class", "std class": "Standard Class",
            "second class": "Second Class", "2nd class": "Second Class",
            "first class": "First Class", "1st class": "First Class",
            "same day": "Same Day", "sameday": "Same Day"}
R["step2"] = {"shipping_mode_variants_before": int(df.shipping_mode.nunique())}
df["shipping_mode"] = df.shipping_mode.str.strip().str.lower().map(mode_map)
R["step2"]["shipping_mode_variants_after"] = int(df.shipping_mode.nunique())

# ---- STEP 3: fix data types --------------------------------------------------
df["product_price"] = pd.to_numeric(df.product_price.astype(str).str.replace(r"[$,]", "", regex=True),
                                    errors="coerce")


def parse_dates(s):
    out = pd.Series(pd.NaT, index=s.index)
    for f in ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"]:          # try each known format
        out = out.fillna(pd.to_datetime(s, format=f, errors="coerce"))
    return out


df["order_date"] = parse_dates(df.order_date)
R["step3"] = {"unparsed_dates": int(df.order_date.isna().sum())}

# ---- STEP 4: duplicates ------------------------------------------------------
n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
R["step4_duplicates_removed"] = n0 - len(df)

# ---- STEP 5: impossible values -> NaN (domain rules) -------------------------
flags = {}
flags["weight_sentinel_or_negative"] = int((df.weight_kg <= 0).sum())
df.loc[df.weight_kg <= 0, "weight_kg"] = np.nan
flags["weight_unit_error_grams_as_kg"] = int((df.weight_kg > 100).sum())
df.loc[df.weight_kg > 100, "weight_kg"] = df.weight_kg / 1000       # convert back to kg
flags["distance_impossible"] = int((df.distance_km > 2000).sum())
df.loc[df.distance_km > 2000, "distance_km"] = np.nan
flags["quantity_invalid"] = int(((df.quantity < 1) | (df.quantity > 50)).sum())
df.loc[(df.quantity < 1) | (df.quantity > 50), "quantity"] = np.nan
flags["actual_days_over_30"] = int((df.actual_days > 30).sum())
df.loc[df.actual_days > 30, "actual_days"] = np.nan
R["step5"] = flags

# ---- STEP 6: cross-field checks ---------------------------------------------
expected = df.product_price * df.quantity * (1 - df.discount_rate)
mismatch = ((expected - df.sales).abs() / df.sales > 0.05) & df.sales.notna() & expected.notna()
R["step6_price_sales_mismatch"] = int(mismatch.sum())
# price typo: trust sales (it was recorded separately), rebuild the price from it
df.loc[mismatch, "product_price"] = (df.sales / (df.quantity * (1 - df.discount_rate)))[mismatch].round(2)

# ---- STEP 7: outlier detection on what is left (IQR and z-score) -------------
def iqr_flags(s):
    q1, q3 = s.quantile([0.25, 0.75])
    i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)


def z_flags(s):
    return ((s - s.mean()) / s.std()).abs() > 3


# demonstration on the RAW price column (before step 6) to compare the two methods
raw_price = pd.to_numeric(raw["Product Price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
R["step7_demo_raw_price"] = {"iqr_flagged": int(iqr_flags(raw_price.dropna()).sum()),
                             "zscore_flagged": int(z_flags(raw_price.dropna()).sum()),
                             "true_typos_injected": 30}
# on what is left, cap (winsorise) at the IQR fences instead of deleting rows.
# IMPORTANT: do it inside each product category, because a 300 price is normal for
# furniture but strange for groceries. One global fence would damage real values.
capped = {}
for col in ["product_price", "weight_kg"]:
    g = df.groupby("product_category")[col]
    q1 = g.transform(lambda s: s.quantile(0.25)); q3 = g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    capped[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    df[col] = df[col].clip(lower=lo, upper=hi)
capped["distance_km"] = 0     # long distances are real, the rule in step 5 already removed the impossible ones
R["step7_capped_after_rules"] = capped

# ---- STEP 8: missing values --------------------------------------------------
# a) drop columns that are useless
drop_cols = [c for c in df.columns if df[c].isna().mean() > 0.6 or df[c].nunique() <= 1]
R["step8_dropped_columns"] = drop_cols
df = df.drop(columns=drop_cols)
# b) target column cannot be guessed -> drop those rows
n0 = len(df)
df = df.dropna(subset=["actual_days"]).reset_index(drop=True)
R["step8_rows_dropped_missing_target"] = n0 - len(df)
# c) keep a flag so the model knows which values were filled in
df["weight_was_missing"] = df.weight_kg.isna().astype(int)
before_fill = df.copy()
# d) fill
df["product_category"] = df.product_category.fillna(df.product_category.mode()[0])
df["shipping_mode"] = df.shipping_mode.fillna(df.shipping_mode.mode()[0])
df["scheduled_days"] = df.scheduled_days.fillna(df.shipping_mode.map(sched))
df["weight_kg"] = df.weight_kg.fillna(df.groupby("product_category").weight_kg.transform("median"))
df["distance_km"] = df.distance_km.fillna(df.groupby("region").distance_km.transform("median"))
df["quantity"] = df.quantity.fillna(df.quantity.median())
df["discount_rate"] = df.discount_rate.fillna(0.0)
df["sales"] = df.sales.fillna((df.product_price * df.quantity * (1 - df.discount_rate)).round(2))
after_missing = (df.isna().mean() * 100).round(1)

# how good was the group-median choice? compare 3 ways of filling weight
tw = true.set_index("order_id").weight_kg
idx = before_fill.index[before_fill.weight_was_missing == 1]
truth = tw.loc[before_fill.loc[idx, "order_id"]].values
m_mean = np.full(len(idx), before_fill.weight_kg.mean())
m_med = np.full(len(idx), before_fill.weight_kg.median())
m_grp = before_fill.loc[idx, "product_category"].map(before_fill.groupby("product_category").weight_kg.median()).values
R["step8_imputation_mae_kg"] = {
    "global_mean": round(float(np.nanmean(np.abs(m_mean - truth))), 2),
    "global_median": round(float(np.nanmean(np.abs(m_med - truth))), 2),
    "category_median": round(float(np.nanmean(np.abs(m_grp - truth))), 2),
    "n_filled": int(len(idx))}

# ---- STEP 9: consistency + new features -------------------------------------
df["delay_days"] = df.actual_days - df.scheduled_days
df["is_late"] = (df.delay_days > 0).astype(int)
new_status = np.select([df.delay_days > 0, df.delay_days < 0], ["Late delivery", "Advance shipping"], "Shipping on time")
R["step9_status_label_wrong"] = int((new_status != df.delivery_status).sum())
df["delivery_status"] = new_status
df["order_month"] = df.order_date.dt.month
df["order_dow"] = df.order_date.dt.day_of_week
df["log_price"] = np.log1p(df.product_price)

# ---- STEP 10: encoding -------------------------------------------------------
enc = pd.get_dummies(df, columns=["shipping_mode", "region", "product_category"], dtype=int)

# ---- STEP 11: scaling (fit on TRAIN only to avoid leakage) -------------------
num = ["product_price", "weight_kg", "distance_km"]
tr, te = train_test_split(enc, test_size=0.2, random_state=42)
scalers = {"MinMax": MinMaxScaler(), "Standard": StandardScaler(), "Robust": RobustScaler()}
scale_table = {}
for name, sc in scalers.items():
    sc.fit(tr[num])
    t = pd.DataFrame(sc.transform(tr[num]), columns=num)
    scale_table[name] = {c: [round(t[c].min(), 2), round(t[c].max(), 2), round(t[c].mean(), 2), round(t[c].std(), 2)] for c in num}
scale_table["Raw"] = {c: [round(tr[c].min(), 2), round(tr[c].max(), 2), round(tr[c].mean(), 2), round(tr[c].std(), 2)] for c in num}
R["step11_scaling"] = scale_table
R["step11_skew"] = {"price_skew": round(float(df.product_price.skew()), 2),
                    "log_price_skew": round(float(df.log_price.skew()), 2)}
final_scaler = StandardScaler().fit(tr[num])
enc_scaled = enc.copy()
enc_scaled[num] = final_scaler.transform(enc[num])

# ---- STEP 12: validation -----------------------------------------------------
assert df.duplicated().sum() == 0
assert df.isna().sum().sum() == 0
assert (df.weight_kg > 0).all() and (df.quantity >= 1).all() and (df.actual_days <= 30).all()
df.to_csv(f"{OUT}/shipments_clean.csv", index=False)
enc_scaled.to_csv(f"{OUT}/shipments_model_ready.csv", index=False)
R["final_shape_clean"] = list(df.shape)
R["final_shape_model_ready"] = list(enc_scaled.shape)

# =============================================================================
# PART C - DID CLEANING HELP?  (we know the truth because we simulated it)
# =============================================================================
tidx = true.set_index("order_id")
feat = ["distance_km", "weight_kg", "scheduled_days", "quantity", "product_price"]
test_ids = set(true.order_id.sample(frac=0.2, random_state=1))
test_true = true[true.order_id.isin(test_ids)]

# "lazy" way: just fix the text prices, drop rows with blanks, keep everything else
lazy = raw.copy()
lazy.columns = df.columns[:0].tolist() or [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in raw.columns]
lazy["product_price"] = pd.to_numeric(lazy.product_price.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
lazy = lazy.rename(columns={"weight_kg": "weight_kg", "distance_km": "distance_km"})
lazy = lazy.dropna(subset=feat + ["actual_days"])
lazy = lazy[~lazy.order_id.isin(test_ids)]
clean_tr = df[~df.order_id.isin(test_ids)]

m_lazy = LinearRegression().fit(lazy[feat], lazy.actual_days)
m_clean = LinearRegression().fit(clean_tr[feat], clean_tr.actual_days)
mae_lazy = mean_absolute_error(test_true.actual_days, m_lazy.predict(test_true[feat]))
mae_clean = mean_absolute_error(test_true.actual_days, m_clean.predict(test_true[feat]))

true_late = (true.actual_days > true.scheduled_days).mean() * 100
lz_late = (lazy.actual_days > lazy.scheduled_days).mean() * 100
R["impact"] = {
    "rows_used_lazy": int(len(lazy)), "rows_used_clean": int(len(clean_tr)),
    "mae_days_lazy": round(float(mae_lazy), 3), "mae_days_clean": round(float(mae_clean), 3),
    "avg_actual_days_true": round(float(true.actual_days.mean()), 2),
    "avg_actual_days_lazy": round(float(lazy.actual_days.mean()), 2),
    "avg_actual_days_clean": round(float(df.actual_days.mean()), 2),
    "late_pct_true": round(float(true_late), 1), "late_pct_lazy": round(float(lz_late), 1),
    "late_pct_clean": round(float(df.is_late.mean() * 100), 1),
    "avg_price_true": round(float(true.product_price.mean()), 2),
    "avg_price_lazy": round(float(lazy.product_price.mean()), 2),
    "avg_price_clean": round(float(df.product_price.mean()), 2),
}

# quality scorecard
R["scorecard"] = {
    "completeness_before_pct": round(float(100 - raw.isna().mean().mean() * 100), 1),
    "completeness_after_pct": 100.0,
    "duplicates_before": 120, "duplicates_after": 0,
}

with open(f"{OUT}/results2.json", "w") as fh:
    json.dump(R, fh, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())

# =============================================================================
# FIGURES
# =============================================================================
plt.rcParams.update({"font.size": 10})
cols = [c for c in before_missing.index if before_missing[c] > 0]
names = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in cols]
fig, ax = plt.subplots(figsize=(7, 3.6))
y = np.arange(len(cols)); h = 0.38
ax.barh(y - h / 2, before_missing[cols].values, h, color="#c0504d", label="before cleaning")
aft = [after_missing.get(n, 0.0) for n in names]
ax.barh(y + h / 2, aft, h, color="#1f4e79", label="after cleaning")
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("% of values missing"); ax.set_title("Missing values per column")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/f1_missing.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
b = [raw_price.dropna(), df.product_price]
axes[0].boxplot(b, tick_labels=["before", "after"]); axes[0].set_yscale("log"); axes[0].set_title("Product price (log scale)")
b = [pd.to_numeric(raw["Actual Days"]).dropna(), df.actual_days]
axes[1].boxplot(b, tick_labels=["before", "after"]); axes[1].set_title("Actual delivery days")
b = [pd.to_numeric(raw["Weight (kg)"]).dropna(), df.weight_kg]
axes[2].boxplot(b, tick_labels=["before", "after"]); axes[2].set_yscale("symlog"); axes[2].set_title("Weight kg (symlog scale)")
fig.tight_layout(); fig.savefig(f"{OUT}/f2_outliers.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
axes[0].hist(df.product_price, bins=40, color="#1f4e79"); axes[0].set_title("price: raw")
axes[1].hist(df.log_price, bins=40, color="#1f4e79"); axes[1].set_title("price: log1p")
axes[2].hist(MinMaxScaler().fit_transform(df[["distance_km"]]), bins=40, color="#2e8b57"); axes[2].set_title("distance: min-max")
axes[3].hist(StandardScaler().fit_transform(df[["distance_km"]]), bins=40, color="#e07b00"); axes[3].set_title("distance: standard")
for a in axes: a.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/f3_scaling.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.bar(["Lazy cleaning", "Full pipeline"], [mae_lazy, mae_clean], color=["#c0504d", "#1f4e79"])
for i, v in enumerate([mae_lazy, mae_clean]):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
ax.set_ylabel("avg error in days (lower is better)")
ax.set_title("Predicting delivery days")
fig.tight_layout(); fig.savefig(f"{OUT}/f4_impact.png", dpi=160); plt.close(fig)

print(json.dumps(R, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))