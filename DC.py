# Assumptions (unchanged from v1): critical columns = product and quantity; missing product is
# inferred from unit_price only when that price maps to exactly ONE product in the data;
# quantities <=0 or >100 are invalid and excluded from total_price; slash dates are DD/MM/YYYY,
# non-ISO dash dates are MM-DD-YYYY; missing city/email become "Unknown".
import numpy as np
import pandas as pd

INPUT_PATH, OUTPUT_PATH = "messy_example.csv", "cleaned_data.csv"
CITIES = {"cairo": "Cairo", "cario": "Cairo", "alexandria": "Alexandria", "alex": "Alexandria",
          "giza": "Giza", "mansoura": "Mansoura", "tanta": "Tanta", "aswan": "Aswan"}
STATUSES = {"delivered": "Delivered", "shipped": "Shipped", "pending": "Pending",
            "cancelled": "Cancelled", "canceled": "Cancelled"}
EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

def report(msg):                      # one place to print what the cleaning did
    print(f"[clean] {msg}")

raw = pd.read_csv(INPUT_PATH)
df = raw.copy()
report(f"loaded {len(df)} rows, {df.shape[1]} columns")

# 1. Drop mostly-empty columns, but say which ones
keep = df.columns[df.notna().sum() >= int(len(df) * 0.5) + 1]
if len(keep) < df.shape[1]:
    report(f"dropped mostly-empty columns: {sorted(set(df.columns) - set(keep))}")
df = df[keep]

# 2. Whitespace first, so later steps (and duplicate detection) compare like with like
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
df["customer_name"] = df["customer_name"].str.replace(r"\s+", " ", regex=True).str.title()

# 3. Duplicates: exact copies are safe to drop; same order_id with DIFFERENT content is a conflict
n = len(df)
df = df.drop_duplicates()
report(f"removed {n - len(df)} exact duplicate rows")
conflicts = df[df.duplicated("order_id", keep=False)]
if len(conflicts):
    report(f"WARNING: {conflicts['order_id'].nunique()} order_ids repeat with different content; keeping first")
df = df.drop_duplicates(subset="order_id", keep="first")

# 4. City: missing -> Unknown, but an unrecognised spelling is reported, not silently erased
city_key = df["city"].str.lower()
mapped = city_key.map(CITIES)
unrecognised = df.loc[city_key.notna() & mapped.isna(), "city"]
if len(unrecognised):
    report(f"WARNING: unrecognised cities kept as typed: {unrecognised.unique().tolist()}")
df["city"] = mapped.fillna(unrecognised.str.title()).fillna("Unknown")

# 5. Status: same idea
status_key = df["status"].str.lower()
status = status_key.map(STATUSES)
if (status_key.notna() & status.isna()).any():
    report(f"WARNING: unmapped statuses: {df.loc[status_key.notna() & status.isna(), 'status'].unique().tolist()}")
df["status"] = status

# 6. Dates
d = df["order_date"].astype("string")
iso   = pd.to_datetime(d.where(d.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)), format="%Y-%m-%d", errors="coerce")
slash = pd.to_datetime(d.where(d.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)), format="%d/%m/%Y", errors="coerce")
dash  = pd.to_datetime(d.where(d.str.match(r"^\d{1,2}-\d{1,2}-\d{4}$", na=False)), format="%m-%d-%Y", errors="coerce")
name  = pd.to_datetime(d.where(d.str.match(r"^\d{1,2} [A-Za-z]{3,9} \d{4}$", na=False)), format="%d %b %Y", errors="coerce")
parsed = iso.fillna(slash).fillna(dash).fillna(name)
bad_dates = d.notna() & parsed.isna()
if bad_dates.any():
    report(f"WARNING: {bad_dates.sum()} dates could not be parsed: {d[bad_dates].tolist()}")
parts = d.str.extract(r"^(\d{1,2})[/-](\d{1,2})-?\d{0,4}")
ambiguous = (slash.notna() | dash.notna()) & (parts[0].astype(float) <= 12) & (parts[1].astype(float) <= 12)
report(f"{ambiguous.sum()} slash/dash dates are ambiguous (day and month both <= 12); relying on the stated format assumption")
df["order_date"] = parsed.dt.strftime("%Y-%m-%d")

# 7. Prices (currency symbols/text removed; assumes every price is in the same currency)
df["unit_price"] = pd.to_numeric(
    df["unit_price"].astype("string").str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

# 8. Product inference from price, now flagged so it can be audited or undone
known = df.dropna(subset=["product", "unit_price"]).groupby("unit_price")["product"].agg(lambda s: s.unique())
price_map = {p: v[0] for p, v in known.items() if len(v) == 1}
missing_product = df["product"].isna()
df["product"] = df["product"].fillna(df["unit_price"].map(price_map))
df["product_imputed"] = missing_product & df["product"].notna()
report(f"inferred product from price for {df['product_imputed'].sum()} rows")

# 9. Quantity: say WHY it is invalid (missing vs out of range)
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
n = len(df)
df = df.dropna(subset=["product", "quantity"], how="all").copy()
report(f"dropped {n - len(df)} rows missing both product and quantity")
df["quantity_valid"] = df["quantity"].between(1, 100)
df["quantity_issue"] = np.select(
    [df["quantity"].isna(), ~df["quantity_valid"]], ["missing", "out_of_range"], default="")

# 10. Email: separate "missing" from "malformed" instead of one boolean
df["email"] = df["email"].str.lower()
df["email_status"] = np.select(
    [df["email"].isna(), ~df["email"].str.match(EMAIL_RE, na=False)], ["missing", "malformed"], default="ok")
df["email_valid"] = df["email_status"].eq("ok")
df["email"] = df["email"].fillna("Unknown")

df["total_price"] = np.where(df["quantity_valid"], df["quantity"] * df["unit_price"], np.nan)
df["quantity"] = df["quantity"].astype("Int64")

df.to_csv(OUTPUT_PATH, index=False)
report(f"wrote {len(df)} rows to {OUTPUT_PATH}")
