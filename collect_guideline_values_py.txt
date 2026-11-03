"""
Collect Tamil Nadu guideline values for the Madurai registration zone
(Section 3.2 and Table 4 of the revised manuscript).

Source pages: verified.realestate, which republishes TNREGINET guideline values
(effective July 2024) as zone -> sub-registrar office (SRO) -> village pages.
Cite TNREGINET in the paper and spot-check a few villages on tnreginet.gov.in.

Outputs (in --out):
  gv_classifications.csv  one row per village x land classification
                          sro, village, taluk, classification, land_use, surveys,
                          value_sqft, value_per_acre
  gv_village_summary.csv  one row per village: survey-weighted median value per acre
                          (column market_value_per_acre), by land use
  gv_real_sample.csv      up to --per-village rows per village, sampled in proportion
                          to survey counts; use as --real for fidelity_check.py

Usage:
  pip install requests beautifulsoup4 pandas
  python collect_guideline_values.py --out gv
  python collect_guideline_values.py --out gv --sro vada-madurai melur-east   # only some SROs
  python collect_guideline_values.py --out gv --browser   # if pages come back empty (needs playwright)

Then, for example:
  python fidelity_check.py --synthetic land_transactions.csv --real gv/gv_real_sample.csv \
      --columns market_value_per_acre --group village

The zone also contains Dindigul and Theni SROs. Filter on the taluk column to keep
Madurai district only. Re-running resumes from the cache in --out/cache.
"""
import argparse
import hashlib
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://verified.realestate/dashboard/utility/guideline-value-survey"
SQFT_PER_ACRE = 43560
SEED = 42


class Fetcher:
    def __init__(self, cache, delay, browser):
        self.cache, self.delay = cache, delay
        os.makedirs(cache, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (research data collection)"
        self.page = None
        if browser:
            from playwright.sync_api import sync_playwright
            self.pw = sync_playwright().start()
            self.page = self.pw.chromium.launch().new_page()

    def get(self, url):
        path = os.path.join(self.cache, hashlib.md5(url.encode()).hexdigest() + ".html")
        if os.path.exists(path):
            return open(path, encoding="utf-8").read()
        for attempt in range(4):
            try:
                if self.page:
                    self.page.goto(url, wait_until="networkidle", timeout=60000)
                    html = self.page.content()
                else:
                    r = self.session.get(url, timeout=30)
                    r.raise_for_status()
                    html = r.text
                break
            except Exception as e:
                print(f"  retry {attempt + 1} for {url}: {e}")
                time.sleep(2 ** (attempt + 1))
        else:
            return ""
        open(path, "w", encoding="utf-8").write(html)
        time.sleep(self.delay)
        return html


def child_links(html, parent):
    """Absolute links exactly one level below `parent`, in page order, without duplicates."""
    soup = BeautifulSoup(html, "html.parser")
    pat = re.compile(re.escape(parent) + r"/([a-z0-9-]+)/?$")
    seen = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://verified.realestate" + href
        m = pat.match(href.split("?")[0].split("#")[0])
        if m and m.group(1) != "select" and href not in seen:
            seen.append(href.rstrip("/"))
    return list(dict.fromkeys(seen))


def tables(html):
    """All HTML tables as lists of dicts keyed by lower-cased header text."""
    out = []
    for t in BeautifulSoup(html, "html.parser").find_all("table"):
        rows = t.find_all("tr")
        if not rows:
            continue
        head = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])]
        body = [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in rows[1:]]
        out.append([dict(zip(head, r)) for r in body if len(r) == len(head)])
    return out


def pick(row, *keys):
    for k, v in row.items():
        if any(key in k for key in keys):
            return v
    return None


def num(s):
    if s is None:
        return np.nan
    s = re.sub(r"[^\d.]", "", s.replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return np.nan


def land_use(cls):
    c = cls.lower()
    if c.startswith(("dry", "wet", "garden", "punja", "nanja", "manavari", "maanavari")):
        return "agricultural"
    if "residential" in c:
        return "residential"
    if "commercial" in c:
        return "commercial"
    if "industrial" in c:
        return "industrial"
    return "other"


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, cw[-1] / 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="madurai")
    ap.add_argument("--sro", nargs="*", help="SRO slugs to include (default: all in the zone)")
    ap.add_argument("--out", default="gv")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--per-village", type=int, default=200)
    ap.add_argument("--browser", action="store_true", help="render pages with Playwright")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = Fetcher(os.path.join(a.out, "cache"), a.delay, a.browser)

    zone_url = f"{BASE}/{a.zone}"
    sros = child_links(f.get(zone_url), zone_url)
    if a.sro:
        sros = [u for u in sros if u.rsplit("/", 1)[1] in a.sro]
    print(f"{len(sros)} SROs")
    if not sros:
        raise SystemExit("No SRO links found. Try --browser, or check that the site is reachable.")

    rows = []
    for i, sro_url in enumerate(sros, 1):
        sro = sro_url.rsplit("/", 1)[1]
        html = f.get(sro_url)
        taluk = {}
        for t in tables(html):
            for r in t:
                v, tk = pick(r, "village"), pick(r, "taluk")
                if v and tk:
                    taluk[v.strip().lower()] = tk
        villages = child_links(html, sro_url)
        print(f"[{i}/{len(sros)}] {sro}: {len(villages)} villages")
        for v_url in villages:
            vhtml = f.get(v_url)
            vslug = v_url.rsplit("/", 1)[1]
            name = vslug.replace("-", " ").title()
            for t in tables(vhtml):
                for r in t:
                    cls = pick(r, "classification", "category")
                    if not cls or cls.startswith("-"):
                        continue
                    lo, hi = num(pick(r, "min")), num(pick(r, "max"))
                    val = np.nanmean([lo, hi]) if not (np.isnan(lo) and np.isnan(hi)) else np.nan
                    rows.append({"sro": sro, "village": name, "village_slug": vslug,
                                 "taluk": taluk.get(name.lower(), ""), "classification": cls,
                                 "land_use": land_use(cls), "surveys": num(pick(r, "survey")),
                                 "value_sqft": val})

    df = pd.DataFrame(rows).dropna(subset=["value_sqft"])
    if df.empty:
        raise SystemExit("No values parsed. Try --browser; the page layout may have changed.")
    df["value_per_acre"] = df["value_sqft"] * SQFT_PER_ACRE
    df["surveys"] = df["surveys"].fillna(1).clip(lower=1)
    df.to_csv(f"{a.out}/gv_classifications.csv", index=False)

    summ = (df.groupby(["sro", "village", "taluk", "land_use"])
              .apply(lambda g: pd.Series({"market_value_per_acre": weighted_median(g.value_per_acre, g.surveys),
                                          "surveys": g.surveys.sum(), "classifications": len(g)}))
              .reset_index())
    summ.to_csv(f"{a.out}/gv_village_summary.csv", index=False)

    rng = np.random.default_rng(SEED)
    samp = []
    for (sro, village, use), g in df.groupby(["sro", "village", "land_use"]):
        n = int(min(a.per_village, g.surveys.sum()))
        idx = rng.choice(len(g), size=n, p=(g.surveys / g.surveys.sum()).values)
        samp.append(g.iloc[idx][["sro", "village", "taluk", "land_use", "classification", "value_per_acre"]])
    real = pd.concat(samp).rename(columns={"value_per_acre": "market_value_per_acre"})
    real.to_csv(f"{a.out}/gv_real_sample.csv", index=False)

    print(f"\n{df.village_slug.nunique()} villages, {len(df)} classification rows")
    print(f"Wrote {a.out}/gv_classifications.csv, gv_village_summary.csv, gv_real_sample.csv")
    print("Report in Section 3.2: number of villages, access date, and 'values effective July 2024'.")


if __name__ == "__main__":
    main()
