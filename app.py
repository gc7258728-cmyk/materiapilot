# -*- coding: utf-8 -*-
"""MatériaPilot — registre de matériaux et prévisions pédagogiques.

Tout le code de l'application est dans ce fichier. Les données persistantes
sont écrites hors du dossier livré. Aucun prix ni modèle fourni n'est une norme.
"""
from __future__ import annotations

import copy
import hashlib
import html
import io
import json
import math
import os
import sqlite3
import uuid
import zipfile
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = "MatériaPilot"
VERSION = "1.0"
EPS = 1e-8
UNITS = ["sac", "m³", "kg", "pièce", "m", "t", "L"]
DISCRETE = {"sac", "pièce"}
KINDS = ["Stock initial", "Consommation", "Retour d’utilisation", "Perte / casse", "Transfert", "Inventaire", "Réception", "Annulation"]
COLORS = ["#2457B8", "#C79513", "#008575", "#8B5FC8", "#D15B48"]
IMPORT_COLS = ["date", "type", "materiau_code", "source_zone", "destination_zone", "quantite", "compte_physique", "document", "acteur", "note"]


def ident(prefix):
    return prefix + uuid.uuid4().hex[:12]


def today():
    return date.today()


def iso(value):
    return date.fromisoformat(str(value)).isoformat()


def number(value, label="Quantité", minimum=None):
    if isinstance(value, bool):
        raise ValueError(f"{label} : nombre attendu.")
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} : nombre attendu.") from None
    if not math.isfinite(n) or abs(n) > 1e12:
        raise ValueError(f"{label} : nombre fini, au plus 10¹² en valeur absolue.")
    if minimum is not None and n < minimum:
        raise ValueError(f"{label} doit être ≥ {minimum}.")
    return n


def text(value, label="Texte", required=False, limit=500):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{label} : texte limité à {limit} caractères.")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{label} est obligatoire.")
    return value


def lookup(p, group, key):
    result = next((x for x in p[group] if x["id"] == key), None)
    if result is None:
        raise ValueError(f"Référence inconnue : {group} / {key}.")
    return result


def fmt(value, decimals=2):
    return f"{float(value):,.{decimals}f}".replace(",", " ").replace(".", ",")


def money(value):
    return fmt(value, 0) + " FCFA"


def blank_project(demo=False):
    d = (today() - timedelta(days=30)).isoformat()
    materials = [
        {"id": "CIM", "code": "CIM", "name": "Ciment", "unit": "sac", "price": None, "price_source": ""},
        {"id": "SAB", "code": "SAB", "name": "Sable", "unit": "m³", "price": None, "price_source": ""},
        {"id": "GRA", "code": "GRA", "name": "Gravier", "unit": "m³", "price": None, "price_source": ""},
        {"id": "BOI", "code": "BOI", "name": "Bois", "unit": "m³", "price": None, "price_source": ""},
        {"id": "ACI", "code": "ACI", "name": "Acier", "unit": "kg", "price": None, "price_source": ""},
    ]
    return {"app": APP, "version": 1, "demo": demo, "start": d,
            "sites": [{"id": "S1", "name": "Mon chantier"}],
            "zones": [{"id": "Z1", "code": "DEP", "name": "Dépôt principal", "site_id": "S1"}],
            "materials": materials, "suppliers": [], "orders": [], "events": [], "plans": {}}


def movement(kind, mat, qty, *, src=None, dst=None, day=None, doc="", **extra):
    return dict(id=ident("M"), date=day or today().isoformat(), type=kind,
                material_id=mat, qty=float(qty), src=src, dst=dst,
                doc=doc, actor="", note="", **extra)


def legs(event, events):
    kind = event["type"]
    qty = event["qty"]
    if kind == "Annulation":
        original = next(x for x in events if x["id"] == event["reverse_of"])
        return [(z, m, -q) for z, m, q in legs(original, events)]
    mat = event["material_id"]
    if kind == "Transfert":
        return [(event["src"], mat, -qty), (event["dst"], mat, qty)]
    if kind in ("Consommation", "Perte / casse"):
        return [(event["src"], mat, -qty)]
    return [(event["dst"], mat, qty)]


def balances(p, asof=None):
    result = {(z["id"], m["id"]): 0.0 for z in p["zones"] for m in p["materials"]}
    for e in p["events"]:
        if asof is not None and e["date"] > str(asof):
            continue
        for z, m, q in legs(e, p["events"]):
            result[z, m] += q
    return {k: max(0.0, v) if abs(v) < EPS else v for k, v in result.items()}


def effective_events(p):
    canceled = {e["reverse_of"] for e in p["events"] if e["type"] == "Annulation"}
    return [e for e in p["events"] if e["type"] != "Annulation" and e["id"] not in canceled]


def received(p, order_id):
    rows = [e for e in effective_events(p) if e["type"] == "Réception" and e["order_id"] == order_id]
    return sum(e["qty"] for e in rows), sum(e.get("rejected", 0) for e in rows)


def order_remaining(p, o):
    return max(0.0, o["qty"] - received(p, o["id"])[0])


def order_status(p, o):
    accepted, _ = received(p, o["id"])
    if o["closed"]:
        return "Clôturée"
    if order_remaining(p, o) < EPS:
        return "Livrée"
    state = "Partielle" if accepted > EPS else "Commandée"
    return state + (" · en retard" if o["eta"] < today().isoformat() else "")


def validate(p):
    if not isinstance(p, dict) or p.get("app") != APP or p.get("version") != 1:
        raise ValueError("Sauvegarde MatériaPilot version 1 attendue.")
    start = iso(p["start"])
    if start > today().isoformat():
        raise ValueError("La date de début doit être passée ou aujourd’hui.")
    if type(p.get("demo")) is not bool:
        raise ValueError("Mode du registre invalide.")
    limits = {"sites": 10, "zones": 40, "materials": 40, "suppliers": 50, "orders": 500, "events": 3000}
    for group, limit in limits.items():
        if not isinstance(p.get(group), list) or len(p[group]) > limit:
            raise ValueError(f"{group} : maximum {limit} éléments.")
        if not all(isinstance(x, dict) for x in p[group]):
            raise ValueError(f"{group} : chaque élément doit être une fiche structurée.")
        ids = [text(x.get("id"), "Identifiant", True, 80) for x in p[group]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Identifiants dupliqués dans {group}.")
    if not p["sites"] or not p["zones"] or not p["materials"]:
        raise ValueError("Il faut au moins un chantier, une zone et un matériau.")
    for group in ("sites", "zones", "materials", "suppliers"):
        for x in p[group]:
            text(x["name"], "Nom", True, 120)
    for group in ("zones", "materials"):
        codes = [text(x["code"], "Code", True, 24) for x in p[group]]
        if len(codes) != len(set(codes)):
            raise ValueError(f"Codes dupliqués : {group}.")
    for z in p["zones"]:
        lookup(p, "sites", z["site_id"])
    for m in p["materials"]:
        if m["unit"] not in UNITS:
            raise ValueError("Unité non reconnue.")
        text(m["price_source"], "Source du prix", limit=200)
        if m["price"] is not None:
            number(m["price"], "Prix unitaire", 0)
            text(m["price_source"], "Source du prix", True, 200)
    codes = []
    for o in p["orders"]:
        codes.append(text(o["code"], "Numéro de commande", True, 80))
        m = lookup(p, "materials", o["material_id"])
        lookup(p, "zones", o["zone_id"])
        lookup(p, "suppliers", o["supplier_id"])
        if not start <= iso(o["date"]) <= today().isoformat() or iso(o["eta"]) < o["date"]:
            raise ValueError("Dates de commande / livraison prévues incohérentes.")
        q = number(o["qty"], "Quantité commandée", EPS)
        if m["unit"] in DISCRETE and not q.is_integer():
            raise ValueError("Les sacs et pièces sont des quantités entières.")
        if o["price"] is not None:
            number(o["price"], "Prix commandé", 0)
        delay = number(o["delay"], "Retard maximal", 0)
        if not delay.is_integer() or delay > 90 or type(o["closed"]) is not bool:
            raise ValueError("Retard entier entre 0 et 90 jours / clôture invalide.")
        text(o["note"], "Note de commande")
    if len(codes) != len(set(codes)):
        raise ValueError("Numéro de commande déjà utilisé.")
    stock = {(z["id"], m["id"]): 0.0 for z in p["zones"] for m in p["materials"]}
    seen, reversed_ids, document_keys, last_date, accepted = {}, set(), set(), start, {}
    for e in p["events"]:
        event_date = iso(e["date"])
        if not last_date <= event_date <= today().isoformat():
            raise ValueError("Le journal suit l’ordre chronologique, sans date future. Utilise la date de prise en compte et mentionne une ancienne date dans la note.")
        last_date = event_date
        if e["type"] not in KINDS:
            raise ValueError("Type de mouvement inconnu.")
        m = lookup(p, "materials", e["material_id"])
        qty = number(e["qty"])
        for key in ("actor", "note"):
            text(e.get(key, ""), key)
        text(e["doc"], "Bon / référence", True, 80)
        if e["type"] == "Annulation":
            orig = seen.get(e.get("reverse_of"))
            if orig is None or orig["type"] == "Annulation" or orig["id"] in reversed_ids:
                raise ValueError("Ce mouvement ne peut pas être annulé une nouvelle fois.")
            if e["material_id"] != orig["material_id"] or abs(qty - orig["qty"]) > EPS:
                raise ValueError("Contre-écriture différente du mouvement original.")
            text(e["note"], "Motif d’annulation", True)
            reversed_ids.add(orig["id"])
            if orig["type"] == "Réception":
                accepted[orig["order_id"]] -= orig["qty"]
        else:
            if e["type"] != "Inventaire" and qty < 0:
                raise ValueError("Quantité négative interdite.")
            if e["type"] not in ("Inventaire", "Réception") and qty <= 0:
                raise ValueError("La quantité doit être strictement positive.")
            if m["unit"] in DISCRETE and not qty.is_integer():
                raise ValueError("Quantités entières pour les sacs et pièces.")
            if e["type"] == "Transfert" and e["src"] == e["dst"]:
                raise ValueError("Choisis deux zones différentes.")
            if e["type"] == "Perte / casse":
                text(e["note"], "Motif de la perte", True)
            if e["type"] == "Inventaire":
                counted = number(e["counted"], "Comptage physique", 0)
                if abs(stock.get((e["dst"], m["id"]), 0) + qty - counted) > EPS:
                    raise ValueError("Écart d’inventaire incohérent avec le stock avant comptage.")
            if e["type"] == "Réception":
                o = lookup(p, "orders", e["order_id"])
                rejected = number(e["rejected"], "Quantité refusée", 0)
                if m["unit"] in DISCRETE and not rejected.is_integer():
                    raise ValueError("Quantité refusée entière attendue.")
                if qty + rejected <= 0 or e["dst"] != o["zone_id"] or m["id"] != o["material_id"] or e["date"] < o["date"]:
                    raise ValueError("Réception incompatible avec la commande.")
                accepted[o["id"]] = accepted.get(o["id"], 0) + qty
                if accepted[o["id"]] > o["qty"] + EPS:
                    raise ValueError("Le cumul accepté dépasse la quantité commandée.")
            doc_key = (e["doc"].casefold(), e["type"], m["id"], e.get("src"), e.get("dst"))
            if doc_key in document_keys:
                raise ValueError("Bon déjà enregistré pour ce mouvement et ce matériau. Pour une correction, utilise une nouvelle référence.")
            document_keys.add(doc_key)
        for z, mat, delta in legs(e, p["events"]):
            lookup(p, "zones", z)
            stock[z, mat] += delta
            if stock[z, mat] < -EPS:
                raise ValueError(f"Stock insuffisant dans {lookup(p, 'zones', z)['code']} pour {m['name']}. L’opération n’est pas enregistrée.")
        seen[e["id"]] = e
    plans = p.get("plans")
    if not isinstance(plans, dict):
        raise ValueError("Profils de consommation invalides.")
    for site, profiles in plans.items():
        lookup(p, "sites", site)
        if not isinstance(profiles, dict):
            raise ValueError("Profils du chantier : objet structuré attendu.")
        for mat, v in profiles.items():
            if not isinstance(v, dict):
                raise ValueError("Profil du matériau : fiche structurée attendue.")
            m = lookup(p, "materials", mat)
            if type(v["enabled"]) is not bool or v["model"] not in ("Constante", "Poisson", "Triangulaire"):
                raise ValueError("Modèle de consommation non reconnu.")
            mu = number(v["mean"], "Consommation moyenne", 0)
            spread = number(v["spread"], "Variabilité", 0)
            number(v["threshold"], "Seuil d’alerte", 0)
            if spread > 100 or (v["model"] == "Poisson" and m["unit"] not in DISCRETE) or (v["model"] == "Triangulaire" and m["unit"] in DISCRETE):
                raise ValueError("Modèle incompatible avec l’unité ou variabilité hors de 0–100 %.")
            if m["unit"] in DISCRETE and v["model"] == "Constante" and not mu.is_integer():
                raise ValueError("La consommation constante de sacs / pièces doit être entière.")
            text(v["source"], "Origine de l’hypothèse", bool(v["enabled"]), 200)
    return p


def append_event(p, e):
    result = copy.deepcopy(p)
    result["events"].append(e)
    return validate(result)


def inventory_event(p, mat, zone, counted, **kwargs):
    delta = float(counted) - balances(p)[zone, mat]
    return movement("Inventaire", mat, delta, dst=zone, counted=float(counted), **kwargs)


def demo_project():
    p = blank_project(True)
    p["sites"] = [{"id": "S1", "name": "Résidence Horizon · fictif"}, {"id": "S2", "name": "Atelier École · fictif"}]
    p["zones"] += [{"id": "Z2", "code": "BLOC-A", "name": "Zone de mise en œuvre", "site_id": "S1"}, {"id": "Z3", "code": "ECO-DEP", "name": "Dépôt école", "site_id": "S2"}]
    p["suppliers"] = [{"id": "F1", "name": "Fournisseur Alpha · fictif"}, {"id": "F2", "name": "Carrière Démo · fictif"}]
    for m, price in zip(p["materials"], [4800, 16000, 18000, 60000, 700]):
        m.update(price=price, price_source="Prix fictif de démonstration, sans référence de marché")
    day = lambda offset: (today() + timedelta(days=offset)).isoformat()
    for code, mat, qty, zone, eta, price, delay in [
        ("BC-001", "CIM", 50, "Z1", -4, 4800, 2), ("BC-002", "SAB", 10, "Z1", 3, 15500, 2),
        ("BC-003", "ACI", 800, "Z1", 5, 650, 3), ("BC-004", "CIM", 80, "Z1", 4, 4900, 1),
        ("BC-005", "CIM", 25, "Z3", 2, 5000, 0)]:
        p["orders"].append(dict(id=code, code=code, material_id=mat, zone_id=zone, supplier_id="F2" if mat == "SAB" else "F1", date=day(-10), eta=day(eta), qty=float(qty), price=price, delay=delay, closed=False, note="Commande fictive"))
    for mat, qty in zip(["CIM", "SAB", "GRA", "BOI", "ACI"], [100, 20, 30, 5, 1000]):
        p["events"].append(movement("Stock initial", mat, qty, dst="Z1", day=day(-12), doc="INIT-" + mat))
    p["events"].append(movement("Stock initial", "CIM", 30, dst="Z3", day=day(-12), doc="INIT-ECO"))
    p["events"].append(movement("Transfert", "CIM", 25, src="Z1", dst="Z2", day=day(-9), doc="TR-001"))
    for mat, qty, zone in [("CIM", 10, "Z2"), ("SAB", 3, "Z1"), ("GRA", 4, "Z1"), ("ACI", 100, "Z1")]:
        p["events"].append(movement("Consommation", mat, qty, src=zone, day=day(-8), doc="CONS-" + mat))
    loss = movement("Perte / casse", "CIM", 2, src="Z2", day=day(-7), doc="PERTE-001")
    loss["note"] = "Deux sacs détériorés — exemple fictif"
    p["events"].append(loss)
    p["events"].append(movement("Réception", "CIM", 20, dst="Z1", day=day(-6), doc="BL-001", order_id="BC-001", rejected=3.0))
    p["events"].append(movement("Retour d’utilisation", "CIM", 3, dst="Z2", day=day(-5), doc="RET-001"))
    p["events"].append(inventory_event(p, "CIM", "Z1", 94, day=day(-1), doc="INV-001"))
    p["plans"]["S1"] = {}
    for m, mu in zip(p["materials"], [12, 2, 3, .3, 80]):
        p["plans"]["S1"][m["id"]] = dict(enabled=True, mean=mu, model="Poisson" if m["unit"] in DISCRETE else "Triangulaire", spread=25, threshold=mu * 3, source="Hypothèse fictive ; consommation par jour d’activité")
    return validate(p)


def data_path():
    root = Path(os.environ.get("MATERIAPILOT_DATA_DIR", str(Path.home() / "Documents/Codex/data/materiapilot"))).expanduser().resolve()
    if root == Path(__file__).resolve().parent or Path(__file__).resolve().parent in root.parents:
        raise ValueError("Le répertoire de données doit être hors du dossier de l’application.")
    root.mkdir(parents=True, exist_ok=True)
    return root / "registre.sqlite3"


def load_real():
    with closing(sqlite3.connect(data_path(), timeout=10)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS workspace (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, payload TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO workspace VALUES (1, 0, ?)", (json.dumps(blank_project(), ensure_ascii=False),))
        db.commit()
        revision, payload = db.execute("SELECT revision,payload FROM workspace WHERE id=1").fetchone()
    return validate(json.loads(payload)), revision


def save_real(p, expected_revision):
    validate(p)
    if p["demo"]:
        raise ValueError("Les exemples fictifs ne sont pas écrits dans le registre réel.")
    with closing(sqlite3.connect(data_path(), timeout=10)) as db:
        cursor = db.execute("UPDATE workspace SET revision=revision+1,payload=? WHERE id=1 AND revision=?", (json.dumps(p, ensure_ascii=False), expected_revision))
        if cursor.rowcount != 1:
            db.rollback()
            raise ValueError("Le registre a changé dans une autre session. Clique sur Actualiser le registre, puis recommence ton opération.")
        db.commit()
    return expected_revision + 1


def plan_for(p, site, mat):
    return copy.deepcopy(p["plans"].get(site, {}).get(mat, dict(enabled=False, mean=0.0, model="Constante", spread=25, threshold=0.0, source="")))


def site_zones(p, site):
    return [z for z in p["zones"] if z["site_id"] == site]


def site_stock(p, site):
    b = balances(p)
    return {m["id"]: sum(b[z["id"], m["id"]] for z in site_zones(p, site)) for m in p["materials"]}


def event_site(p, e, site):
    ids = {z["id"] for z in site_zones(p, site)}
    return any(z in ids for z, _, _ in legs(e, p["events"]))


def inventory_table(p, site=None):
    b = balances(p)
    rows = []
    for z in p["zones"]:
        if site and z["site_id"] != site:
            continue
        for m in p["materials"]:
            q = b[z["id"], m["id"]]
            rows.append({"Chantier": lookup(p, "sites", z["site_id"])["name"], "Zone": z["code"], "Matériau": m["name"], "Code": m["code"], "Stock": q, "Unité": m["unit"], "Prix de référence FCFA/unité": m["price"], "Valeur de référence FCFA": None if m["price"] is None else q * m["price"]})
    return pd.DataFrame(rows)


def journal_table(p, site=None):
    rows = []
    canceled = {e["reverse_of"] for e in p["events"] if e["type"] == "Annulation"}
    for e in reversed(p["events"]):
        if site and not event_site(p, e, site):
            continue
        m = lookup(p, "materials", e["material_id"])
        label = lambda key: lookup(p, "zones", e[key])["code"] if e.get(key) else "—"
        rows.append({"ID": e["id"], "Date": e["date"], "Type": e["type"], "Matériau": m["name"], "Quantité / écart": e["qty"], "Unité": m["unit"], "Source": label("src"), "Destination": label("dst"), "Référence": e["doc"], "État": "Contre-écriture" if e["type"] == "Annulation" else "Annulé" if e["id"] in canceled else "Actif", "Acteur": e.get("actor", ""), "Note": e.get("note", "")})
    return pd.DataFrame(rows, columns=["ID", "Date", "Type", "Matériau", "Quantité / écart", "Unité", "Source", "Destination", "Référence", "État", "Acteur", "Note"])


def orders_table(p, site=None):
    rows = []
    for o in p["orders"]:
        z = lookup(p, "zones", o["zone_id"])
        if site and z["site_id"] != site:
            continue
        m = lookup(p, "materials", o["material_id"])
        a, r = received(p, o["id"])
        rows.append({"Commande": o["code"], "Fournisseur": lookup(p, "suppliers", o["supplier_id"])["name"], "Matériau": m["name"], "Zone": z["code"], "Commandé": o["qty"], "Accepté": a, "Refusé": r, "Reste dû": order_remaining(p, o), "Unité": m["unit"], "Prévu le": o["eta"], "Prix FCFA/unité": o["price"], "Valeur acceptée FCFA": None if o["price"] is None else a * o["price"], "État": order_status(p, o)})
    return pd.DataFrame(rows, columns=["Commande", "Fournisseur", "Matériau", "Zone", "Commandé", "Accepté", "Refusé", "Reste dû", "Unité", "Prévu le", "Prix FCFA/unité", "Valeur acceptée FCFA", "État"])


def fingerprint(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def wilson(k, n):
    ph = k / n
    z = 1.95996398454
    denom = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def risk_label(first, q, horizon):
    value = int(np.quantile(first, q, method="inverted_cdf"))
    return f"> {horizon} j" if value > horizon else f"J{value}"


@st.cache_data(ttl=1800, max_entries=6, show_spinner=False)
def simulate(payload, site, settings_json):
    p = validate(json.loads(payload))
    cfg = json.loads(settings_json)
    n, h = int(cfg["n"]), int(cfg["h"])
    if n not in (500, 2000, 5000, 10000, 20000) or not 7 <= h <= 90:
        raise ValueError("Dimensions de simulation non autorisées.")
    interruption = number(cfg["interruption"], "Interruption", 0)
    if interruption > 100 or not cfg["weekdays"] or any(x not in range(7) for x in cfg["weekdays"]):
        raise ValueError("Calendrier ou probabilité d’interruption invalide.")
    rng = np.random.default_rng(int(cfg["seed"]))
    asof = date.fromisoformat(cfg["asof"])
    dates = [asof + timedelta(days=j) for j in range(1, h + 1)]
    active = np.array([x.weekday() in cfg["weekdays"] for x in dates])
    work = (rng.random((n, h)) >= interruption / 100) & active[None, :]
    stocks = site_stock(p, site)
    zones = {z["id"] for z in site_zones(p, site)}
    profiles = [(m, plan_for(p, site, m["id"])) for m in p["materials"] if plan_for(p, site, m["id"])["enabled"]]
    if not 1 <= len(profiles) <= 10:
        raise ValueError("Active entre 1 et 10 profils de consommation.")
    out, global_first, ignored = [], np.full(n, h + 1, dtype=int), []
    for m, plan in profiles:
        mu = plan["mean"]
        if plan["model"] == "Poisson":
            demand = rng.poisson(mu, size=(n, h)).astype(float)
        elif plan["model"] == "Triangulaire" and mu * plan["spread"] > 0:
            span = mu * plan["spread"] / 100
            demand = rng.triangular(mu - span, mu, mu + span, size=(n, h))
        else:
            demand = np.full((n, h), mu, dtype=float)
        demand *= work
        receipts = np.zeros((n, h), dtype=float)
        for o in p["orders"]:
            remaining = order_remaining(p, o)
            if o["zone_id"] not in zones or o["material_id"] != m["id"] or o["closed"] or remaining < EPS:
                continue
            base_day = (date.fromisoformat(o["eta"]) - asof).days
            if base_day <= 0:
                if not cfg["overdue"]:
                    ignored.append(o["code"])
                    continue
                base_day = 1
            arrival = base_day + rng.integers(0, int(o["delay"]) + 1, size=n)
            eligible = np.where((arrival >= 1) & (arrival <= h))[0]
            receipts[eligible, arrival[eligible] - 1] += remaining
        # Le levier est déterministe : mêmes tirages aléatoires dans les deux scénarios.
        extra = cfg.get("extra", {})
        if extra.get("material_id") == m["id"]:
            extra_qty = number(extra["qty"], "Apport hypothétique", 0)
            extra_day = int(extra["day"])
            if not 1 <= extra_day <= h or (m["unit"] in DISCRETE and not extra_qty.is_integer()):
                raise ValueError("Apport hypothétique invalide.")
            receipts[:, extra_day - 1] += extra_qty
        paths = np.zeros((n, h + 1), dtype=float)
        paths[:, 0] = stocks[m["id"]]
        first = np.full(n, h + 1, dtype=int)
        requested, served, missing = [], [], []
        for j in range(h):
            available = paths[:, j] + (receipts[:, j] if cfg["timing"] == "Début de journée" else 0)
            unmet = np.maximum(demand[:, j] - available, 0)
            first[(first == h + 1) & (unmet > EPS)] = j + 1
            used = demand[:, j] - unmet
            paths[:, j + 1] = available - used + (receipts[:, j] if cfg["timing"] == "Fin de journée" else 0)
            requested.append(float(demand[:, j].mean()))
            served.append(float(used.mean()))
            missing.append(float(unmet.mean()))
        global_first = np.minimum(global_first, first)
        out.append(dict(id=m["id"], name=m["name"], unit=m["unit"], start=stocks[m["id"]], first=first,
                        bands=np.quantile(paths, [.1, .5, .9], axis=0, method="inverted_cdf"), requested=requested, served=served, missing=missing, model=plan["model"]))
    return dict(materials=out, global_first=global_first, ignored=ignored, dates=[d.isoformat() for d in dates], n=n, h=h, cfg=cfg)


def forecast_tables(result):
    summary, daily = [], []
    for r in result["materials"]:
        k = int(np.count_nonzero(r["first"] <= result["h"]))
        lo, hi = wilson(k, result["n"])
        summary.append({"Matériau": r["name"], "Unité": r["unit"], "Stock initial": r["start"], "P rupture": k / result["n"], "IC MC 95% bas": lo, "IC MC 95% haut": hi, "Premier risque Q10": risk_label(r["first"], .1, result["h"]), "Premier risque médiane": risk_label(r["first"], .5, result["h"]), "Premier risque Q90": risk_label(r["first"], .9, result["h"])})
        for j, d in enumerate(result["dates"]):
            daily.append({"Date": d, "Jour": j + 1, "Matériau": r["name"], "Unité": r["unit"], "Stock Q10": r["bands"][0, j + 1], "Stock médian": r["bands"][1, j + 1], "Stock Q90": r["bands"][2, j + 1], "Demande moyenne": r["requested"][j], "Servi moyen": r["served"][j], "Non servi moyen": r["missing"][j]})
    return pd.DataFrame(summary), pd.DataFrame(daily)


def safe_csv(df):
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(("=", "+", "-", "@")) else x)
    return clean.to_csv(index=False).encode("utf-8-sig")


def import_movements(p, frame):
    if set(frame.columns) != set(IMPORT_COLS) or not 1 <= len(frame) <= 300:
        raise ValueError("Le modèle exact de colonnes est requis ; de 1 à 300 lignes par import.")
    trial = copy.deepcopy(p)
    materials = {m["code"]: m["id"] for m in p["materials"]}
    zones = {z["code"]: z["id"] for z in p["zones"]}
    for index, row in frame.fillna("").iterrows():
        try:
            kind = str(row["type"]).strip()
            if kind not in KINDS[:6]:
                raise ValueError("Réceptions et annulations se font dans leur outil dédié.")
            mat = materials[str(row["materiau_code"]).strip()]
            src = zones.get(str(row["source_zone"]).strip())
            dst = zones.get(str(row["destination_zone"]).strip())
            kwargs = dict(day=iso(str(row["date"])[:10]), doc=str(row["document"]).strip())
            if kind == "Inventaire":
                e = inventory_event(trial, mat, dst, number(row["compte_physique"], minimum=0), **kwargs)
            else:
                e = movement(kind, mat, number(row["quantite"], minimum=EPS), src=src, dst=dst, **kwargs)
            e.update(actor=str(row["acteur"]), note=str(row["note"]))
            trial = append_event(trial, e)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Ligne {index + 2} : {exc}. Aucun mouvement de ce fichier n’a été ajouté.") from None
    return trial


def excel_bytes(p, site=None, forecast=None):
    buf = io.BytesIO()
    sheets = {"Stocks": inventory_table(p, site), "Journal": journal_table(p, site), "Commandes": orders_table(p, site), "Catalogue": pd.DataFrame(p["materials"]), "Fournisseurs": pd.DataFrame(p["suppliers"])}
    if forecast:
        sheets["Prévisions"], sheets["Courbes simulées"] = forecast_tables(forecast)
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=name)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for row in ws:
                for cell in row:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
            from openpyxl.styles import Font, PatternFill
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="14283F")
                cell.font = Font(color="FFFFFF", bold=True)
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(45, max(14, max(len(str(c.value or "")) for c in col[:100]) + 2))
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Interface : commandes Streamlit natives, habillage embarqué et responsive.
# -----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--navy:#102239;--ink:#172D46;--muted:#52677E;--blue:#2457B8;--line:#D8E1ED;--signal:#F4D35E}
.stApp{background:#EDF2F8;color:var(--ink);font-family:'DM Sans',sans-serif}
[data-testid="stMainBlockContainer"]{max-width:1580px;padding-top:4rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:var(--ink);letter-spacing:-.035em}
h1{font-size:2.3rem!important}h2{font-size:1.6rem!important}h3{font-size:1.15rem!important}
[data-testid="stSidebar"]{background:#102239!important}
[data-testid="stSidebar"] *{color:#F0F5FF}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] [data-baseweb="select"] div{color:#F0F5FF}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#203B5B;border-color:#496480}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:#284B77;border-left:3px solid #F4D35E}
[data-testid="stSidebar"] button{background:#203B5B;border:1px solid #496480}
[data-testid="stSidebar"] [role="alert"]{background:#203B5B}
[data-testid="stForm"],[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:16px}
[data-testid="stForm"]{background:#F9FBFE;border:1px solid var(--line)}
.stButton>button,.stDownloadButton>button{min-height:2.7rem;border-radius:10px;font-weight:600}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#2457B8;color:#FFF;border:0}
.mast{display:flex;justify-content:space-between;gap:20px;align-items:center;border-bottom:1px solid var(--line);padding:0 0 18px;margin-bottom:22px}
.brand{display:flex;gap:12px;align-items:center}.brand-mark{background:#102239;color:#F4D35E;border-radius:12px;width:44px;height:44px;display:grid;place-items:center;font-size:24px;font-weight:700}
.brand-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:22px;letter-spacing:-1px}.eyebrow{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#52677E}
.status-pill{border:1px solid #C5D2E3;border-radius:30px;padding:7px 12px;font-size:11px;font-weight:600;white-space:nowrap;background:#FAFCFF}
.command{position:relative;overflow:hidden;background:#102239;border-radius:20px;color:#F4F7FF;padding:27px 30px;margin:12px 0 22px;display:grid;grid-template-columns:1.3fr 1fr;gap:28px;background-image:linear-gradient(125deg,#142C48,#0C192A)}
.command h2{color:#FFF;font-size:1.9rem!important;margin:8px 0!important;line-height:1.15}.command p{color:#C8D6E7;font-size:13px;line-height:1.65;margin:0;max-width:520px}
.command .eyebrow{color:#F4D35E}.command-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-content:center}
.signalbox{border:1px solid #40546D;border-radius:12px;padding:14px;background:#FFFFFF06}.signalbox strong{display:block;font-size:26px;font-family:'Space Grotesk',sans-serif;color:#F4D35E}.signalbox span{font-size:11px;color:#D8E4F4}
.kpi{height:100%;min-height:110px;border:1px solid var(--line);background:#FCFDFF;border-radius:15px;padding:17px 19px;border-top:4px solid var(--accent,#2457B8);margin-bottom:8px}
.kpi-label{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#52677E;font-weight:700}.kpi-value{font-size:27px;line-height:1.35;font-family:'Space Grotesk',sans-serif;font-weight:700;color:#102239;overflow-wrap:anywhere}.kpi-note{font-size:11px;color:#52677E;margin-top:4px}
.material-tile{border:1px solid #D1DCEA;background:#FFF;border-radius:16px;padding:19px;min-height:170px;border-left:5px solid var(--accent);margin:4px 0 12px}.material-top{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:#52677E}.material-name{font-weight:700;color:#19314F;font-size:16px;margin-top:8px}.material-value{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:600;color:#102239;line-height:1.45}.material-value small{font-size:12px;font-family:'DM Sans',sans-serif;font-weight:400}.stockbar{height:5px;background:#E9EEF6;border-radius:4px;overflow:hidden;margin:10px 0}.stockbar i{display:block;height:100%;background:var(--accent)}
.material-name,.command h2,.brand-name{overflow-wrap:anywhere}
.note{border-left:3px solid #C4991E;background:#FBF3D8;border-radius:0 10px 10px 0;padding:12px 15px;color:#5A4919;font-size:12px;line-height:1.6;margin:10px 0 20px}
.guidebox{border:1px solid #C7D8F2;background:#E4ECF9;border-radius:13px;padding:15px 18px;font-size:13px;line-height:1.7;color:#203D64;margin:12px 0}
.section-label{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#52677E;margin:18px 0 4px;font-weight:700}
[data-testid="stMetricValue"]{color:#102239;font-family:'Space Grotesk',sans-serif}
[data-testid="stCaptionContainer"]{color:#52677E;opacity:1!important}
[data-testid="stMain"] [data-testid="stCaptionContainer"] p{color:#52677E!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#C4D3E7!important}
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"],
[data-testid="stCheckbox"] p,[data-testid="stWidgetLabel"] p{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:anywhere}
@media(max-width:900px){.command{grid-template-columns:1fr;padding:24px}.command-grid{grid-template-columns:1fr 1fr}.command h2{font-size:1.65rem!important}}
@media(max-width:640px){[data-testid="stMainBlockContainer"]{padding:4rem .85rem 3rem}.mast{gap:10px;align-items:flex-start}.brand-name{font-size:18px}.eyebrow{font-size:8px}.brand-mark{width:36px;height:36px}.status-pill{font-size:9px;padding:6px 8px}.command{padding:21px 18px;border-radius:15px;gap:18px}.command p{font-size:12px}.signalbox{padding:12px}.signalbox strong{font-size:23px}.kpi{min-height:94px;padding:15px}.kpi-value{font-size:24px}h1{font-size:1.9rem!important}.material-tile{min-height:145px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media(max-width:380px){.mast{flex-wrap:wrap}.status-pill{font-size:9px}.brand-name{font-size:19px}}
</style>
"""


def esc(value):
    return html.escape(str(value))


def current():
    return st.session_state["mp_project"]


def site_id():
    return st.session_state["mp_site"]


def ui_key(label):
    return f"{label}_{st.session_state['mp_mode']}_{st.session_state['mp_revision']}_{site_id()}"


def commit(p, message):
    try:
        validate(p)
        if st.session_state["mp_mode"] == "Mes données":
            st.session_state["mp_db_revision"] = save_real(p, st.session_state["mp_db_revision"])
        st.session_state["mp_project"] = p
        st.session_state["mp_revision"] += 1
        st.session_state.pop("mp_forecast", None)
        st.session_state["mp_message"] = message
        st.rerun()
    except (ValueError, KeyError, TypeError, sqlite3.Error, OSError) as exc:
        st.error(str(exc))


def context_changed():
    mode = st.session_state["mp_mode"]
    if mode == "Démonstration fictive":
        st.session_state["mp_project"] = demo_project()
    else:
        try:
            p, rev = load_real()
        except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
            st.session_state["mp_load_error"] = str(exc)
            return
        st.session_state["mp_project"] = p
        st.session_state["mp_db_revision"] = rev
    st.session_state["mp_revision"] += 1
    st.session_state["mp_site"] = st.session_state["mp_project"]["sites"][0]["id"]
    st.session_state.pop("mp_forecast", None)
    st.session_state.pop("mp_load_error", None)


def table_ui(df, label, **kwargs):
    st.dataframe(df, hide_index=True, width="stretch", alt=label, **kwargs)


def chart_ui(fig, label, height=330):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=15, r=15, t=25, b=45), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFCFF", font=dict(family="Arial", color="#29425F", size=12), colorway=COLORS, legend=dict(orientation="h", y=-.22, x=0), hovermode="x unified")
    st.plotly_chart(fig, width="stretch", alt=label, config={"displayModeBar": False, "responsive": True})


def kpis(items):
    cols = st.columns(len(items), gap="small")
    for col, (label, value, note, color) in zip(cols, items):
        with col:
            st.markdown(f'<div class="kpi" style="--accent:{color}"><div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div><div class="kpi-note">{esc(note)}</div></div>', unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="section-label">{esc(label)}</div>', unsafe_allow_html=True)


def title(name, caption):
    st.title(name)
    st.caption(caption)


def pick_material(p, label="Matériau", key=None):
    return st.selectbox(label, [m["id"] for m in p["materials"]], format_func=lambda x: f"{lookup(p, 'materials', x)['code']} · {lookup(p, 'materials', x)['name']} ({lookup(p, 'materials', x)['unit']})", key=key)


def pick_zone(p, label, zones=None, key=None):
    zones = p["zones"] if zones is None else zones
    return st.selectbox(label, [z["id"] for z in zones], format_func=lambda x: f"{lookup(p, 'zones', x)['code']} · {lookup(p, 'zones', x)['name']}", key=key)


def qty_input(label, m, value=0.0, key=None):
    return st.number_input(f"{label} ({m['unit']})", min_value=0 if m["unit"] in DISCRETE else 0.0, value=int(value) if m["unit"] in DISCRETE else float(value), step=1 if m["unit"] in DISCRETE else .1, key=key)


def latest_day(p):
    return date.fromisoformat(p["events"][-1]["date"] if p["events"] else p["start"])


def current_forecast():
    stored = st.session_state.get("mp_forecast")
    if stored and stored["site"] == site_id() and stored["hash"] == fingerprint(current()) and stored["result"]["cfg"]["asof"] == today().isoformat():
        return stored["result"]
    return None


def cockpit_page():
    p, sid = current(), site_id()
    stocks = site_stock(p, sid)
    open_orders = [o for o in p["orders"] if lookup(p, "zones", o["zone_id"])["site_id"] == sid and not o["closed"] and order_remaining(p, o) > EPS]
    late = sum(o["eta"] < today().isoformat() for o in open_orders)
    alert = sum(stocks[m["id"]] <= plan_for(p, sid, m["id"])["threshold"] for m in p["materials"] if plan_for(p, sid, m["id"])["threshold"] > 0)
    title("Le chantier, côté matières.", "Stocks par zone · Traçabilité des mouvements · Réceptions et prévisions")
    st.markdown(f'''<div class="command"><div><div class="eyebrow">Poste de pilotage / {esc(today().strftime('%d.%m.%Y'))}</div><h2>{esc(lookup(p, 'sites', sid)['name'])}</h2><p>Ce qui entre. Ce qui sort. Ce qui reste.<br>Un registre pour rapprocher vos bons, vos équipes et vos stocks physiques.</p></div><div class="command-grid"><div class="signalbox"><strong>{len(site_zones(p, sid)):02d}</strong><span>zones de stockage</span></div><div class="signalbox"><strong>{len(open_orders):02d}</strong><span>commandes ouvertes</span></div><div class="signalbox"><strong>{late:02d}</strong><span>échéances dépassées</span></div><div class="signalbox"><strong>{alert:02d}</strong><span>seuils de stock atteints</span></div></div></div>''', unsafe_allow_html=True)
    if p["demo"]:
        st.markdown('<div class="note"><b>Démonstration fictive.</b> Quantités, prix et fournisseurs sont inventés pour tester les outils. Choisis « Mes données » dans le menu pour commencer un registre vide.</div>', unsafe_allow_html=True)
    section("01 / Disponibilité des matériaux")
    for offset in range(0, len(p["materials"]), 3):
        for col, m in zip(st.columns(3, gap="small"), p["materials"][offset:offset + 3]):
            q = stocks[m["id"]]
            threshold = plan_for(p, sid, m["id"])["threshold"]
            state = "À réapprovisionner" if threshold > 0 and q <= threshold else "Seuil non défini" if threshold == 0 else "Au-dessus du seuil"
            ratio = min(100, q / threshold * 100) if threshold else 0
            threshold_note = f"Seuil : {fmt(threshold)} {m['unit']}" if threshold else "Définir le seuil dans Prévisions"
            idx = p["materials"].index(m)
            with col:
                st.markdown(f'<div class="material-tile" style="--accent:{COLORS[idx % 5]}"><div class="material-top"><span>{esc(m["code"])}</span><span>STOCK THÉORIQUE</span></div><div class="material-name">{esc(m["name"])}</div><div class="material-value">{fmt(q, 0 if m["unit"] in DISCRETE else 2)} <small>{esc(m["unit"])}</small></div><div class="stockbar" title="Couverture du seuil choisi, plafonnée à 100 %"><i style="width:{ratio}%"></i></div><div class="material-top">{esc(state)}</div><div class="material-top">{esc(threshold_note)}</div></div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    if a.button("＋ Enregistrer un mouvement", type="primary", width="stretch"):
        st.switch_page(PAGES[2])
    if b.button("↗ Réceptionner une livraison", width="stretch"):
        st.switch_page(PAGES[3])
    if c.button("◉ Prévoir une rupture", width="stretch"):
        st.switch_page(PAGES[5])
    section("02 / Flux et approvisionnements")
    a, b = st.columns([1.15, 1])
    with a:
        st.subheader("Où se trouve le stock ?")
        mid = pick_material(p, key=ui_key("cockpit_material"))
        m = lookup(p, "materials", mid)
        bb = balances(p)
        fig = go.Figure(go.Bar(x=[z["code"] for z in site_zones(p, sid)], y=[bb[z["id"], mid] for z in site_zones(p, sid)], marker_color=COLORS[p["materials"].index(m) % 5], text=[fmt(bb[z["id"], mid]) for z in site_zones(p, sid)], textposition="outside"))
        fig.update_yaxes(title=m["unit"], rangemode="tozero")
        chart_ui(fig, f"Stock de {m['name']} par zone en {m['unit']}.")
    with b:
        st.subheader("Prochaines livraisons")
        df = orders_table(p, sid)
        if open_orders:
            codes = {o["code"] for o in open_orders}
            table_ui(df[df["Commande"].isin(codes)].sort_values("Prévu le")[["Commande", "Matériau", "Reste dû", "Unité", "Prévu le"]], "Commandes ouvertes et reste dû par unité.", height=330)
        else:
            st.info("Aucune livraison en attente. Crée une commande pour suivre les prochaines entrées.")
    value = sum(stocks[m["id"]] * m["price"] for m in p["materials"] if m["price"] is not None)
    unknown = sum(stocks[m["id"]] > EPS and m["price"] is None for m in p["materials"])
    kpis([("Valeur de référence connue", money(value), f"{unknown} référence(s) en stock sans prix", COLORS[0]), ("Journal du chantier", str(len(journal_table(p, sid))), "Mouvements et contre-écritures", COLORS[2]), ("Dernier enregistrement", p["events"][-1]["date"] if p["events"] else "Aucun", "Le comptage physique reste nécessaire", COLORS[1])])
    st.subheader("Derniers mouvements")
    df = journal_table(p, sid)
    if df.empty:
        st.info("Ton registre est prêt. Commence par un stock initial, puis enregistre les consommations et les livraisons.")
    else:
        table_ui(df.head(8).drop(columns=["ID", "Acteur", "Note"]), "Les huit derniers mouvements du chantier.")


def catalogue_page():
    p, sid = current(), site_id()
    title("Le référentiel", "Chantiers, zones, matériaux et fournisseurs : des repères stables pour chaque bon.")
    tabs = st.tabs(["Matériaux", "Chantiers & zones", "Fournisseurs"])
    with tabs[0]:
        st.subheader("Une unité fixe par référence")
        st.caption("Pas de conversion automatique entre sacs, tonnes et mètres cubes. Le prix est une référence de valorisation, distincte du prix d’une commande.")
        choices = ["__new__"] + [m["id"] for m in p["materials"]]
        mid = st.selectbox("Référence à éditer", choices, format_func=lambda x: "＋ Nouveau matériau" if x == "__new__" else lookup(p, "materials", x)["name"], key=ui_key("cat_mat"))
        old = None if mid == "__new__" else lookup(p, "materials", mid)
        used = old and (any(e["material_id"] == mid for e in p["events"]) or any(o["material_id"] == mid for o in p["orders"]))
        with st.form(ui_key("material_form") + mid):
            a, b, c = st.columns([1, 2, 1])
            code = a.text_input("Code unique", value=old["code"] if old else "", max_chars=24, disabled=bool(old))
            name = b.text_input("Nom", value=old["name"] if old else "", max_chars=120)
            unit = c.selectbox("Unité", UNITS, index=UNITS.index(old["unit"]) if old else 0, disabled=bool(used))
            priced = st.checkbox("Prix de référence renseigné", value=old is not None and old["price"] is not None)
            price = st.number_input("Prix de référence (FCFA / unité)", min_value=0.0, value=float(old["price"] or 0) if old else 0.0)
            source = st.text_input("Origine / date de ce prix", value=old["price_source"] if old else "", max_chars=200)
            if used:
                st.caption("L’unité est verrouillée dès qu’un mouvement ou une commande existe.")
            if st.form_submit_button("Enregistrer la référence", type="primary"):
                q = copy.deepcopy(p)
                item = dict(id=old["id"] if old else ident("MAT"), code=code.strip().upper(), name=name, unit=unit, price=price if priced else None, price_source=source if priced else "")
                if old:
                    q["materials"][q["materials"].index(lookup(q, "materials", mid))] = item
                    if old["unit"] != unit:
                        for profile in q["plans"].values():
                            profile.pop(mid, None)
                else:
                    q["materials"].append(item)
                commit(q, "Référence enregistrée.")
        rows = [{"Code": m["code"], "Matériau": m["name"], "Unité": m["unit"], "Prix FCFA/unité": m["price"], "Source": m["price_source"]} for m in p["materials"]]
        table_ui(pd.DataFrame(rows), "Catalogue des matériaux et prix de référence.")
    with tabs[1]:
        a, b = st.columns(2)
        with a:
            with st.form(ui_key("new_site")):
                st.subheader("Créer un chantier")
                name = st.text_input("Nom du nouveau chantier", max_chars=120)
                zone_code = st.text_input("Code du premier dépôt", max_chars=24, help="Code unique dans tout le registre.")
                if st.form_submit_button("Créer le chantier", type="primary"):
                    q = copy.deepcopy(p)
                    ns = ident("SITE")
                    q["sites"].append(dict(id=ns, name=name))
                    q["zones"].append(dict(id=ident("ZONE"), code=zone_code.strip().upper(), name="Dépôt principal", site_id=ns))
                    commit(q, "Chantier créé. Sélectionne-le dans le menu.")
        with b:
            with st.form(ui_key("new_zone")):
                st.subheader("Ajouter une zone ici")
                code = st.text_input("Code unique de la zone", max_chars=24)
                name = st.text_input("Nom de la zone", max_chars=120)
                if st.form_submit_button("Ajouter la zone"):
                    q = copy.deepcopy(p)
                    q["zones"].append(dict(id=ident("ZONE"), code=code.strip().upper(), name=name, site_id=sid))
                    commit(q, "Zone ajoutée au chantier.")
        rows = [{"Chantier": lookup(p, "sites", z["site_id"])["name"], "Zone": z["code"], "Nom": z["name"]} for z in p["zones"]]
        table_ui(pd.DataFrame(rows), "Chantiers et zones de stockage du registre.")
        with st.expander("Renommer le chantier sélectionné"):
            with st.form(ui_key("rename_site")):
                name = st.text_input("Nouveau nom", value=lookup(p, "sites", sid)["name"], max_chars=120)
                if st.form_submit_button("Modifier le nom"):
                    q = copy.deepcopy(p)
                    lookup(q, "sites", sid)["name"] = name
                    commit(q, "Chantier renommé.")
    with tabs[2]:
        st.subheader("Les partenaires d’approvisionnement")
        with st.form(ui_key("supplier")):
            name = st.text_input("Nom du fournisseur à ajouter", max_chars=120)
            if st.form_submit_button("Ajouter le fournisseur", type="primary"):
                q = copy.deepcopy(p)
                q["suppliers"].append(dict(id=ident("FOUR"), name=name))
                commit(q, "Fournisseur ajouté.")
        rows = []
        for supplier in p["suppliers"]:
            orders = [o for o in p["orders"] if o["supplier_id"] == supplier["id"]]
            rows.append({"Fournisseur": supplier["name"], "Commandes": len(orders), "Ouvertes": sum(not o["closed"] and order_remaining(p, o) > EPS for o in orders), "En retard": sum(not o["closed"] and order_remaining(p, o) > EPS and o["eta"] < today().isoformat() for o in orders), "Bons avec refus": sum(e["type"] == "Réception" and e["order_id"] in {o["id"] for o in orders} and e.get("rejected", 0) > 0 for e in effective_events(p))})
        if rows:
            table_ui(pd.DataFrame(rows), "Indicateurs factuels par fournisseur, sans additionner des unités différentes.")


def movement_page():
    p, sid = current(), site_id()
    title("Le journal des mouvements", "Un bon, une date de prise en compte, une origine et une destination. Aucun stock négatif accepté.")
    entry, journal, imports = st.tabs(["Nouveau mouvement", "Journal & corrections", "Import CSV / Excel"])
    with entry:
        a, b = st.columns(2)
        kind = a.selectbox("Opération", KINDS[:5], key=ui_key("movement_kind"))
        mid = pick_material(p, key=ui_key("movement_mat"))
        m = lookup(p, "materials", mid)
        src = pick_zone(p, "Zone de départ", key=ui_key("movement_src")) if kind in ("Consommation", "Perte / casse", "Transfert") else None
        dst = pick_zone(p, "Zone de destination", key=ui_key("movement_dst")) if kind in ("Stock initial", "Retour d’utilisation", "Transfert") else None
        if src:
            st.info(f"Disponible à cet instant : {fmt(balances(p)[src, mid])} {m['unit']} dans {lookup(p, 'zones', src)['code']}.")
        with st.form(ui_key("movement_form") + mid + kind):
            a, b = st.columns(2)
            with a:
                qty = qty_input("Quantité", m, value=1)
                d = st.date_input("Date de prise en compte", value=today(), min_value=latest_day(p), max_value=today())
            with b:
                doc = st.text_input("Bon / référence unique", max_chars=80, placeholder="Ex. SORTIE-2026-001")
                actor = st.text_input("Agent / équipe (facultatif)", max_chars=100)
            note = st.text_area("Motif / observations", max_chars=500, help="Obligatoire pour une perte. Une ancienne date documentaire peut être précisée ici.")
            if st.form_submit_button("Inscrire le mouvement", type="primary"):
                e = movement(kind, mid, qty, src=src, dst=dst, day=d.isoformat(), doc=doc)
                e.update(actor=actor, note=note)
                try:
                    commit(append_event(p, e), "Mouvement enregistré. Les stocks sont recalculés.")
                except ValueError as exc:
                    st.error(str(exc))
        st.caption("Un retour d’utilisation est une restitution déclarée. L’application ne prouve pas l’origine du matériau ; rapproche les bons et le terrain.")
    with journal:
        df = journal_table(p, sid)
        table_ui(df, "Journal chronologique du chantier, y compris les annulations.")
        st.download_button("Télécharger le journal CSV", safe_csv(df), "materiapilot-journal.csv", "text/csv", on_click="ignore")
        with st.expander("Corriger une erreur par une annulation"):
            candidates = [e for e in effective_events(p) if event_site(p, e, sid)]
            if not candidates:
                st.info("Aucun mouvement actif à annuler.")
            else:
                eid = st.selectbox("Mouvement original", [e["id"] for e in reversed(candidates)], format_func=lambda x: next(f"{e['date']} · {e['doc']} · {e['type']} · {fmt(e['qty'])} {lookup(p, 'materials', e['material_id'])['unit']}" for e in candidates if e["id"] == x), key=ui_key("reverse_choice"))
                with st.form(ui_key("reverse_form") + eid):
                    note = st.text_input("Motif de l’annulation", max_chars=500)
                    confirmed = st.checkbox("Je confirme la contre-écriture ; le mouvement original restera visible.")
                    if st.form_submit_button("Annuler ce mouvement"):
                        if not confirmed:
                            st.error("Confirme la contre-écriture.")
                        else:
                            orig = next(e for e in p["events"] if e["id"] == eid)
                            e = movement("Annulation", orig["material_id"], orig["qty"], doc="ANN-" + ident("")[:10], reverse_of=eid)
                            e["note"] = note
                            try:
                                commit(append_event(p, e), "Contre-écriture enregistrée.")
                            except ValueError as exc:
                                st.error(str(exc))
                st.caption("Une annulation est refusée si elle rend un stock négatif. Après annulation, saisis le bon corrigé avec une nouvelle référence.")
    with imports:
        st.subheader("Importer un lot de bons")
        st.caption("Maximum 300 lignes et 5 Mo. Le fichier entier est vérifié avant tout enregistrement. Codes et dates doivent correspondre au registre ; les bons sont triés par date croissante.")
        st.download_button("Modèle CSV vide", safe_csv(pd.DataFrame(columns=IMPORT_COLS)), "modele-mouvements.csv", "text/csv", on_click="ignore")
        st.code("date,type,materiau_code,source_zone,destination_zone,quantite,compte_physique,document,acteur,note\n" + today().isoformat() + ",Stock initial,CIM,,DEP,10,,EXEMPLE-FICTIF,Équipe,Exemple fictif à adapter", language="csv")
        upload = st.file_uploader("Fichier de mouvements", type=["csv", "xlsx"], key=ui_key("import_upload"))
        if upload:
            try:
                if upload.size > 5_000_000:
                    raise ValueError("Fichier trop volumineux : maximum 5 Mo.")
                if upload.name.lower().endswith(".xlsx"):
                    with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as archive:
                        if len(archive.infolist()) > 500 or sum(x.file_size for x in archive.infolist()) > 20_000_000:
                            raise ValueError("Classeur décompressé trop volumineux : maximum 20 Mo et 500 éléments internes.")
                    frame = pd.read_excel(upload, dtype=str, keep_default_na=False, engine="openpyxl")
                else:
                    frame = pd.read_csv(upload, dtype=str, keep_default_na=False)
                table_ui(frame.head(30), "Aperçu des trente premières lignes du fichier à importer.")
                trial = import_movements(p, frame)
                st.info(f"{len(frame)} mouvements valides. Aucun n’est encore enregistré.")
                confirmed = st.checkbox("J’ai vérifié les unités et les références et je souhaite ajouter ce lot.", key=ui_key("import_confirm"))
                if st.button("Enregistrer le lot validé", type="primary", disabled=not confirmed):
                    commit(trial, f"{len(frame)} mouvements importés.")
            except (ValueError, KeyError, TypeError, OSError, ImportError, zipfile.BadZipFile) as exc:
                st.error(str(exc))


def orders_page():
    p, sid = current(), site_id()
    title("Commandes & réceptions", "L’accepté entre en stock ; le refusé reste dû. Une commande peut être livrée en plusieurs fois.")
    table_ui(orders_table(p, sid), "Suivi des commandes du chantier, livraisons partielles et refus.")
    tabs = st.tabs(["Créer une commande", "Réceptionner", "Réviser / clôturer"])
    with tabs[0]:
        if not p["suppliers"]:
            st.info("Ajoute un fournisseur dans le Référentiel avant de créer une commande.")
        else:
            mid = pick_material(p, key=ui_key("order_mat"))
            m = lookup(p, "materials", mid)
            zone = pick_zone(p, "Dépôt destinataire", site_zones(p, sid), key=ui_key("order_zone"))
            with st.form(ui_key("order_new") + mid):
                a, b = st.columns(2)
                with a:
                    code = st.text_input("N° de commande", max_chars=80)
                    supplier = st.selectbox("Fournisseur", [s["id"] for s in p["suppliers"]], format_func=lambda x: lookup(p, "suppliers", x)["name"])
                    qty = qty_input("Quantité commandée", m, value=1)
                    priced = st.checkbox("Prix commandé connu")
                    price = st.number_input("Prix commandé (FCFA / unité)", min_value=0.0, value=float(m["price"] or 0))
                with b:
                    d = st.date_input("Date de commande", today(), min_value=date.fromisoformat(p["start"]), max_value=today())
                    eta = st.date_input("Date prévue de livraison", today() + timedelta(days=3), min_value=date.fromisoformat(p["start"]))
                    delay = st.number_input("Retard maximal supposé (jours)", min_value=0, max_value=90, value=2, help="Utilisé dans la simulation : retard uniforme entier entre 0 et cette valeur, sans estimation automatique.")
                    note = st.text_area("Référence du devis / note", max_chars=500)
                if st.form_submit_button("Créer la commande", type="primary"):
                    q = copy.deepcopy(p)
                    q["orders"].append(dict(id=ident("BC"), code=code.strip(), material_id=mid, zone_id=zone, supplier_id=supplier, qty=float(qty), price=price if priced else None, date=d.isoformat(), eta=eta.isoformat(), delay=delay, closed=False, note=note))
                    commit(q, "Commande créée.")
    with tabs[1]:
        options = [o for o in p["orders"] if lookup(p, "zones", o["zone_id"])["site_id"] == sid and not o["closed"] and order_remaining(p, o) > EPS]
        if not options:
            st.info("Aucune commande ouverte à réceptionner sur ce chantier.")
        else:
            oid = st.selectbox("Commande à réceptionner", [o["id"] for o in options], format_func=lambda x: lookup(p, "orders", x)["code"] + " · " + lookup(p, "materials", lookup(p, "orders", x)["material_id"])["name"], key=ui_key("receipt_choice"))
            o = lookup(p, "orders", oid)
            m = lookup(p, "materials", o["material_id"])
            st.info(f"Reste dû : {fmt(order_remaining(p, o))} {m['unit']} · destination {lookup(p, 'zones', o['zone_id'])['code']}.")
            with st.form(ui_key("receipt_form") + oid):
                a, b = st.columns(2)
                with a:
                    qty = qty_input("Quantité acceptée", m, value=0)
                    rejected = qty_input("Quantité refusée", m, value=0)
                with b:
                    d = st.date_input("Date de prise en compte", today(), min_value=max(latest_day(p), date.fromisoformat(o["date"])), max_value=today())
                    doc = st.text_input("N° du bon de livraison", max_chars=80)
                note = st.text_area("Observations / motif du refus", max_chars=500)
                if st.form_submit_button("Enregistrer la réception", type="primary"):
                    e = movement("Réception", m["id"], qty, dst=o["zone_id"], day=d.isoformat(), doc=doc, order_id=oid, rejected=float(rejected))
                    e["note"] = note
                    try:
                        if rejected > 0 and not note.strip():
                            raise ValueError("Précise le motif de la quantité refusée.")
                        commit(append_event(p, e), "Réception enregistrée. Seule la quantité acceptée entre en stock.")
                    except ValueError as exc:
                        st.error(str(exc))
    with tabs[2]:
        options = [o for o in p["orders"] if lookup(p, "zones", o["zone_id"])["site_id"] == sid]
        if not options:
            st.info("Crée d’abord une commande.")
        else:
            oid = st.selectbox("Commande à réviser", [o["id"] for o in options], format_func=lambda x: lookup(p, "orders", x)["code"], key=ui_key("revise_order"))
            o = lookup(p, "orders", oid)
            st.caption("La référence, le matériau, le fournisseur, le prix et le dépôt sont conservés. Une correction de réception passe par le journal.")
            with st.form(ui_key("revise_order_form") + oid):
                eta = st.date_input("Nouvelle date prévue", date.fromisoformat(o["eta"]), min_value=date.fromisoformat(o["date"]))
                delay = st.number_input("Retard maximal supposé (jours)", min_value=0, max_value=90, value=int(o["delay"]))
                closed = st.checkbox("Clôturer : ne plus attendre le solde de cette commande", value=o["closed"])
                note = st.text_area("Note de suivi", value=o["note"], max_chars=500)
                if st.form_submit_button("Mettre à jour la commande"):
                    q = copy.deepcopy(p)
                    lookup(q, "orders", oid).update(eta=eta.isoformat(), delay=delay, closed=closed, note=note)
                    commit(q, "Commande mise à jour. Les prévisions précédentes ont été invalidées.")


def inventory_page():
    p, sid = current(), site_id()
    title("Inventaire & consommation", "Comparer le stock enregistré au terrain, puis comprendre les sorties réellement déclarées.")
    count_tab, analysis_tab = st.tabs(["Comptage physique", "Analyse des mouvements"])
    with count_tab:
        a, b = st.columns(2)
        with a:
            mid = pick_material(p, key=ui_key("count_mat"))
        with b:
            zone = pick_zone(p, "Zone comptée", site_zones(p, sid), key=ui_key("count_zone"))
        m = lookup(p, "materials", mid)
        theoretical = balances(p)[zone, mid]
        st.markdown(f'<div class="guidebox">Stock théorique : <b>{fmt(theoretical)} {esc(m["unit"])}</b>.<br>Écart = quantité comptée − quantité théorique. Un comptage égal au stock produit un contrôle à écart nul.</div>', unsafe_allow_html=True)
        with st.form(ui_key("count_form") + mid + zone):
            count = qty_input("Quantité réellement comptée", m, value=theoretical)
            doc = st.text_input("Référence de la fiche d’inventaire", max_chars=80)
            note = st.text_area("Explication / méthode du comptage", max_chars=500)
            if st.form_submit_button("Valider le comptage", type="primary"):
                e = inventory_event(p, mid, zone, count, doc=doc)
                e["note"] = note
                try:
                    commit(append_event(p, e), "Comptage enregistré ; l’écart est intégré au stock.")
                except ValueError as exc:
                    st.error(str(exc))
        table_ui(inventory_table(p, sid), "Inventaire théorique détaillé par zone et matériau.")
    with analysis_tab:
        mid = pick_material(p, key=ui_key("analysis_mat"))
        m = lookup(p, "materials", mid)
        events = [e for e in effective_events(p) if e["material_id"] == mid and event_site(p, e, sid)]
        used = sum(e["qty"] for e in events if e["type"] == "Consommation")
        returned = sum(e["qty"] for e in events if e["type"] == "Retour d’utilisation")
        lost = sum(e["qty"] for e in events if e["type"] == "Perte / casse")
        kpis([("Consommation déclarée", fmt(used) + " " + m["unit"], "Hors mouvements annulés", COLORS[0]), ("Retours d’utilisation", fmt(returned) + " " + m["unit"], "Ne sont pas des livraisons", COLORS[2]), ("Pertes déclarées", fmt(lost) + " " + m["unit"], "À distinguer des écarts de comptage", COLORS[4])])
        if not events:
            st.info("Aucun mouvement actif pour cette référence sur ce chantier.")
        else:
            days = sorted({e["date"] for e in p["events"] if event_site(p, e, sid)})
            zones = {z["id"] for z in site_zones(p, sid)}
            fig = go.Figure()
            hist = []
            for d in days:
                b = balances(p, d)
                hist.append(sum(b[z, mid] for z in zones))
            fig.add_trace(go.Scatter(x=days, y=hist, mode="lines+markers", line=dict(color=COLORS[0], width=3, shape="hv"), name="Stock théorique"))
            fig.update_yaxes(title=m["unit"], rangemode="tozero")
            chart_ui(fig, f"Évolution du stock théorique enregistré de {m['name']}, fin de chaque date de mouvement.")
            st.caption("L’historique du stock inclut les contre-écritures à leur date. Les totaux de consommation ci-dessus excluent les mouvements annulés.")
            st.subheader("Utilisation et pertes par date déclarée")
            rows = [{"Date": e["date"], "Type": e["type"], "Quantité": e["qty"]} for e in events if e["type"] in ("Consommation", "Retour d’utilisation", "Perte / casse")]
            if rows:
                df = pd.DataFrame(rows).groupby(["Date", "Type"], as_index=False)["Quantité"].sum()
                fig = go.Figure()
                for kind, color in zip(["Consommation", "Retour d’utilisation", "Perte / casse"], [COLORS[0], COLORS[2], COLORS[4]]):
                    part = df[df["Type"] == kind]
                    fig.add_trace(go.Bar(x=part["Date"], y=part["Quantité"], name=kind, marker_color=color))
                fig.update_layout(barmode="group")
                fig.update_yaxes(title=m["unit"])
                chart_ui(fig, "Consommations, retours et pertes enregistrés par date et par type.")
                table_ui(df, "Données déclarées utilisées dans le graphique de consommation.")
            st.caption("Une date sans bon n’est pas interprétée comme une journée observée à consommation nulle. Les paramètres probabilistes sont saisis séparément.")


def forecast_page():
    p, sid = current(), site_id()
    title("Anticiper les besoins", "Demandes incertaines, calendrier de travail et réceptions planifiées : comparer des scénarios explicites.")
    profile_tab, simulation_tab, method_tab = st.tabs(["1 · Profils", "2 · Simulation", "Comprendre la méthode"])
    with profile_tab:
        mid = pick_material(p, key=ui_key("profile_mat"))
        m = lookup(p, "materials", mid)
        v = plan_for(p, sid, mid)
        models = ["Constante", "Poisson"] if m["unit"] in DISCRETE else ["Constante", "Triangulaire"]
        with st.form(ui_key("profile_form") + mid):
            enabled = st.checkbox("Inclure ce matériau dans les prévisions de ce chantier", value=v["enabled"])
            a, b = st.columns(2)
            with a:
                model = st.selectbox("Modèle de demande par jour d’activité", models, index=models.index(v["model"]))
                mean = st.number_input(f"Demande moyenne / jour d’activité ({m['unit']})", min_value=0.0, value=float(v["mean"]), step=.5)
                threshold = st.number_input(f"Seuil d’alerte du stock chantier ({m['unit']}, 0 = non défini)", min_value=0.0, value=float(v["threshold"]))
            with b:
                spread = st.slider("Amplitude triangulaire ± (%) · quantités continues", min_value=0, max_value=100, value=int(v["spread"]))
                source = st.text_input("Origine de l’hypothèse / date", value=v["source"], max_chars=200, help="Exemple : prévision de l’équipe pour la semaine à venir ; ou hypothèse fictive d’exercice.")
                st.caption("Poisson : compte entier de sacs / pièces, variance égale à la moyenne. Triangulaire : minimum μ(1−a), mode μ, maximum μ(1+a).")
            if st.form_submit_button("Enregistrer ce profil", type="primary"):
                q = copy.deepcopy(p)
                q["plans"].setdefault(sid, {})[mid] = dict(enabled=enabled, model=model, mean=mean, spread=spread, source=source, threshold=threshold)
                commit(q, "Profil de consommation enregistré pour ce chantier.")
        rows = []
        for mm in p["materials"]:
            vv = plan_for(p, sid, mm["id"])
            rows.append({"Matériau": mm["name"], "Actif": vv["enabled"], "Modèle": vv["model"], "Moyenne / jour actif": vv["mean"], "Unité": mm["unit"], "Origine": vv["source"]})
        table_ui(pd.DataFrame(rows), "Profils de demande supposée par matériau pour le chantier.")
    with simulation_tab:
        active = [m for m in p["materials"] if plan_for(p, sid, m["id"])["enabled"]]
        if not active:
            st.info("Active au moins un profil dans l’onglet « 1 · Profils », puis reviens ici.")
        else:
            with st.form(ui_key("simulation_settings")):
                a, b, c = st.columns(3)
                with a:
                    n = st.selectbox("Nombre de scénarios", [500, 2000, 5000, 10000, 20000], index=2)
                    h = st.slider("Horizon (jours calendaires)", 7, 90, 30)
                    seed = st.number_input("Graine reproductible", min_value=0, max_value=999999, value=42)
                with b:
                    days = st.multiselect("Jours d’activité", list(range(7)), default=list(range(6)), format_func=lambda x: ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][x])
                    interruption = st.slider("Probabilité d’interruption commune par jour (%)", 0, 100, 0, help="La même interruption annule toutes les consommations du jour dans un scénario ; les livraisons restent possibles.")
                with c:
                    timing = st.selectbox("Arrivée des livraisons", ["Début de journée", "Fin de journée"])
                    overdue = st.checkbox("Reporter les commandes échues à J1", value=False, help="Hypothèse explicite de report. Sinon, les commandes échues sont exclues. Une livraison aujourd’hui non enregistrée est aussi exclue par défaut.")
                st.caption("Le retard de chaque commande est uniforme entre 0 et le maximum saisi dans Commandes. J1 = demain. Le stock de toutes les zones du chantier est supposé immédiatement accessible.")
                st.markdown("**Levier à comparer (facultatif)**")
                a, b, c = st.columns(3)
                extra_mat = a.selectbox("Apport supplémentaire hypothétique", ["__none__"] + [m["id"] for m in active], format_func=lambda x: "Aucun" if x == "__none__" else lookup(p, "materials", x)["name"])
                extra_qty = b.number_input("Quantité supplémentaire (unité du matériau)", min_value=0.0, value=0.0)
                extra_day = c.number_input("Jour d’arrivée de cet apport", min_value=1, max_value=90, value=3)
                if st.form_submit_button("Lancer les simulations", type="primary", width="stretch"):
                    cfg = dict(n=n, h=h, seed=seed, weekdays=days, interruption=interruption, timing=timing, overdue=overdue, asof=today().isoformat())
                    try:
                        with st.spinner("Calcul des demandes, réceptions et premiers manques…"):
                            result = simulate(json.dumps(p), sid, json.dumps(cfg))
                            compare = None
                            if extra_mat != "__none__":
                                cfg_extra = copy.deepcopy(cfg)
                                cfg_extra["extra"] = dict(material_id=extra_mat, qty=extra_qty, day=extra_day)
                                compare = simulate(json.dumps(p), sid, json.dumps(cfg_extra))
                            st.session_state["mp_forecast"] = dict(site=sid, hash=fingerprint(p), result=result, compare=compare)
                    except (ValueError, TypeError, OverflowError) as exc:
                        st.error(str(exc))
            result = current_forecast()
            if result:
                show_forecast(result)
            else:
                st.caption("Chaque changement du registre ou des profils invalide les anciens résultats. Lance un calcul pour obtenir une prévision à jour.")
    with method_tab:
        st.markdown("""### Ce qui est calculé
Le premier jour de rupture est le jour où une demande dépasse le stock disponible **au moment de son utilisation**. Un stock nul en fin de journée ne signifie pas, à lui seul, qu’une demande a été refusée.

1. Le stock initial provient du journal du chantier sélectionné.
2. Une demande est tirée par matériau et par jour d’activité. Les jours fermés n’ont pas de demande. Une interruption commune peut suspendre tous les matériaux le même jour.
3. Les restes dus des commandes ouvertes arrivent à la date prévue, augmentée d’un retard entier supposé uniforme. Les refus déjà déclarés n’augmentent pas le stock.
4. Le service s’effectue avec livraison en début ou fin de journée, selon ton choix. La demande non servie est perdue : aucun arriéré n’est reporté.
5. On estime la proportion de scénarios avec au moins un manque. Le risque global porte sur **au moins un matériau**, sans additionner les pourcentages.

### Lire les graphiques
- **Q10 / médiane / Q90 du stock** : des repères entre scénarios ; ce ne sont pas trois trajectoires garanties.
- **Jour médian du premier manque** : calcul sur tous les scénarios. « > H j » signifie qu’il dépasse l’horizon d’étude, sans inventer une date de rupture.
- **IC MC 95 %** : intervalle de Wilson pour l’erreur d’échantillonnage de la simulation. Il ne couvre pas les erreurs de modèle ni de saisie.

### Hypothèses et limites
Les demandes des matériaux sont indépendantes, sauf l’interruption commune. Les retards sont indépendants des demandes et entre commandes. Une loi de Poisson ne convient pas à tous les rythmes de chantier ; choisis une constante pour une consommation imposée. Les sacs et pièces restent entiers. Les zones sont agrégées sans temps de transfert. Aucun réapprovisionnement automatique n’est créé par une prévision. Le stock théorique doit être rapproché du comptage physique.
""")
        st.latex(r"S_{j+1}=\max(S_j+A_j-D_j,0)\quad\text{(livraison en début de journée)}")
        st.caption("S = stock ; A = quantité reçue ; D = demande. Les manques sont comptés avant la remise à zéro.")


def show_forecast(result):
    n, h = result["n"], result["h"]
    k = int(np.count_nonzero(result["global_first"] <= h))
    lo, hi = wilson(k, n)
    section("Résultats simulés / distincts des mouvements réels")
    kpis([("Au moins une rupture", fmt(k / n * 100, 1) + " %", f"{n:,} scénarios / {h} jours".replace(",", " "), COLORS[4]), ("IC Monte-Carlo 95 %", f"{fmt(lo*100,1)}–{fmt(hi*100,1)} %", "Échantillonnage du modèle uniquement", COLORS[0]), ("Premier manque médian", risk_label(result["global_first"], .5, h), "Tous matériaux actifs confondus", COLORS[1])])
    if result["ignored"]:
        st.warning("Commandes échues exclues, date à revoir : " + ", ".join(result["ignored"]))
    summary, daily = forecast_tables(result)
    display = summary.copy()
    for col in ("P rupture", "IC MC 95% bas", "IC MC 95% haut"):
        display[col] = display[col].map(lambda v: fmt(v * 100, 1) + " %")
    table_ui(display, "Probabilités de rupture et quantiles du premier jour de manque par matériau.")
    chosen = st.selectbox("Matériau à visualiser", [r["id"] for r in result["materials"]], format_func=lambda x: next(r["name"] for r in result["materials"] if r["id"] == x), key=ui_key("forecast_chart"))
    r = next(r for r in result["materials"] if r["id"] == chosen)
    days = [0] + list(range(1, h + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=r["bands"][2], line=dict(width=0), showlegend=False, name="Q90"))
    fig.add_trace(go.Scatter(x=days, y=r["bands"][0], fill="tonexty", fillcolor="rgba(36,87,184,.14)", line=dict(width=0), name="Bande Q10–Q90"))
    fig.add_trace(go.Scatter(x=days, y=r["bands"][1], line=dict(color=COLORS[0], width=3), name="Stock médian"))
    fig.update_xaxes(title="Jour calendaire (J0 = aujourd’hui)")
    fig.update_yaxes(title=r["unit"], rangemode="tozero")
    st.subheader("Trajectoire du stock disponible")
    chart_ui(fig, f"Stock simulé de {r['name']} : médiane et bande des quantiles 10 et 90 pour cent.", 380)
    a, b = st.columns(2)
    with a:
        st.subheader("Risque cumulé de premier manque")
        fig = go.Figure(go.Scatter(x=list(range(1, h + 1)), y=[100 * np.mean(r["first"] <= j) for j in range(1, h + 1)], fill="tozeroy", line=dict(color=COLORS[4], width=3), name="Risque cumulé"))
        fig.update_yaxes(title="Scénarios avec manque (%)", range=[0, 100])
        fig.update_xaxes(title="Jour")
        chart_ui(fig, "Part des scénarios avec un premier manque avant chaque jour.")
    with b:
        st.subheader("Demande et service moyens")
        fig = go.Figure()
        for label, vals, color in [("Servi", r["served"], COLORS[2]), ("Non servi", r["missing"], COLORS[4])]:
            fig.add_trace(go.Bar(x=list(range(1, h + 1)), y=vals, name=label, marker_color=color))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title=r["unit"] + " / jour")
        chart_ui(fig, "Quantités moyennes demandées, servies et non servies par jour.")
    comparison = st.session_state["mp_forecast"].get("compare")
    if comparison:
        k2 = int(np.count_nonzero(comparison["global_first"] <= h))
        extra = comparison["cfg"]["extra"]
        m = lookup(current(), "materials", extra["material_id"])
        st.markdown(f'<div class="guidebox"><b>Comparer un apport hypothétique</b><br>{fmt(extra["qty"])} {esc(m["unit"])} de {esc(m["name"])} à J{extra["day"]} : risque global {fmt(100*k/n,1)} % → <b>{fmt(100*k2/n,1)} %</b>.<br>Mêmes tirages pour les deux scénarios. Cet apport ne modifie ni le journal ni les commandes.</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.download_button("Résumé CSV", safe_csv(summary), "previsions-resume.csv", "text/csv", on_click="ignore", width="stretch")
    b.download_button("Courbes CSV", safe_csv(daily), "previsions-courbes.csv", "text/csv", on_click="ignore", width="stretch")
    scenarios = pd.DataFrame({f"{x['name']} · jour premier manque": np.where(x["first"] <= h, x["first"], np.nan) for x in result["materials"]})
    scenarios.insert(0, "Scénario", np.arange(1, n + 1))
    scenarios["Horizon jours"] = h
    c.download_button("Scénarios CSV", safe_csv(scenarios), "previsions-scenarios.csv", "text/csv", on_click="ignore", width="stretch")
    st.caption("Dans les scénarios CSV, un jour vide signifie : aucun manque observé avant la fin de l’horizon. Ce n’est pas un jour de rupture inconnu transformé en H+1.")


def reports_page():
    p, sid = current(), site_id()
    title("Rapports & sauvegardes", "Emporter les résultats, conserver le registre et disposer d’exemples vérifiables.")
    exports, backup, guide = st.tabs(["Rapports", "Sauvegarde / restauration", "Guide & tests"])
    with exports:
        forecast = current_forecast()
        st.subheader("Le dossier matériaux du chantier")
        st.caption("Les exports couvrent le chantier sélectionné. Le classeur inclut aussi le catalogue général. Les prévisions sont incluses seulement lorsqu’elles sont à jour et correspondent à ce chantier.")
        a, b, c = st.columns(3)
        a.download_button("↓ Stocks CSV", safe_csv(inventory_table(p, sid)), "materiapilot-stocks.csv", "text/csv", width="stretch", on_click="ignore")
        b.download_button("↓ Classeur Excel", excel_bytes(p, sid, forecast), "materiapilot-chantier.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch", on_click="ignore")
        c.download_button("↓ Rapport PDF", report_pdf(p, sid, forecast), "materiapilot-chantier.pdf", "application/pdf", width="stretch", on_click="ignore")
        table_ui(inventory_table(p, sid), "Données de stock du chantier incluses dans les rapports.")
    with backup:
        st.subheader("Sauvegarde complète du registre")
        st.caption("Le JSON contient tous les chantiers, commandes, profils et mouvements. Une sauvegarde de démonstration reste une démonstration ; elle ne peut pas remplacer les données réelles.")
        st.download_button("Télécharger la sauvegarde JSON", json.dumps(p, ensure_ascii=False, indent=2).encode(), "materiapilot-demo.json" if p["demo"] else "materiapilot-registre.json", "application/json", on_click="ignore")
        with st.expander("Restaurer une sauvegarde complète"):
            upload = st.file_uploader("Sauvegarde MatériaPilot JSON", type="json", key=ui_key("restore_upload"))
            if upload:
                try:
                    if upload.size > 5_000_000:
                        raise ValueError("Maximum 5 Mo par sauvegarde.")
                    trial = validate(json.loads(upload.getvalue().decode("utf-8-sig")))
                    if trial["demo"] != p["demo"]:
                        raise ValueError("Le mode de la sauvegarde doit correspondre au mode actuel (fictif ou réel).")
                    st.info(f"Sauvegarde valide : {len(trial['sites'])} chantier(s), {len(trial['events'])} mouvement(s).")
                    confirmed = st.checkbox("Je remplace le registre actuel de ce mode par cette sauvegarde complète.", key=ui_key("restore_confirm"))
                    if st.button("Restaurer le registre", disabled=not confirmed):
                        commit(trial, "Sauvegarde restaurée.")
                except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                    st.error(str(exc))
        st.markdown("**Enregistrement réel** : SQLite sur le serveur, hors du dossier de code. Les visiteurs du même serveur utilisent le même registre. Une révision empêche une session ancienne d’écraser le travail d’une autre. Le disque d’un hébergement temporaire peut être réinitialisé : conserve les sauvegardes JSON.")
    with guide:
        st.markdown("""### Commencer en cinq étapes
1. Choisis **Mes données** dans le menu. Le registre réel démarre sans stock et sans prix.
2. Dans **Référentiel**, crée tes chantiers, zones et fournisseurs. Adapte les références et leurs unités avant le premier bon.
3. Dans **Mouvements**, saisis le stock initial constaté, puis les consommations, retours, pertes et transferts.
4. Dans **Commandes**, crée un bon d’achat et enregistre chaque livraison partielle. Seul l’accepté entre au dépôt.
5. Fais un **Inventaire**, prépare les **Profils** de demande, puis lance une simulation. Sauvegarde ensuite le registre.

### Test 1 · Retrouver la démonstration
Recharge la démonstration fictive, puis sélectionne Résidence Horizon.
- Ciment : **110 sacs**, dont **94 au dépôt DEP** et **16 en BLOC-A**.
- Sable : **17 m³** ; gravier : **26 m³** ; bois : **5 m³** ; acier : **900 kg**.
- BC-001 : 50 sacs commandés, 20 acceptés, 3 refusés : **30 sacs restent dus**.
- Valeur connue du stock de ce chantier : **2 198 000 FCFA**, avec les prix fictifs fournis.

### Test 2 · Mouvement et correction
Dans la démonstration, consomme **4 sacs de CIM dans BLOC-A**, avec un nouveau bon. Le stock BLOC-A passe de 16 à **12 sacs**, le total du chantier à **106 sacs**. Annule ce mouvement dans le journal avec un motif : les quantités reviennent à **16 et 110 sacs**. L’original et sa contre-écriture restent visibles.

### Test 3 · Réception partielle
Réceptionne **10 sacs acceptés et 2 refusés** sur BC-001, avec un nouveau bon. Le dépôt DEP passe de 94 à **104 sacs** ; le total ciment à **120 sacs** ; le reste dû de la commande devient **20 sacs**. Les quantités refusées cumulées atteignent **5 sacs**. Annule la réception pour revenir à la situation précédente.

### Test 4 · Comptage physique
À partir de la démonstration initiale, compte **92 sacs** dans DEP. L’écart est **−2 sacs** ; le total ciment du chantier passe à **108 sacs**. Le comptage ne modifie pas le stock d’un autre dépôt.

### Test 5 · Une prévision vérifiable à la calculatrice
Dans Mes données, crée un chantier dédié au test. Enregistre **100 sacs** de ciment en stock initial. Active seulement son profil : demande **Constante, 10 sacs / jour actif**. Lance 500 scénarios, 30 jours, **les sept jours actifs**, interruption 0 %, sans commande. Le premier manque est à **J11**, avec un risque de **100 %** sur 30 jours. À J10 le stock vaut zéro, mais toute la demande a été servie : 100 ÷ 10 = 10 journées complètes.

Ajoute ensuite une commande de **50 sacs à J5**, retard maximal 0 jour. Avec les mêmes réglages, le premier manque devient **J16** : (100 + 50) ÷ 10 = 15 journées complètes. Le levier hypothétique de 50 sacs à J5 permet le même test sans créer de commande réelle.

### Test 6 · Calendrier et horaire de réception
Si aujourd’hui est un vendredi, active seulement lundi–vendredi, avec stock 0 et demande constante 10 : le premier manque survient **J3, lundi**. Avec 90 sacs et une livraison de 100 sacs à J10, les sept jours actifs : en début de journée, aucun manque avant J20 ; en fin de journée, un manque se produit à **J10**, même si du stock est présent après la réception.

### Test 7 · Import et exports
Télécharge le modèle CSV, remplis un bon fictif avec tes codes réels et une référence nouvelle. Un lot contenant une sortie supérieure au stock doit être entièrement refusé. Télécharge le CSV, le classeur Excel et le PDF ; compare leurs stocks au tableau de bord. Sauvegarde en JSON avant tout essai de restauration.

### Bien interpréter
Une alerte de seuil est une règle choisie, pas une probabilité. Les retours et les pertes sont des déclarations ; le comptage physique rapproche le registre du terrain. Les prix sont saisis ou fictifs ; les stocks de matériaux différents ne sont jamais additionnés en une quantité physique globale. Cette application est un outil pédagogique de suivi, sans validation réglementaire.
""")


def main():
    global PAGES
    st.set_page_config(page_title="MatériaPilot · Suivi des matériaux", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    if "mp_project" not in st.session_state:
        st.session_state.update(mp_project=demo_project(), mp_revision=0, mp_mode="Démonstration fictive", mp_site="S1")
    with st.sidebar:
        st.markdown("### ◈ MatériaPilot")
        st.caption("REGISTRE & APPROVISIONNEMENTS")
        st.selectbox("Espace de travail", ["Démonstration fictive", "Mes données"], key="mp_mode", on_change=context_changed)
        if "mp_load_error" in st.session_state:
            st.error("Impossible de lire le registre réel : " + st.session_state["mp_load_error"])
            st.button("Réessayer la lecture", on_click=context_changed)
            st.stop()
        p = current()
        if st.session_state.get("mp_site") not in {s["id"] for s in p["sites"]}:
            st.session_state["mp_site"] = p["sites"][0]["id"]
        st.selectbox("Chantier actif", [s["id"] for s in p["sites"]], format_func=lambda x: lookup(p, "sites", x)["name"], key="mp_site")
        st.button("↻ Recharger l’exemple" if p["demo"] else "↻ Actualiser le registre", width="stretch", on_click=context_changed)
        st.divider()
    PAGES = [
        st.Page(cockpit_page, title="Cockpit", icon=":material/dashboard:", default=True),
        st.Page(catalogue_page, title="Référentiel", icon=":material/category:"),
        st.Page(movement_page, title="Mouvements", icon=":material/swap_horiz:"),
        st.Page(orders_page, title="Commandes", icon=":material/local_shipping:"),
        st.Page(inventory_page, title="Inventaire & analyse", icon=":material/fact_check:"),
        st.Page(forecast_page, title="Prévisions", icon=":material/query_stats:"),
        st.Page(reports_page, title="Rapports & guide", icon=":material/description:"),
    ]
    pg = st.navigation(PAGES)
    mode = "EXEMPLE FICTIF" if current()["demo"] else "DONNÉES DU REGISTRE"
    st.markdown(f'<div class="mast"><div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">MatériaPilot</div><div class="eyebrow">La matière au bon endroit</div></div></div><div class="status-pill">● {mode}</div></div>', unsafe_allow_html=True)
    if "mp_message" in st.session_state:
        st.toast(st.session_state.pop("mp_message"), icon="✅")
    pg.run()
    with st.sidebar:
        st.divider()
        st.caption("v1.0 · Français · FCFA\n\nDonnées fictives et registre réel séparés. Le registre réel est partagé sur le serveur.")


def report_pdf(p, site, forecast=None):
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    fonts = Path(rl_config.__file__).parent / "fonts"
    for name, file in [("MPVera", "Vera.ttf"), ("MPVeraB", "VeraBd.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("MPVera", normal="MPVera", bold="MPVeraB", italic="MPVera", boldItalic="MPVeraB")
    styles = {
        "title": ParagraphStyle("title", fontName="MPVeraB", fontSize=23, leading=29, textColor=colors.HexColor("#14283F"), spaceAfter=13),
        "h": ParagraphStyle("h", fontName="MPVeraB", fontSize=13, leading=18, spaceBefore=15, spaceAfter=8, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="MPVera", fontSize=9, leading=14, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="MPVera", fontSize=8, leading=12, alignment=TA_LEFT),
    }
    para = lambda s, style="body": Paragraph(html.escape(str(s)).replace("\n", "<br/>"), styles[style])
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=(21 * cm, 29.7 * cm), leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm, title=APP + " — rapport matériaux", author=APP)
    story = [para(APP + " / Rapport matériaux", "title"), para(lookup(p, "sites", site)["name"]), para(("DONNÉES FICTIVES DE DÉMONSTRATION" if p["demo"] else "REGISTRE UTILISATEUR") + " — Édité le " + today().isoformat()), para("Stock théorique issu des bons enregistrés. Les valeurs monétaires utilisent des prix de référence saisis ; elles ne constituent pas une comptabilité certifiée.")]
    def table(headers, rows, widths):
        content = [[para(x, "cell") for x in headers]] + [[para(x, "cell") for x in r] for r in rows]
        t = Table(content, colWidths=[x * cm for x in widths], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FA")]), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#8AA0BF"))]))
        story.append(t)
    stock = site_stock(p, site)
    story.append(para("01 / Stocks par matériau", "h"))
    table(["Matériau", "Stock", "Unité", "Valeur de référence"], [[m["name"], fmt(stock[m["id"]]), m["unit"], "Prix non renseigné" if m["price"] is None else money(stock[m["id"]] * m["price"])] for m in p["materials"]], [4.5, 3, 2, 8.1])
    inv = inventory_table(p, site)
    story.append(para("02 / Répartition dans les zones", "h"))
    table(["Zone", "Matériau", "Stock", "Unité"], [[r["Zone"], r["Matériau"], fmt(r["Stock"]), r["Unité"]] for _, r in inv.iterrows()], [4, 6, 4, 3.6])
    orders = orders_table(p, site)
    story.append(para("03 / Commandes et réceptions", "h"))
    if orders.empty:
        story.append(para("Aucune commande enregistrée."))
    else:
        table(["Commande", "Matériau", "Reste dû", "Prévu / état"], [[r["Commande"], r["Matériau"], fmt(r["Reste dû"]) + " " + r["Unité"], r["Prévu le"] + " / " + r["État"]] for _, r in orders.iterrows()], [3.5, 4, 4, 6.1])
    story.append(para("04 / Pertes et écarts d’inventaire", "h"))
    rows = []
    for m in p["materials"]:
        events = [e for e in effective_events(p) if e["material_id"] == m["id"] and event_site(p, e, site)]
        rows.append([m["name"], fmt(sum(e["qty"] for e in events if e["type"] == "Perte / casse")), fmt(sum(e["qty"] for e in events if e["type"] == "Inventaire")), m["unit"]])
    table(["Matériau", "Pertes déclarées", "Écart net comptages", "Unité"], rows, [5, 4.5, 5.5, 2.6])
    story.append(para("05 / Prévisions de rupture", "h"))
    if forecast:
        cfg = forecast["cfg"]
        story.append(para(f"Simulation du {cfg['asof']} : {forecast['n']} scénarios, {forecast['h']} jours calendaires. Livraison : {cfg['timing'].lower()}. Interruption commune : {fmt(cfg['interruption'])} %. Les profils et délais sont des hypothèses utilisateur, non des fréquences observées."))
        sf, _ = forecast_tables(forecast)
        table(["Matériau", "Risque sur l’horizon", "Jour médian du premier manque"], [[r["Matériau"], fmt(100 * r["P rupture"]) + " %", r["Premier risque médiane"]] for _, r in sf.iterrows()], [5.5, 5.5, 6.6])
        if forecast["ignored"]:
            story.append(para("Commandes en retard exclues : " + ", ".join(forecast["ignored"])))
    else:
        story.append(para("Aucune simulation à jour pour ce chantier. Un stock nul n’est une rupture que lorsqu’une demande ne peut pas être servie."))
    story += [para("Méthode et limites", "h"), para("Stock = stock initial + réceptions acceptées + retours + transferts entrants − consommations − pertes − transferts sortants + ajustements. Les contre-écritures inversent les mouvements annulés. Un refus de livraison n’entre jamais en stock et reste dû tant que la commande demeure ouverte. Le registre est chronologique ; une référence documentaire permet le rapprochement."), para("Les prévisions additionnent les zones du chantier sans modéliser le temps de transfert. Les demandes non servies ne sont pas reportées. La loi de Poisson compte des sacs ou pièces ; la loi triangulaire représente une plage supposée pour une quantité continue. Les probabilités dépendent des hypothèses. L’intervalle de Monte-Carlo ne mesure pas leur justesse. Aucun résultat ne valide un ouvrage ou une conformité réglementaire.")]
    def footer(canvas, document):
        canvas.setFont("MPVera", 8)
        canvas.setFillColor(colors.HexColor("#53657D"))
        canvas.drawString(1.7 * cm, 1.05 * cm, APP + " • Suivi des matériaux • Outil pédagogique")
        canvas.drawRightString(19.3 * cm, 1.05 * cm, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


if __name__ == "__main__":
    main()

