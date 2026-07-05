"""Turn decoded message text into structured, mappable records.

Aero position reports and STD-C ship reports carry latitude/longitude. This
module extracts them and exports to formats that mapping tools read directly
(CSV, GeoJSON, KML).
"""
from __future__ import annotations

import json
import re

# N/S dd(.d)  ...  E/W ddd(.d)
_POS = re.compile(r"([NS])\s*(\d{1,2}(?:\.\d+)?)\D{0,6}([EW])\s*(\d{1,3}(?:\.\d+)?)")
_AES = re.compile(r"AES\s+([0-9A-Fa-f]{4,6})")
_FLT = re.compile(r"FLT\s+([A-Z0-9]+)")
_SHIP = re.compile(r"SHIP\s+([A-Z0-9 ]+?)\s{2,}")


def classify(text: str) -> str:
    if text.startswith("AES"):
        return "Aero"
    if text.startswith("STD-C"):
        return "STD-C"
    return "-"


def parse_position(text: str):
    """Return (lat, lon) in decimal degrees, or None."""
    m = _POS.search(text)
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    lat = float(lat) * (1 if ns == "N" else -1)
    lon = float(lon) * (1 if ew == "E" else -1)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def identifier(text: str) -> str:
    for rx in (_FLT, _AES, _SHIP):
        m = rx.search(text)
        if m:
            return m.group(1).strip()
    return ""


def make_record(text: str, ts: str) -> dict:
    pos = parse_position(text)
    return {
        "time": ts,
        "kind": classify(text),
        "id": identifier(text),
        "lat": pos[0] if pos else None,
        "lon": pos[1] if pos else None,
        "text": text,
    }


def to_csv(records) -> str:
    out = ["time,kind,id,lat,lon,text"]
    for r in records:
        text = r["text"].replace('"', "'")
        lat = "" if r["lat"] is None else f"{r['lat']:.5f}"
        lon = "" if r["lon"] is None else f"{r['lon']:.5f}"
        out.append(f'{r["time"]},{r["kind"]},{r["id"]},{lat},{lon},"{text}"')
    return "\n".join(out) + "\n"


def to_geojson(records) -> str:
    feats = []
    for r in records:
        if r["lat"] is None:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {k: r[k] for k in ("time", "kind", "id", "text")},
        })
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=2)


def to_kml(records) -> str:
    marks = []
    for r in records:
        if r["lat"] is None:
            continue
        name = r["id"] or r["kind"]
        marks.append(
            f"  <Placemark><name>{name}</name>"
            f"<description>{r['time']} {r['text']}</description>"
            f"<Point><coordinates>{r['lon']},{r['lat']}</coordinates></Point>"
            f"</Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
            + "\n".join(marks) + "\n</Document></kml>\n")
