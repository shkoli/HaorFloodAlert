"""
generate_susceptibility_map.py
===============================
Builds a pixel-level flood susceptibility map for the Sunamganj Haor
region using the trained RF model exported to Google Earth Engine (GEE).

Method (two-step):
  Step 1 — Feature stack (GEE): Compose a multi-band image from
           Sentinel-1, CHIRPS, ERA5-Land, SRTM.
  Step 2 — Classify with sklearn model: Because the RF is a Python
           model, we pull the feature bands as a numpy array and
           run predict_proba locally, then re-upload as an asset
           OR export a classified raster.

Alternative (fully in GEE): if you want a pure GEE classifier you can
train a ee.Classifier.smileRandomForest directly in GEE using the
labeled points — see the commented section at the bottom.

Run:
    python generate_susceptibility_map.py --mode local
    python generate_susceptibility_map.py --mode gee_export
"""

import argparse
import json
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd

ROOT       = Path(__file__).parent.resolve()
MODELS_DIR = ROOT / "models"
RESULTS_DIR= ROOT / "results"
GEE_PROJECT = "flood-haor-project"

# Sunamganj study area — wider bbox for susceptibility map
STUDY_BBOX = [91.20, 24.65, 91.70, 25.10]

FEATURE_NAMES = [
    "VV", "VH", "vv_vh_ratio",
    "rainfall", "soil_moisture",
    "temp", "wind", "slope",
    "forecast_rain_next_12h",
]

# Flood season: April-June (pre-monsoon flash flood window)
FLOOD_START = "2024-04-01"
FLOOD_END   = "2024-06-15"


def init_gee():
    ee.Initialize(project=GEE_PROJECT)
    return ee.Geometry.Rectangle(STUDY_BBOX)


def build_feature_image(region):
    """
    Compose a multi-band feature image at 30m resolution.
    Returns ee.Image with bands matching FEATURE_NAMES.
    """

    # --- Sentinel-1 median (flood season) ---
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(FLOOD_START, FLOOD_END)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .median()
    )
    vv_vh_ratio = s1.select("VV").divide(s1.select("VH")).rename("vv_vh_ratio")

    # --- CHIRPS 7-day rainfall (end of flood_start + 7 days) ---
    chirps = (
        ee.ImageCollection("UCSB/CHIRPS/DAILY")
        .filterDate(FLOOD_START, FLOOD_END)
        .sum()
        .select("precipitation")
        .rename("rainfall")
    )

    # --- ERA5 soil moisture ---
    era5_soil = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(FLOOD_START, FLOOD_END)
        .select("volumetric_soil_water_layer_1")
        .mean()
        .multiply(100)
        .rename("soil_moisture")
    )

    # --- ERA5 temperature & wind (monthly mean proxy) ---
    era5_tw = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(FLOOD_START, FLOOD_END)
        .select(["temperature_2m", "u_component_of_wind_10m",
                 "v_component_of_wind_10m"])
        .mean()
    )
    temp = era5_tw.select("temperature_2m").subtract(273.15).rename("temp")
    wind = (
        era5_tw.select("u_component_of_wind_10m").pow(2)
        .add(era5_tw.select("v_component_of_wind_10m").pow(2))
        .sqrt()
        .multiply(3.6)  # m/s → km/h
        .rename("wind")
    )

    # --- SRTM slope ---
    dem   = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem).rename("slope")

    # --- Forecast rain proxy (use CHIRPS recent 3-day mean * 0.5) ---
    forecast = (
        ee.ImageCollection("UCSB/CHIRPS/DAILY")
        .filterDate("2024-06-10", FLOOD_END)
        .mean()
        .select("precipitation")
        .multiply(0.5)
        .rename("forecast_rain_next_12h")
    )

    feature_img = (
        s1.addBands(vv_vh_ratio)
          .addBands(chirps)
          .addBands(era5_soil)
          .addBands(temp)
          .addBands(wind)
          .addBands(slope)
          .addBands(forecast)
    ).select(FEATURE_NAMES)

    return feature_img


def local_susceptibility_map(region, scale: int = 500):
    """
    Pull feature values as a sampled grid, run sklearn RF prediction,
    save as CSV with lat/lon/probability for visualisation.
    """
    print("Building feature image ...")
    feat_img = build_feature_image(region)

    print(f"Sampling at {scale}m resolution ...")
    samples = feat_img.sample(
        region=region,
        scale=scale,
        numPixels=5000,
        seed=42,
        geometries=True,
    )

    features_list = samples.getInfo()["features"]
    if not features_list:
        print("No samples returned from GEE. Check date range and region.")
        return

    rows = []
    for f in features_list:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        row = {name: props.get(name, 0.0) for name in FEATURE_NAMES}
        row["lon"] = coords[0]
        row["lat"] = coords[1]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=FEATURE_NAMES)

    print(f"  {len(df)} sample points fetched")

    rf = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")

    probs_rf  = rf.predict_proba(df[FEATURE_NAMES])[:, 1]
    probs_xgb = xgb.predict_proba(df[FEATURE_NAMES])[:, 1]
    df["flood_prob"] = 0.55 * probs_rf + 0.45 * probs_xgb

    df["risk_class"] = pd.cut(
        df["flood_prob"],
        bins=[0, 0.40, 0.65, 0.85, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "EXTREME"],
    )

    out_path = RESULTS_DIR / "susceptibility_map.csv"
    df[["lat", "lon", "flood_prob", "risk_class"] + FEATURE_NAMES].to_csv(
        out_path, index=False
    )
    print(f"  Susceptibility map saved → {out_path}")

    risk_counts = df["risk_class"].value_counts()
    print("\n  Risk distribution:")
    for level in ["EXTREME", "HIGH", "MEDIUM", "LOW"]:
        count = risk_counts.get(level, 0)
        pct   = count / len(df) * 100
        print(f"    {level:<8}  {count:>5} pixels ({pct:.1f}%)")


def gee_export_susceptibility(region):
    """
    Train a GEE native classifier and export a classified raster to
    Google Drive for use in QGIS / ArcGIS.
    """
    print("Preparing GEE native classifier export ...")
    feat_img = build_feature_image(region)

    # Labeled training points (haor flood/dry events as ee.FeatureCollection)
    flood_pts = [
        ee.Feature(ee.Geometry.Point([91.45, 24.87]), {"flood": 1}),
        ee.Feature(ee.Geometry.Point([91.42, 24.92]), {"flood": 1}),
        ee.Feature(ee.Geometry.Point([91.50, 24.96]), {"flood": 1}),
        ee.Feature(ee.Geometry.Point([91.38, 24.80]), {"flood": 1}),
        ee.Feature(ee.Geometry.Point([91.46, 25.00]), {"flood": 1}),
    ]
    dry_pts = [
        ee.Feature(ee.Geometry.Point([91.40, 24.78]), {"flood": 0}),
        ee.Feature(ee.Geometry.Point([91.55, 24.82]), {"flood": 0}),
        ee.Feature(ee.Geometry.Point([91.35, 24.90]), {"flood": 0}),
        ee.Feature(ee.Geometry.Point([91.52, 24.98]), {"flood": 0}),
        ee.Feature(ee.Geometry.Point([91.44, 24.75]), {"flood": 0}),
    ]
    training_pts = ee.FeatureCollection(flood_pts + dry_pts)

    training = feat_img.sampleRegions(
        collection=training_pts,
        properties=["flood"],
        scale=30,
    )

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=200
    ).train(
        features=training,
        classProperty="flood",
        inputProperties=FEATURE_NAMES,
    )

    classified = feat_img.classify(classifier).rename("flood_susceptibility")

    # Export to Google Drive
    task = ee.batch.Export.image.toDrive(
        image=classified,
        description="Haor_Flood_Susceptibility_Map",
        folder="HaorFloodAlert",
        fileNamePrefix="sunamganj_susceptibility_2024",
        region=region,
        scale=30,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    print(f"  Export task started: {task.status()['description']}")
    print("  Check Google Drive → HaorFloodAlert folder when complete.")
    print("  Open the GeoTIFF in QGIS to visualise the susceptibility map.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["local", "gee_export"], default="local",
        help="'local' = sklearn prediction on sampled points (fast)\n"
             "'gee_export' = GEE native classifier → Google Drive GeoTIFF"
    )
    parser.add_argument("--scale", type=int, default=500,
                        help="Pixel scale in metres for local mode (default 500)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Sunamganj Haor — Flood Susceptibility Mapping")
    print("=" * 55)

    region = init_gee()

    if args.mode == "local":
        local_susceptibility_map(region, scale=args.scale)
    else:
        gee_export_susceptibility(region)


if __name__ == "__main__":
    main()
