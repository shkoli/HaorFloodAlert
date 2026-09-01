/**
 * 01_sentinel1_flood_detection.js
 * ================================
 * Sentinel-1 SAR-based flood inundation mapping for Sunamganj Haor, Bangladesh.
 * Method: Otsu bi-modal thresholding on change detection image (pre - post).
 *
 * How to use:
 *   1. Paste this script into Google Earth Engine Code Editor (code.earthengine.google.com)
 *   2. Set PRE_START/PRE_END to a dry season reference period
 *   3. Set FLOOD_START/FLOOD_END to the flood event dates
 *   4. Click Run → flood extent layer appears on the map
 *
 * Reference: Sentinel-1 SAR flood mapping methodology (Manjusree et al., 2012;
 *            Twele et al., 2016). Haor-specific threshold from BRRI field surveys.
 *
 * Author: Salma Hoque Talukdar Koli
 * Institution: RTM Al-Kabir Technical University, CSE
 * Thesis: Flood Prediction in the Haor Regions of Bangladesh Using ML and Satellite Data
 */

// ── Study area: Sunamganj Haor, Bangladesh ──────────────────────────────────
var HAOR = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00]);

// ── Date windows: adjust these for your flood event ─────────────────────────
var PRE_START   = '2024-01-01';   // Dry season reference (Jan–Feb = dry)
var PRE_END     = '2024-02-28';
var FLOOD_START = '2024-04-05';   // Flash flood event dates
var FLOOD_END   = '2024-04-25';

// ── 1. Load Sentinel-1 GRD collections ──────────────────────────────────────
var s1_pre = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(HAOR)
  .filterDate(PRE_START, PRE_END)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select(['VV'])
  .median();

var s1_flood = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(HAOR)
  .filterDate(FLOOD_START, FLOOD_END)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select(['VV'])
  .median();

print('Pre-flood S1 images count:',
  ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(HAOR).filterDate(PRE_START, PRE_END).size());

// ── 2. Speckle filtering (Lee filter approximation via focal_mean) ───────────
var s1_pre_smooth   = s1_pre.focal_mean(radius=50, kernelType='circle', units='meters');
var s1_flood_smooth = s1_flood.focal_mean(radius=50, kernelType='circle', units='meters');

// ── 3. Change detection: difference image (dB scale) ────────────────────────
// Flood signal: VV drops by >3 dB from dry baseline (open water has low backscatter)
var vv_change = s1_pre_smooth.subtract(s1_flood_smooth).rename('VV_change');

// ── 4. Otsu automatic threshold on the change image ─────────────────────────
// The threshold separates "changed = flooded" from "unchanged = dry land"
var histogram = vv_change.reduceRegion({
  reducer: ee.Reducer.histogram({maxBuckets: 255, minBucketWidth: 0.1}),
  geometry: HAOR,
  scale: 10,
  maxPixels: 1e9
});

// Otsu implementation: maximize inter-class variance
function otsuThreshold(hist) {
  var counts = ee.Array(ee.Dictionary(hist).get('histogram'));
  var means  = ee.Array(ee.Dictionary(hist).get('bucketMeans'));
  var size   = means.length().get([0]);
  var total  = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum    = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean   = sum.divide(total);
  var indices = ee.List.sequence(1, size);
  var bss = indices.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount  = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans  = means.slice(0, 0, i);
    var aMean   = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0])
                        .divide(aCount);
    var bCount  = total.subtract(aCount);
    var bMean   = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
           .add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return means.sort(bss).get([-1]);
}

var threshold = otsuThreshold(histogram.get('VV_change'));
print('Otsu threshold (dB change):', threshold);

// ── 5. Apply threshold to create flood mask ──────────────────────────────────
var flood_mask = vv_change.gt(ee.Number(threshold));

// Remove permanent water bodies (pre-flood open water)
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality');
var permanent_water = jrc.gte(10);   // water present ≥10 months/year
var flood_only = flood_mask.where(permanent_water, 0).selfMask();

// ── 6. Calculate flood area statistics ──────────────────────────────────────
var flood_area = flood_only.multiply(ee.Image.pixelArea()).divide(1e6);  // km²
var area_stats = flood_area.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: HAOR,
  scale: 10,
  maxPixels: 1e9
});
print('Estimated flood area (km²):', area_stats);

// ── 7. Visualise ─────────────────────────────────────────────────────────────
Map.centerObject(HAOR, 11);
Map.addLayer(HAOR, {color: 'yellow'}, 'Study Area — Sunamganj Haor');

Map.addLayer(s1_pre_smooth, {min: -25, max: 0, palette: ['black','white']},
             'S1 VV — Pre-flood (dry)');

Map.addLayer(s1_flood_smooth, {min: -25, max: 0, palette: ['black','white']},
             'S1 VV — At-flood');

Map.addLayer(vv_change, {min: -5, max: 10, palette: ['blue','white','red']},
             'VV Change (dB) — positive = flooded');

Map.addLayer(flood_only, {palette: ['0000FF'], opacity: 0.7},
             'Flood Extent (Otsu threshold)');

Map.addLayer(permanent_water.selfMask(), {palette: ['00FFFF'], opacity: 0.5},
             'Permanent Water (JRC, excluded)');

// ── 8. Export flood mask to Google Drive ────────────────────────────────────
Export.image.toDrive({
  image: flood_only,
  description: 'Haor_flood_mask_' + FLOOD_START,
  folder: 'HaorFloodAlert',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

// ── Notes for thesis ─────────────────────────────────────────────────────────
// 1. Otsu threshold automatically adapts to each flood event — no manual tuning
// 2. Threshold values typically 3–6 dB for haor flash floods (Sentinel-1 C-band)
// 3. This script is reproducible: any researcher can run it on GEE with the same dates
// 4. JRC Global Surface Water is used to exclude permanent water bodies
// 5. Output can be imported into QGIS for upazila-level area calculation
