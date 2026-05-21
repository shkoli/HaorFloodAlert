/**
 * 02_sentinel2_ndwi_mapping.js
 * ==============================
 * Sentinel-2 Normalized Difference Water Index (NDWI) mapping for Sunamganj Haor.
 * NDWI = (Green - NIR) / (Green + NIR) = (B3 - B8) / (B3 + B8)
 * Positive NDWI → water surface. Negative → dry land / vegetation.
 *
 * Also generates NDVI to map Boro rice growth stage (used in crop damage estimation).
 * NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
 *
 * How to use:
 *   1. Paste into GEE Code Editor
 *   2. Adjust DATE_START / DATE_END for your target period
 *   3. Run → NDWI and NDVI layers appear on the map
 *
 * Reference: McFeeters (1996) NDWI; Haboudane et al. (2004) for rice NDVI stages.
 *
 * Author: Salma Hoque Talukdar Koli
 * Institution: RTM Al-Kabir Technical University, CSE
 */

// ── Study area ────────────────────────────────────────────────────────────────
var HAOR = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00]);

// ── Date window: choose a period with <50% cloud cover ───────────────────────
// For monsoon season use longer window (30+ days) due to persistent cloud cover
var DATE_START = '2024-03-01';
var DATE_END   = '2024-04-30';

// ── 1. Load Sentinel-2 SR (surface reflectance, cloud-masked) ────────────────
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // SCL classes to keep: 4=vegetation, 5=bare soil, 6=water, 11=snow
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11));
  return image.updateMask(mask).divide(10000);  // scale to 0–1
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(HAOR)
  .filterDate(DATE_START, DATE_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2clouds)
  .median();

print('Number of S2 images in window:',
  ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(HAOR).filterDate(DATE_START, DATE_END)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)).size()
);

// ── 2. Calculate NDWI (water detection) ─────────────────────────────────────
var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var water_mask = ndwi.gt(0.0).selfMask();   // NDWI > 0 = water surface

// ── 3. Calculate NDVI (vegetation/rice growth stage) ────────────────────────
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Boro rice growth stage classification (BRRI 2022):
// NDVI 0.20–0.45 → vegetative/booting, 0.45–0.70 → heading/grain fill, >0.70 → mature
var rice_stage = ndvi
  .where(ndvi.lt(0.20), 0)   // bare soil / young seedling
  .where(ndvi.gte(0.20).and(ndvi.lt(0.45)), 1)   // vegetative
  .where(ndvi.gte(0.45).and(ndvi.lt(0.70)), 2)   // booting / heading
  .where(ndvi.gte(0.70), 3)                       // grain fill / mature
  .rename('rice_stage');

// ── 4. Water area statistics ─────────────────────────────────────────────────
var ndwi_stats = ndwi.reduceRegion({
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.min(),
    sharedInputs: true
  }).combine({
    reducer2: ee.Reducer.max(),
    sharedInputs: true
  }),
  geometry: HAOR,
  scale: 10,
  maxPixels: 1e9
});
print('NDWI statistics (mean/min/max):', ndwi_stats);

var water_area_km2 = water_mask.multiply(ee.Image.pixelArea()).divide(1e6)
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: HAOR, scale: 10, maxPixels: 1e9});
print('Open water area (km²):', water_area_km2);

// ── 5. Visualise ─────────────────────────────────────────────────────────────
Map.centerObject(HAOR, 11);
Map.addLayer(HAOR, {color: 'yellow'}, 'Study Area');

// True colour composite
Map.addLayer(s2, {bands: ['B4','B3','B2'], min: 0, max: 0.3},
             'Sentinel-2 True Colour');

// False colour (NIR-R-G) — water appears dark, rice appears red
Map.addLayer(s2, {bands: ['B8','B4','B3'], min: 0, max: 0.4},
             'Sentinel-2 False Colour (NIR-R-G)');

// NDWI — blue = water, red = dry
Map.addLayer(ndwi, {min: -0.5, max: 0.5, palette: ['red','white','blue']},
             'NDWI (water index)');

// Water mask
Map.addLayer(water_mask, {palette: ['0000FF'], opacity: 0.6},
             'Water Extent (NDWI > 0)');

// NDVI — dark red = mature rice, green = vegetative
Map.addLayer(ndvi, {min: 0, max: 1, palette: ['white','yellow','green','darkgreen']},
             'NDVI (rice vegetation)');

// Rice stage classification
Map.addLayer(rice_stage, {
  min: 0, max: 3,
  palette: ['grey','#90EE90','#FFD700','#FF8C00']
}, 'Boro Rice Stage (0=bare, 1=veg, 2=heading, 3=mature)');

// ── 6. Export outputs ─────────────────────────────────────────────────────────
Export.image.toDrive({
  image: ndwi,
  description: 'Haor_NDWI_' + DATE_START,
  folder: 'HaorFloodAlert',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9
});

Export.image.toDrive({
  image: rice_stage,
  description: 'Haor_rice_stage_' + DATE_START,
  folder: 'HaorFloodAlert',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9
});

// ── Notes for thesis ──────────────────────────────────────────────────────────
// 1. NDWI > 0.3 is a stricter threshold for permanent open water in haors
// 2. NDWI 0.0–0.3 often indicates shallow flooding or saturated soil
// 3. Sentinel-2 cannot penetrate clouds — use Sentinel-1 for monsoon flood mapping
// 4. NDVI-based rice stage classification validates the phenological calendar
//    used in the crop damage estimation module (6_CropDamage.py)
// 5. Compare NDWI water extent with Sentinel-1 flood mask for cross-validation
