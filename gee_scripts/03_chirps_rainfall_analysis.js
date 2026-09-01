/**
 * 03_chirps_rainfall_analysis.js
 * ================================
 * CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) seasonal
 * rainfall analysis for Sunamganj Haor, Bangladesh.
 *
 * Outputs:
 *   1. Annual rainfall time series (2010–2024)
 *   2. Monthly climatology (seasonal pattern) for Sunamganj
 *   3. Pre-monsoon cumulative rainfall (Feb–May) — critical for boro flash flood
 *   4. Rainfall anomaly map (current vs historical average)
 *
 * GEE Asset ID: UCSB-CHG/CHIRPS/DAILY  (note: hyphen after UCSB, not slash)
 * Resolution: 5566m (~0.05°), daily data from 1981 to present
 *
 * Author: Salma Hoque Talukdar Koli
 * Institution: RTM Al-Kabir Technical University, CSE
 */

// ── Study area ────────────────────────────────────────────────────────────────
var HAOR        = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00]);
var SUNAMGANJ   = ee.Geometry.Point([91.45, 24.87]);

// ── Parameters ─────────────────────────────────────────────────────────────────
var HIST_START  = '2010-01-01';
var HIST_END    = '2024-12-31';
var TARGET_YEAR = 2024;         // year to analyse in detail

// ── 1. Annual total rainfall — time series 2010–2024 ──────────────────────────
var years = ee.List.sequence(2010, TARGET_YEAR);

var annual_totals = years.map(function(yr) {
  var start = ee.Date.fromYMD(yr, 1, 1);
  var end   = ee.Date.fromYMD(yr, 12, 31);
  var total = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(start, end)
    .sum()
    .select('precipitation');
  var val = total.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: HAOR,
    scale: 5566,
    maxPixels: 1e6
  }).get('precipitation');
  return ee.Feature(null, {year: yr, rainfall_mm: val});
});
print('Annual rainfall (mm) 2010–2024:', ee.FeatureCollection(annual_totals));

// ── 2. Monthly climatology — 2010–2023 average per month ─────────────────────
var months = ee.List.sequence(1, 12);
var month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

var monthly_clim = months.map(function(m) {
  var monthly = years.map(function(yr) {
    var start = ee.Date.fromYMD(yr, m, 1);
    var end   = start.advance(1, 'month');
    return ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
      .filterDate(start, end).sum().select('precipitation');
  });
  var clim_val = ee.ImageCollection(monthly).mean()
    .reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: HAOR,
      scale: 5566,
      maxPixels: 1e6
    }).get('precipitation');
  return ee.Feature(null, {month: m, mean_rainfall_mm: clim_val});
});
print('Monthly climatology (mm/month, 2010–2023):', ee.FeatureCollection(monthly_clim));

// ── 3. Pre-monsoon cumulative rainfall (Feb–May) — boro flash flood driver ────
// This 4-month window is when boro rice is most vulnerable and flash floods occur
var premonsoon = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(HAOR)
  .filter(ee.Filter.calendarRange(TARGET_YEAR, TARGET_YEAR, 'year'))
  .filter(ee.Filter.calendarRange(2, 5, 'month'))
  .sum()
  .select('precipitation');

var premonsoon_mm = premonsoon.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: HAOR,
  scale: 5566,
  maxPixels: 1e6
});
print('Pre-monsoon total rainfall Feb–May ' + TARGET_YEAR + ' (mm):', premonsoon_mm);

// ── 4. 7-day rolling cumulative rainfall (for ML feature replication) ─────────
// This is the "rainfall" feature used in the ML model — 7-day window before event
var FLOOD_DATE = ee.Date('2024-04-15');   // change to your event date
var seven_day  = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(FLOOD_DATE.advance(-7, 'day'), FLOOD_DATE)
  .sum()
  .select('precipitation');

var seven_day_mm = seven_day.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: HAOR,
  scale: 5566,
  maxPixels: 1e6
});
print('7-day rainfall before ' + FLOOD_DATE.format('YYYY-MM-dd').getInfo() + ' (mm):', seven_day_mm);

// ── 5. Rainfall anomaly map (TARGET_YEAR vs historical mean) ──────────────────
var hist_mean = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(HAOR)
  .filterDate(HIST_START, (TARGET_YEAR - 1) + '-12-31')
  .filter(ee.Filter.calendarRange(2, 5, 'month'))
  .mean()
  .multiply(120);   // ×120 days ≈ 4 months total

var current = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(HAOR)
  .filter(ee.Filter.calendarRange(TARGET_YEAR, TARGET_YEAR, 'year'))
  .filter(ee.Filter.calendarRange(2, 5, 'month'))
  .sum();

var anomaly = current.subtract(hist_mean).rename('anomaly_mm');

// ── 6. Visualise ──────────────────────────────────────────────────────────────
Map.centerObject(HAOR, 10);
Map.addLayer(HAOR, {color: 'yellow'}, 'Sunamganj Haor');

Map.addLayer(premonsoon, {
  min: 0, max: 2000,
  palette: ['white', '#90EE90', '#0000FF', '#8B0000']
}, 'Pre-monsoon rainfall (Feb–May ' + TARGET_YEAR + ')');

Map.addLayer(anomaly, {
  min: -500, max: 500,
  palette: ['#FF4B4B', 'white', '#0000FF']
}, 'Rainfall anomaly (vs 2010–' + (TARGET_YEAR-1) + ' mean) — blue=wetter');

Map.addLayer(seven_day, {
  min: 0, max: 400,
  palette: ['white', '#90EE90', '#0000FF']
}, '7-day cumulative rainfall before ' + '2024-04-15');

// ── 7. Export ─────────────────────────────────────────────────────────────────
Export.image.toDrive({
  image: premonsoon,
  description: 'CHIRPS_premonsoon_' + TARGET_YEAR,
  folder: 'HaorFloodAlert',
  region: HAOR,
  scale: 5566,
  maxPixels: 1e9
});

Export.image.toDrive({
  image: anomaly,
  description: 'CHIRPS_anomaly_FebMay_' + TARGET_YEAR,
  folder: 'HaorFloodAlert',
  region: HAOR,
  scale: 5566,
  maxPixels: 1e9
});

// ── Notes for thesis ──────────────────────────────────────────────────────────
// 1. CHIRPS GEE asset ID: "UCSB-CHG/CHIRPS/DAILY" (not "UCSB/CHIRPS/DAILY")
//    The hyphen is required — incorrect ID causes silent failure returning null
// 2. CHIRPS resolution (5566m) is coarser than Sentinel — acceptable for haor
//    rainfall which is spatially homogeneous (flat terrain, small haor)
// 3. The 7-day cumulative window matches the ML model's "rainfall" feature
//    (see config.py DEFAULTS and collect_real_data_v3.py)
// 4. Pre-monsoon rainfall anomaly correlates strongly with haor flash flood risk
//    (Mondal et al., 2021 — correlation r=0.82 for Sunamganj district)
// 5. For real-time monitoring, replace sum().select() with filterDate(today-7, today)
