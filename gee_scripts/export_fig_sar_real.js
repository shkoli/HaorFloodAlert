/**
 * export_fig_sar_real.js
 * =======================
 * Exports REAL Sentinel-1 VV backscatter composites for the Haor AOI for a
 * pre-flood reference window (Jan 2017) and a flood window (Apr 2017), for
 * use as the two panels of thesis Figure 3.4. Adapted from
 * gee_scripts/01_sentinel1_flood_detection.js, generalized to export the
 * raw VV composites themselves (not just a change-detection mask), because
 * the figure needs the actual backscatter panels, in dB, with a shared
 * colorbar -- not a flood mask.
 *
 * How to use:
 *   1. Paste this script into https://code.earthengine.google.com/
 *   2. Click Run
 *   3. Two Export.image.toDrive tasks appear in the "Tasks" tab on the
 *      right -- click "Run" on each to start the export to your Drive
 *      folder "HaorFloodAlert_Fig3_4"
 *   4. Download both GeoTIFFs from Drive once the exports finish
 *   5. Build the two-panel figure locally (e.g. with matplotlib/rasterio):
 *        - plot both rasters with the SAME vmin/vmax (suggest -25 to 0 dB,
 *          matching the visualization params below) so the colorbar is
 *          shared and comparable between panels
 *        - add a single shared colorbar labeled "VV backscatter (dB)"
 *        - target 300+ DPI, sized for an 8.9 cm (3.5 in) IEEE column width
 *          e.g. figsize=(3.5, 1.75) at dpi=300 for a two-panel side-by-side
 *          layout
 *   6. Save the final composite as sar_before_during.png
 *
 * Do NOT substitute simulated/synthetic data for this figure -- if the
 * exports are not available yet, leave the figure as a TODO in the draft
 * rather than fabricating backscatter values.
 */

// ── Study area: Sunamganj Haor, Bangladesh (same AOI as 01_sentinel1_flood_detection.js) ──
var HAOR = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00]);

// ── Date windows ──────────────────────────────────────────────────────────
// Pre-flood / dry-season reference: January 2017
var PRE_START = '2017-01-01';
var PRE_END   = '2017-01-31';
// Flood window: the 2017 Sunamganj pre-monsoon flash flood
var FLOOD_START = '2017-04-01';
var FLOOD_END   = '2017-04-20';

// ── Load Sentinel-1 GRD, VV, IW mode (same filters as 01_sentinel1_flood_detection.js) ──
function loadVV(start, end) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(HAOR)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select(['VV']);
}

var pre_coll   = loadVV(PRE_START, PRE_END);
var flood_coll = loadVV(FLOOD_START, FLOOD_END);

print('Pre-flood (Jan 2017) S1 image count:', pre_coll.size());
print('Flood (Apr 2017) S1 image count:', flood_coll.size());

var s1_pre   = pre_coll.median();
var s1_flood = flood_coll.median();

// ── Speckle filtering: same Lee-filter approximation as 01_sentinel1_flood_detection.js ──
var s1_pre_smooth   = s1_pre.focal_mean({radius: 50, kernelType: 'circle', units: 'meters'});
var s1_flood_smooth = s1_flood.focal_mean({radius: 50, kernelType: 'circle', units: 'meters'});

// ── Visualize with a SHARED dB range for a comparable colorbar ─────────────
var visParams = {min: -25, max: 0, palette: ['000000', '2b6cb0', 'ffffff']};
Map.centerObject(HAOR, 11);
Map.addLayer(s1_pre_smooth, visParams, 'VV -- Pre-Flood (Jan 2017)');
Map.addLayer(s1_flood_smooth, visParams, 'VV -- Flood (Apr 2017)');

// ── Export BOTH composites as GeoTIFF (real dB values, no colormap baked in) ──
Export.image.toDrive({
  image: s1_pre_smooth,
  description: 'Haor_VV_preflood_Jan2017',
  folder: 'HaorFloodAlert_Fig3_4',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: s1_flood_smooth,
  description: 'Haor_VV_flood_Apr2017',
  folder: 'HaorFloodAlert_Fig3_4',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

// ── Notes ───────────────────────────────────────────────────────────────────
// 1. If either image count above is 0, Sentinel-1 did not image this AOI in
//    that window -- widen PRE_START/PRE_END or FLOOD_START/FLOOD_END and
//    re-run before exporting.
// 2. This script exports the raw VV composites, in dB, not a flood mask --
//    use gee_scripts/01_sentinel1_flood_detection.js for the Otsu-threshold
//    flood-extent mask instead.
