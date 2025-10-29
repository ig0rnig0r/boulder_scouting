GitHub Copilot

Summary walkthrough — what the script does and where to look:

Load DGM (GeoTIFF)

File: dgm_filepath (default: data/obergail/dgm_4117-5101c.tif)
Reads band 1 into dgm_data, gets dgm_transform and dgm_profile
Replaces nodata with NaN
Load known boulders CSV

File: 20251029_Boulder shared_mod.csv (project folder)
load_known_boulders_from_csv parses each row, extracts lat/lon from links/text and returns a GeoDataFrame in EPSG:4326
Compute terrain metrics

calculate_roughness(dgm_data, window_size=2) → local std dev map
calculate_local_relief(dgm_data, window_size=4) → local max-min
calculate_slope(dgm_data, dgm_transform) → slope in degrees
These arrays are used for detection and for inferring parameters
Basic candidate extraction (diagnostic)

A simple combined_mask is built with thresholds (roughness, local relief, slope)
Connected components are labeled and small clusters are converted to centroids (boulder_coords)
A candidates GeoDataFrame is prepared (but single-run image is no longer saved)
Determine known boulders inside the DGM extent

Known points are projected into DGM CRS and filtered to those within the DEM bbox
Those points are kept for parameter inference and final overlay
Inference of parameters from known boulders

infer_parameters_from_known samples the three metric maps at the known points
Computes median local_relief, roughness, slope, and an estimated pixel-area
Builds one targeted parameter set (or a small list around the median) tuned to the known examples
Detection using inferred parameters (run_finer_sweep)

For the derived parameter set only (no broad sweep), the script:
Computes local_relief for the window size(s)
Builds the combined mask (roughness + relief + slope)
Labels connected regions and filters by inferred area limits
Converts selected regions to centroids and collects them into a GeoDataFrame
Project results to Web Mercator (EPSG:3857) and prepare plotting bounds
Final map saved (single image)

Adds satellite basemap tiles via contextily (Esri.WorldImagery)
Plots candidate centroids (color per inferred label) and overlays known boulders as black "x"
Saves one image: boulder_results/boulders_inferred_sweep_{tif_base}.png
tif_base is derived from the input TIFF path with "data/" removed
No intermediate figures are shown
Files and outputs

Final image(s) saved to: /home/ig0r/Documents/python_wk/boulder_scouting/boulder_results/
Look for boulders_inferred_sweep_{tif_base}.png
How to run

From project directory:
source your venv if needed, then:
python import_visualize.py
Requirements: rasterio, numpy, matplotlib, scipy, geopandas, contextily, shapely, pandas
Internet access required for contextily tile fetching

Quick tuning tips
If detections miss known boulders: adjust the inference logic or increase sampling radius in infer_parameters_from_known
To change plotted tile zoom: edit ctx.add_basemap(..., zoom=16) in run_finer_sweep (higher = closer)
To change marker sizes/colors: adjust the plotting subset.plot(...) calls
