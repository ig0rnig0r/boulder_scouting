import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import generic_filter, uniform_filter, label, find_objects # For roughness, local relief, and potential slope smoothing
import os
from rasterio.transform import xy
from rasterio.warp import transform as rio_transform
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, box
import traceback
import pandas as pd
import re

def _extract_coords_from_text(s):
    """Try several patterns to extract (lat, lon) from a text (returns tuple lon,lat)."""
    if not isinstance(s, str):
        return None
    # common google maps style: /@lat,lon,zoom
    m = re.search(r'@(-?\d+\.\d+),\s*(-?\d+\.\d+)', s)
    if m:
        lat = float(m.group(1)); lon = float(m.group(2))
        return lon, lat
    # params ll=lat,lon or q=lat,lon
    m = re.search(r'[?&](?:ll|q)=(-?\d+\.\d+),\s*(-?\d+\.\d+)', s)
    if m:
        lat = float(m.group(1)); lon = float(m.group(2))
        return lon, lat
    # fallback: last pair of decimals in the string
    matches = re.findall(r'(-?\d+\.\d+)', s)
    if len(matches) >= 2:
        # try to take last two as lat,lon or lon,lat — assume lat then lon if within -90..90
        a = float(matches[-2]); b = float(matches[-1])
        if -90 <= a <= 90 and -180 <= b <= 180:
            return b, a  # convert to lon,lat
        if -90 <= b <= 90 and -180 <= a <= 180:
            return a, b
    return None

def load_known_boulders_from_csv(csv_path):
    """
    Load CSV and try to parse coordinates from the last column / link text.
    Returns GeoDataFrame in EPSG:4326 (lon, lat) or empty GeoDataFrame.
    """
    try:
        df = pd.read_csv(csv_path, header=0, dtype=str, keep_default_na=False)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    coords = []
    # try each row: check all string fields (end of line typically)
    for _, row in df.iterrows():
        # check last non-empty cell
        last_val = None
        for val in row[::-1]:
            if isinstance(val, str) and val.strip() != "":
                last_val = val.strip()
                break
        if not last_val:
            continue
        parsed = _extract_coords_from_text(last_val)
        if parsed:
            lon, lat = parsed
            coords.append(Point(lon, lat))
    if not coords:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(geometry=coords, crs="EPSG:4326")

def plot_dgm_as_3d_points(filepath, title="Digitales Geländemodell (DGM)", 
                          ax=None, sample_factor=5, z_exaggeration=1.0, 
                          cmap='terrain', alpha=0.8):
    """
    Reads a DGM GeoTIFF, converts it to a 3D point cloud (X, Y, Z), and plots it.
    Args:
        filepath (str): Path to the DGM GeoTIFF file.
        title (str): Title for the plot.
        ax (Axes3D, optional): Matplotlib Axes3D object to plot on. If None, a new figure is created.
        sample_factor (int): Factor by which to sample points (e.g., 5 means 1 in 5 pixels).
                             Use a higher number for large files to avoid memory issues.
        z_exaggeration (float): Factor to exaggerate the Z-axis for better visual relief.
        cmap (str): Colormap for the points (e.g., 'terrain', 'viridis', 'gist_earth').
        alpha (float): Transparency of the points (0.0 to 1.0).
    Returns:
        Axes3D: The matplotlib Axes3D object used for plotting.
    """
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1) # Read the first band (elevation)
            transform = src.transform
            crs = src.crs
            profile = src.profile
            print(f"Loading {title} from: {filepath}")
            print(f"Original data shape: {data.shape}")
            print(f"CRS: {crs}")
            print(f"Resolution (x, y): {transform.a:.2f}, {transform.e:.2f}") 
            print(f"NoData value: {profile.get('nodata')}")
            # Handle NoData values: replace with NaN
            nodata_val = profile.get('nodata')
            if nodata_val is not None:
                data = np.where(data == nodata_val, np.nan, data)
            rows, cols = data.shape
            x_coords, y_coords = rasterio.transform.xy(transform, np.arange(rows), np.arange(cols))
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
            valid_mask = ~np.isnan(data)
            x_points = X_grid[valid_mask]
            y_points = Y_grid[valid_mask]
            z_points = data[valid_mask]
            print(f"Total valid points found: {len(x_points)}")
            if sample_factor > 1:
                indices = np.arange(len(x_points))
                rng = np.random.default_rng(42) 
                rng.shuffle(indices) 
                sampled_indices = indices[::sample_factor]
                x_points = x_points[sampled_indices]
                y_points = y_points[sampled_indices]
                z_points = z_points[sampled_indices]
                print(f"Sampled {len(x_points)} points for plotting (sample_factor={sample_factor})")
            else:
                print(f"Plotting all {len(x_points)} valid points.")
            if ax is None:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x_points, y_points, z_points * z_exaggeration, 
                            c=z_points, cmap=cmap, s=0.1, alpha=alpha)
            ax.set_xlabel('X Coordinate (m)') 
            ax.set_ylabel('Y Coordinate (m)') 
            ax.set_zlabel('Z Elevation (m)')
            ax.set_title(title)
            plt.colorbar(sc, label='Elevation (m)', shrink=0.5, aspect=10)
            return ax
    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading GeoTIFF {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def plot_dgm_as_3d_surface(filepath, title="Digitales Geländemodell (DGM) - Surface", 
                           ax=None, sample_factor=1, z_exaggeration=1.0, 
                           cmap='terrain', shade=True, plot_type='surface'):
    """
    Reads a DGM GeoTIFF, converts it to a 3D surface or wireframe, and plots it.
    NaN values are filled for plotting robustness.
    Args:
        filepath (str): Path to the DGM GeoTIFF file.
        title (str): Title for the plot.
        ax (Axes3D, optional): Matplotlib Axes3D object to plot on. If None, a new figure is created.
        sample_factor (int): Factor by which to sample the grid for plotting (e.g., 10 means 1 in 10 rows/cols).
                             Use a higher number for large files to avoid memory issues and render faster.
                             For surface plots, this is crucial.
        z_exaggeration (float): Factor to exaggerate the Z-axis for better visual relief.
        cmap (str): Colormap for the surface.
        shade (bool): Whether to apply light shading to the surface (only for 'surface' plot_type).
        plot_type (str): 'surface' for a shaded surface, 'wireframe' for lines.
    Returns:
        Axes3D: The matplotlib Axes3D object used for plotting.
    """
    if plot_type not in ['surface', 'wireframe']:
        raise ValueError("plot_type must be 'surface' or 'wireframe'")
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            profile = src.profile
            print(f"Loading {title} for {plot_type} plot from: {filepath}")
            print(f"Original data shape: {data.shape}")
            print(f"Resolution (x, y): {transform.a:.2f}, {transform.e:.2f}") 
            print(f"NoData value: {profile.get('nodata')}")
            # Handle NoData values for surface plotting
            nodata_val = profile.get('nodata')
            if nodata_val is not None:
                # Replace nodata with NaN
                data = np.where(data == nodata_val, np.nan, data)
            # Fill NaNs for plotting (e.g., with a small value or mean)
            # This is CRUCIAL for plot_surface/plot_wireframe
            data_filled = np.nan_to_num(data, nan=np.nanmin(data) - 10) # Fill with a value below min
            rows, cols = data_filled.shape
            x_coords, y_coords = rasterio.transform.xy(transform, np.arange(rows), np.arange(cols))
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
            print(f"Plotting {plot_type} with sample factor: {sample_factor}")
            # --- Downsampling for plotting efficiency ---
            # Apply sample_factor to the grids (X, Y, Z) directly
            X_plot = X_grid[::sample_factor, ::sample_factor]
            Y_plot = Y_grid[::sample_factor, ::sample_factor]
            Z_plot = data_filled[::sample_factor, ::sample_factor] * z_exaggeration
            if ax is None:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, projection='3d')
            if plot_type == 'surface':
                surf = ax.plot_surface(X_plot, Y_plot, Z_plot, 
                                       cmap=cmap, 
                                       rstride=1, cstride=1, # Plot every row/column of sampled grid
                                       linewidth=0, antialiased=False,
                                       shade=shade)
                plt.colorbar(surf, label='Elevation (m)', shrink=0.5, aspect=10)
            elif plot_type == 'wireframe':
                ax.plot_wireframe(X_plot, Y_plot, Z_plot, 
                                  rstride=1, cstride=1, # Plot every row/column of sampled grid
                                  color='gray', linewidth=0.5)
            ax.set_xlabel('X Coordinate (m)') 
            ax.set_ylabel('Y Coordinate (m)') 
            ax.set_zlabel('Z Elevation (m)')
            ax.set_title(title)
            return ax
    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading GeoTIFF {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- TERRAIN METRICS ---
def calculate_roughness(dem_data, window_size=3):
    """
    Calculates local roughness (standard deviation of elevation) within a moving window.
    A larger window_size captures larger features.
    """
    print(f"Calculating roughness with window size: {window_size}x{window_size} pixels...")
    # Create a copy to avoid modifying the original data (important if it has NaNs)
    processed_data = dem_data.copy()
    # Replace NaNs with a value that doesn't affect std dev much, or handle them specifically.
    # For std dev, it's safer to use `np.nanstd` within generic_filter if available,
    # or interpolate/mask NaNs before. scipy's generic_filter has issues with NaNs directly.
    # For simplicity, we'll convert NaNs to zeros *for the calculation*, but know this can affect edges.
    # A more robust approach involves masking or interpolation.
    temp_data = np.nan_to_num(processed_data, nan=np.nanmean(processed_data)) # Fill NaNs with mean for robust calculation
    # Define the function to apply: standard deviation
    roughness_map = generic_filter(temp_data, np.std, size=window_size, mode='constant', cval=np.nanmean(processed_data))
    # Restore NaNs for areas where the original data was NaN
    roughness_map[np.isnan(dem_data)] = np.nan
    return roughness_map

def calculate_slope(dem_data, transform):
    """
    Calculates slope in degrees from a DTM.
    Assumes DTM values are in meters and resolution (transform.a, transform.e) is in meters.
    """
    print("Calculating slope...")
    # Calculate gradients in x and y directions
    # np.gradient handles NaNs by returning NaN for derivatives involving them.
    # We need to consider the cell size (resolution) for accurate slope calculation.
    dx = np.abs(transform.a) # X resolution
    dy = np.abs(transform.e) # Y resolution (often negative, so use abs)
    # Use a smoothed version of the DEM data for more robust slope calculation
    # A small uniform filter can help reduce noise that would result in spiky slopes.
    smoothed_dem = uniform_filter(np.nan_to_num(dem_data, nan=np.nanmean(dem_data)), size=1) # 3x3 filter
    dz_dy, dz_dx = np.gradient(smoothed_dem, dy, dx) # Order matters for np.gradient output (rows, cols)
    # Slope in radians: atan(sqrt(dz/dx)^2 + (dz/dy)^2)
    slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    # Convert to degrees
    slope_degrees = np.degrees(slope_radians)
    # Restore NaNs for areas where the original data was NaN
    slope_degrees[np.isnan(dem_data)] = np.nan
    return slope_degrees

def calculate_local_relief(dem_data, window_size=5):
    """
    Calculates local relief (max elevation - min elevation) within a moving window.
    """
    print(f"Calculating local relief with window size: {window_size}x{window_size} pixels...")
    processed_data = dem_data.copy()
    temp_data = np.nan_to_num(processed_data, nan=np.nanmean(processed_data)) 
    # Define functions for min and max
    min_filter = generic_filter(temp_data, np.min, size=window_size, mode='constant', cval=np.nanmean(processed_data))
    max_filter = generic_filter(temp_data, np.max, size=window_size, mode='constant', cval=np.nanmean(processed_data))
    local_relief_map = max_filter - min_filter
    # Restore NaNs
    local_relief_map[np.isnan(dem_data)] = np.nan
    return local_relief_map

def plot_2d_raster(data, title, cmap='viridis', figsize=(10, 10), cbar_label='Value'):
    """
    Helper function to plot a 2D raster (e.g., a metric map).
    """
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap, origin='upper')
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":

    try:
        # DGM path (edit if needed)
        # dgm_filepath = 'data/obergail/dgm_4117-5101c.tif'
        # dgm_filepath = 'data/maltatal/schleierwasserfall/dgm_4621-5103a.tif'
        # dgm_filepath = 'data/schütt/dgm_4916-5000c.tif'
        dgm_filepath = 'data/vassach/dgm_5017-5002a.tif'

        # --- Load DGM ---
        with rasterio.open(dgm_filepath) as src:
            dgm_data = src.read(1)
            dgm_transform = src.transform
            dgm_profile = src.profile
            nodata_val = dgm_profile.get('nodata')
            if nodata_val is not None:
                dgm_data = np.where(dgm_data == nodata_val, np.nan, dgm_data)
        print(f"DGM loaded: {dgm_filepath}  shape={None if dgm_data is None else dgm_data.shape}")

        # --- Load known boulders CSV (if present) ---
        csv_path = os.path.join(os.path.dirname(__file__), "20251029_Boulder shared_mod.csv")
        known_gdf = load_known_boulders_from_csv(csv_path)
        if known_gdf.empty:
            print("No valid known boulder coordinates parsed from CSV (or CSV missing).")
        else:
            print(f"Loaded {len(known_gdf)} known boulders from CSV (EPSG:4326).")

        # --- Calculate terrain metrics ---
        if dgm_data is None or dgm_transform is None:
            raise RuntimeError("DGM data or transform not loaded")

        roughness_map = calculate_roughness(dgm_data, window_size=2)
        local_relief_map = calculate_local_relief(dgm_data, window_size=4)
        slope_map = calculate_slope(dgm_data, dgm_transform)
        print("Terrain metrics computed.")

        # --- Basic candidate extraction (keeps previous default logic) ---
        threshold_roughness = 0.1
        min_boulder_height = 2.0
        max_boulder_height = 5.0
        threshold_slope = 60

        combined_mask = (
            (roughness_map > threshold_roughness) &
            (local_relief_map > min_boulder_height) &
            (local_relief_map < max_boulder_height) &
            (slope_map > threshold_slope)
        )

        labeled_arr, num_features = label(combined_mask)
        print(f"Found {num_features} potential clusters (pre-filter).")

        # filter by size
        min_boulder_size = 0
        max_boulder_size = 12
        boulder_slices = find_objects(labeled_arr)

        boulder_coords = []
        for i, slc in enumerate(boulder_slices):
            if slc is None:
                continue
            region = (labeled_arr[slc] == (i + 1))
            region_size = int(np.sum(region))
            if not (min_boulder_size <= region_size <= max_boulder_size):
                continue
            rows, cols = np.where(region)
            if rows.size == 0:
                continue
            row_c = rows.mean() + slc[0].start
            col_c = cols.mean() + slc[1].start
            x, y = xy(dgm_transform, row_c, col_c)
            boulder_coords.append((x, y))

        print(f"Computed {len(boulder_coords)} potential boulder centroids.")

        # Prepare GeoDataFrame of candidates
        candidates_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in boulder_coords],
                                          crs=dgm_profile.get('crs') or "EPSG:4326")

        # If known points exist, project them into DGM CRS and keep only those inside the DGM extent
        known_in_bbox = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        if not known_gdf.empty:
            dgm_crs = dgm_profile.get('crs') or "EPSG:4326"
            # compute DGM corner coordinates (map CRS) and build bbox using shapely.box
            ulx, uly = xy(dgm_transform, 0, 0)  # upper-left
            lrx, lry = xy(dgm_transform, dgm_data.shape[0]-1, dgm_data.shape[1]-1)  # lower-right
            minx = min(ulx, lrx); maxx = max(ulx, lrx)
            miny = min(uly, lry); maxy = max(uly, lry)
            bbox_geom = box(minx, miny, maxx, maxy)
            bbox = gpd.GeoDataFrame(geometry=[bbox_geom], crs=dgm_crs)
            try:
                if known_gdf.crs != dgm_crs:
                    known_proj = known_gdf.to_crs(dgm_crs)
                else:
                    known_proj = known_gdf
                known_in_bbox = known_proj[known_proj.within(bbox.loc[0, 'geometry'])]
                print(f"{len(known_in_bbox)} known boulders fall inside the DGM extent.")
            except Exception:
                print("Failed to project/filter known points — proceeding without known overlay.")
                traceback.print_exc()

        # --- Final step: call inference-based sweep if available, otherwise save one combined image ---
        tif_relpath = os.path.relpath(dgm_filepath, "data")
        tif_base = os.path.splitext(tif_relpath)[0].replace(os.sep, "_")
        out_dir = "boulder_results"
        os.makedirs(out_dir, exist_ok=True)

        if 'run_finer_sweep' in globals() and callable(globals()['run_finer_sweep']):
            try:
                # prefer function signature that expects metrics if present
                func = globals()['run_finer_sweep']
                try:
                    func(dgm_data, dgm_transform, dgm_profile, out_dir, tif_base,
                         roughness_map, local_relief_map, slope_map)
                except TypeError:
                    # fallback: call with simpler signature
                    func(dgm_data, dgm_transform, dgm_profile, out_dir, tif_base, roughness_map)
            except Exception:
                print("Error running run_finer_sweep:")
                traceback.print_exc()
        else:
            # produce a single combined image (candidates + known points) as fallback
            print("run_finer_sweep not found — producing one fallback combined image.")
            # project to web mercator
            try:
                cmap_crs = candidates_gdf.crs or dgm_profile.get('crs') or "EPSG:4326"
                cand_web = candidates_gdf.to_crs(epsg=3857) if not candidates_gdf.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
                known_web = known_in_bbox.to_crs(epsg=3857) if not known_in_bbox.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
                all_g = cand_web if cand_web.shape[0] else known_web
                if all_g.empty:
                    print("No geometries to plot in fallback image.")
                else:
                    minx, miny, maxx, maxy = all_g.total_bounds
                    pad = 100
                    minx -= pad; miny -= pad; maxx += pad; maxy += pad
                    fig, ax = plt.subplots(figsize=(12, 12))
                    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
                    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=cand_web.crs.to_string(), zoom=16)
                    if not cand_web.empty:
                        cand_web.plot(ax=ax, color='red', markersize=50, label='candidates')
                    if not known_web.empty:
                        known_web.plot(ax=ax, marker='x', color='black', markersize=120, linewidth=2, label='known')
                    ax.axis('off'); plt.legend()
                    out_fname = os.path.join(out_dir, f"boulders_combined_{tif_base}.png")
                    plt.tight_layout(); plt.savefig(out_fname, dpi=300); plt.close()
                    print(f"Saved fallback combined image: {out_fname}")
            except Exception:
                print("Failed to create fallback combined image:")
                traceback.print_exc()

    except Exception:
        print("Fatal error in main execution — full traceback:")
        traceback.print_exc()