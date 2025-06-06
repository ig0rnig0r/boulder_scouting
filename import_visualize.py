import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import generic_filter, uniform_filter, label, find_objects # For roughness, local relief, and potential slope smoothing

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
    smoothed_dem = uniform_filter(np.nan_to_num(dem_data, nan=np.nanmean(dem_data)), size=3) # 3x3 filter
    
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
    # --- IMPORTANT: Replace this with your actual DGM file path ---
    dgm_filepath = 'data/maltatal/steinbruch/dgm_4622-5002a.tif'
    # dgm_filepath = 'data/maltatal/schleierwasserfall/dgm_4621-5103a.tif'
    # dgm_filepath = 'data/vassach/dgm_5017-5002a.tif'
    # dgm_filepath = 'data/zajesera/dgm_4916-5002a.tif'

    # --- Load DGM Data ---
    # We need to load the full DGM data to pass to calculation functions,
    # and also its transform for slope calculation.
    dgm_data = None
    dgm_transform = None
    dgm_profile = None

    try:
        with rasterio.open(dgm_filepath) as src:
            dgm_data = src.read(1)
            dgm_transform = src.transform
            dgm_profile = src.profile
            # Handle NoData values for calculations
            nodata_val = dgm_profile.get('nodata')
            if nodata_val is not None:
                dgm_data = np.where(dgm_data == nodata_val, np.nan, dgm_data)
        print(f"DGM data loaded for calculations from {dgm_filepath}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error loading DGM for calculations {dgm_filepath}: {e}")
        print("Exiting script as DGM data is essential.")
        exit() # Exit if DGM data cannot be loaded

    # # --- Plot DGM (3D Point Cloud) ---
    # print("\n--- Plotting DGM (3D Point Cloud) ---")
    # dgm_ax = plot_dgm_as_3d_points(dgm_filepath, sample_factor=5)
    # if dgm_ax:
    #     dgm_ax.view_init(elev=45, azim=-60) # Adjust view angle as desired
    #     plt.tight_layout()
    #     plt.show()

    # # --- Plot DGM as 3D Wireframe or Shaded surface ---
    # print("\n--- Plotting DGM as 3D Wireframe (Optional) ---")
    # wireframe_ax = plot_dgm_as_3d_surface(dgm_filepath, title="Digitales Geländemodell (DGM) - Wireframe", 
    #                                       sample_factor=2, z_exaggeration=2.0, plot_type='wireframe') # surface / wireframe
    # if wireframe_ax:
    #     wireframe_ax.view_init(elev=45, azim=-60)
    #     plt.tight_layout()
    #     plt.show()

    # --- Calculate and Plot Terrain Metrics ---
    if dgm_data is not None and dgm_transform is not None:
        # --- Calculate metrics ---
        roughness_map = calculate_roughness(dgm_data, window_size=3)
        # plot_2d_raster(roughness_map, 'DGM Roughness (Std Dev)', cmap='hot', cbar_label='Std Dev of Elevation (m)')

        local_relief_map = calculate_local_relief(dgm_data, window_size=5)
        # plot_2d_raster(local_relief_map, 'DGM Local Relief (Max - Min)', cmap='YlOrRd', cbar_label='Elevation Range (m)')

        slope_map = calculate_slope(dgm_data, dgm_transform)
        # plot_2d_raster(slope_map, 'DGM Slope', cmap='Greys_r', cbar_label='Slope (degrees)') # Greys_r for darker steep areas

        # --- Combined thresholding with max_boulder_height ---
        threshold_roughness = 0.5
        threshold_relief = 1.0      # Minimum local relief (meters)
        threshold_slope = 40        # Degrees, try 15-35
        max_boulder_height = 4.0    # Maximum local relief (meters) for a boulder

        combined_mask = (
            (roughness_map > threshold_roughness) &
            (local_relief_map > threshold_relief) &
            (local_relief_map < max_boulder_height) &  # Exclude big cliffs
            (slope_map > threshold_slope)
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(combined_mask, cmap='Greens', origin='upper')
        plt.title('Potential Boulder Areas (Combined Metrics, Height Limited)')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.tight_layout()
        plt.show()

        # --- Boulder Candidate Clustering ---
        labeled, num_features = label(combined_mask)
        print(f"Found {num_features} potential boulder clusters.")

        # Optionally, filter by size (area in pixels)
        min_boulder_size = 5
        boulder_slices = find_objects(labeled)
        filtered_mask = np.zeros_like(combined_mask)
        for i, slc in enumerate(boulder_slices):
            if slc is not None:
                region = (labeled[slc] == (i+1))
                if np.sum(region) >= min_boulder_size:
                    filtered_mask[slc][region] = True

        plt.figure(figsize=(10, 10))
        plt.imshow(filtered_mask, cmap='Blues', origin='upper')
        plt.title('Filtered Boulder Candidates (min size)')
        plt.show()

        # --- Improved Visualization: Overlay boulder candidates in red on local relief ---
        background = local_relief_map  # Or use dgm_data or slope_map for different context

        plt.figure(figsize=(12, 12))
        plt.imshow(background, cmap='gray', origin='upper')
        plt.imshow(
            np.ma.masked_where(filtered_mask == 0, filtered_mask),
            cmap='Reds', alpha=0.7, origin='upper'
        )
        plt.title('Filtered Boulder Candidates (highlighted in red)')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.colorbar(label='Local Relief (m)')
        plt.tight_layout()
        plt.show()

    print("\n--- DGM processing and metric calculations complete. ---")
    print("The next steps involve refining these thresholds, applying more advanced segmentation,")
    print("and leveraging your reference boulder data for machine learning.")