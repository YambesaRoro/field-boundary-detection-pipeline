"""
Data loading and geospatial raster/vector processing utilities
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
import warnings

warnings.filterwarnings("ignore")


class GeoRasterLoader:
    """Load and validate geospatial raster and vector data"""
    
    def __init__(self, raster_path, shapefile_path):
        self.raster_path = raster_path
        self.shapefile_path = shapefile_path
        self.src = None
        self.gdf = None
        self.transform = None
        self.crs = None
        self.bounds = None
        self.shape = None
        
    def load_raster(self):
        """Load raster metadata and data"""
        self.src = rasterio.open(self.raster_path)
        self.transform = self.src.transform
        self.crs = self.src.crs
        self.bounds = self.src.bounds
        self.shape = (self.src.height, self.src.width)
        
        print(f"✓ Raster loaded: {self.shape}")
        print(f"  CRS: {self.crs}")
        print(f"  Bounds: {self.bounds}")
        print(f"  Transform: {self.transform}")
        return self.src
    
    def load_shapefile(self):
        """Load and validate shapefile"""
        self.gdf = gpd.read_file(self.shapefile_path)
        
        print(f"✓ Shapefile loaded: {len(self.gdf)} features")
        print(f"  CRS: {self.gdf.crs}")
        
        # Reproject if needed
        if self.gdf.crs != self.crs:
            print(f"  Reprojecting to {self.crs}...")
            self.gdf = self.gdf.to_crs(self.crs)
        
        return self.gdf
    
    def read_raster_data(self):
        """Read full raster array"""
        if self.src is None:
            self.load_raster()
        
        data = self.src.read()  # (bands, height, width)
        print(f"✓ Raster data read: {data.shape}")
        return data
    
    def close(self):
        """Close raster file"""
        if self.src is not None:
            self.src.close()


def normalize_image(img, img_min=0, img_max=255):
    """Normalize image to [0, 1]"""
    img = img.astype(np.float32)
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return np.clip(img, 0, 1)


def rasterize_polygons(gdf, raster_shape, transform, fill_value=1):
    """
    Rasterize polygon geometries to binary mask
    
    Args:
        gdf: GeoDataFrame with polygon geometries
        raster_shape: (height, width) tuple
        transform: rasterio.transform.Affine object
        fill_value: value for polygon pixels
    
    Returns:
        mask: (height, width) numpy array
    """
    geoms = [(geom, fill_value) for geom in gdf.geometry if geom is not None and geom.is_valid]
    
    if not geoms:
        raise ValueError("No valid geometries found in GeoDataFrame")
    
    mask = rasterize(
        geoms,
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    
    print(f"✓ Polygons rasterized: {mask.shape}, {np.sum(mask > 0)} pixels")
    return mask


def get_bbox_from_raster(src):
    """Get bounding box from raster source"""
    return box(*src.bounds)


def validate_overlap(gdf, raster_bounds):
    """Check if shapefile overlaps with raster"""
    raster_box = box(*raster_bounds)
    intersections = gdf.geometry.intersects(raster_box)
    overlap_count = intersections.sum()
    
    print(f"✓ Overlap validation: {overlap_count}/{len(gdf)} features overlap raster")
    return overlap_count > 0
