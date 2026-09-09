"""
Interactive map visualization module for SatQuery-AI M5.
Generates rich Folium maps with detected regions, bounding boxes, popups, and layer controls.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
import folium
from folium import LayerControl, TileLayer
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import mapping

from geospatial.area import calculate_polygon_area


def generate_folium_map(
    polygons: Optional[Sequence[Union[Polygon, MultiPolygon]]] = None,
    bboxes_geo: Optional[List[Dict[str, Any]]] = None,
    center_coords: Optional[Tuple[float, float]] = None,
    target: str = "detected_change",
    confidence: float = 0.85,
    output_html_path: Optional[Union[str, Path]] = None,
    zoom_start: int = 15,
) -> folium.Map:
    """
    Generate an interactive Folium map showing detected regions, bounding boxes,
    and metadata popups.

    Args:
        polygons: List of Shapely Polygons (in EPSG:4326, [lon, lat]).
        bboxes_geo: List of bbox dictionaries from coordinates.pixel_bbox_to_geo_bbox.
        center_coords: Optional center coordinate tuple (latitude, longitude).
        target: Target label string (e.g. 'deforestation', 'facility').
        confidence: Confidence score of detection.
        output_html_path: File path to save generated interactive HTML map.
        zoom_start: Initial zoom level.

    Returns:
        folium.Map instance.
    """
    polygons = polygons or []
    bboxes_geo = bboxes_geo or []

    # Calculate overall bounding box to determine map center and bounds
    all_lats: List[float] = []
    all_lons: List[float] = []

    for poly in polygons:
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        all_lons.extend([min_lon, max_lon])
        all_lats.extend([min_lat, max_lat])

    for bbox in bboxes_geo:
        g_bbox = bbox.get("geo_bbox", [])
        if len(g_bbox) == 4:
            all_lons.extend([g_bbox[0], g_bbox[2]])
            all_lats.extend([g_bbox[1], g_bbox[3]])

    if center_coords:
        map_center = [center_coords[0], center_coords[1]]
    elif all_lats and all_lons:
        map_center = [(min(all_lats) + max(all_lats)) / 2.0, (min(all_lons) + max(all_lons)) / 2.0]
    else:
        map_center = [20.0, 0.0]

    # Initialize Folium Map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles=None,  # custom tiles added below
        control_scale=True,
    )

    # Add Basemaps
    TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri World Imagery)",
        overlay=False,
        control=True,
    ).add_to(m)

    TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Feature Group: Detected Change Polygons
    poly_group = folium.FeatureGroup(name="Detected Regions (Polygons)", show=True)

    for idx, poly in enumerate(polygons):
        area_info = calculate_polygon_area(poly, crs="EPSG:4326")
        centroid = poly.centroid

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px; font-size: 13px;">
            <h4 style="margin: 0 0 8px; color: #d9534f; border-bottom: 2px solid #d9534f; padding-bottom: 4px;">
                Detected Change #{idx + 1}
            </h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Target:</b></td><td><span style="background: #fdf7e7; color: #b7791f; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{target}</span></td></tr>
                <tr><td><b>Confidence:</b></td><td>{confidence * 100:.1f}%</td></tr>
                <tr><td><b>Area (m²):</b></td><td><b>{area_info['area_sq_meters']:,.1f} m²</b></td></tr>
                <tr><td><b>Area (ha):</b></td><td><b>{area_info['area_hectares']:.4f} ha</b></td></tr>
                <tr><td><b>Centroid:</b></td><td>{centroid.y:.6f}°N, {centroid.x:.6f}°E</td></tr>
            </table>
        </div>
        """

        geo_json_data = {
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "id": idx + 1,
                "target": target,
                "confidence": confidence,
                "area_sq_meters": area_info["area_sq_meters"],
                "area_hectares": area_info["area_hectares"],
            },
        }

        folium.GeoJson(
            geo_json_data,
            style_function=lambda x: {
                "fillColor": "#ff4d4f",
                "color": "#cf1322",
                "weight": 3,
                "fillOpacity": 0.45,
            },
            highlight_function=lambda x: {
                "fillColor": "#ff7875",
                "color": "#a8071a",
                "weight": 4,
                "fillOpacity": 0.7,
            },
            tooltip=folium.Tooltip(
                f"<b>{target.upper()}</b> | Area: {area_info['area_hectares']:.3f} ha ({area_info['area_sq_meters']:,.0f} m²)"
            ),
            popup=cast(Any, folium.Popup(popup_html, max_width=320)),
        ).add_to(poly_group)

    poly_group.add_to(m)

    # Feature Group: Bounding Boxes
    bbox_group = folium.FeatureGroup(name="Bounding Boxes", show=True)

    for idx, bbox in enumerate(bboxes_geo):
        polygon_coords = bbox.get("polygon_coords", [])
        centroid = bbox.get("centroid", (0, 0))
        geo_bbox = bbox.get("geo_bbox", [0, 0, 0, 0])

        # Convert [lon, lat] pairs to [lat, lon] for Folium Polygon/PolyLine
        folium_coords = [[pt[1], pt[0]] for pt in polygon_coords]

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 180px; font-size: 13px;">
            <h4 style="margin: 0 0 6px; color: #1890ff;">Bounding Box #{idx + 1}</h4>
            <p style="margin: 3px 0;"><b>Target:</b> {target}</p>
            <p style="margin: 3px 0;"><b>Center:</b> {centroid[1]:.6f}°N, {centroid[0]:.6f}°E</p>
            <p style="margin: 3px 0;"><b>Bounds:</b> [{geo_bbox[0]:.4f}, {geo_bbox[1]:.4f}, {geo_bbox[2]:.4f}, {geo_bbox[3]:.4f}]</p>
        </div>
        """

        folium.Polygon(
            locations=folium_coords,
            color="#1890ff",
            weight=2,
            dash_array="6, 6",
            fill=True,
            fill_color="#1890ff",
            fill_opacity=0.15,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"BBox #{idx + 1} ({target})",
        ).add_to(bbox_group)

        # Centroid Marker
        folium.CircleMarker(
            location=[centroid[1], centroid[0]],
            radius=5,
            color="#1890ff",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=1.0,
            tooltip=f"Centroid #{idx + 1}: {centroid[1]:.5f}°N, {centroid[0]:.5f}°E",
        ).add_to(bbox_group)

    bbox_group.add_to(m)

    # Add Layer Control
    LayerControl(collapsed=False).add_to(m)

    # Fit bounds if we have features
    if all_lats and all_lons:
        south_west = [min(all_lats), min(all_lons)]
        north_east = [max(all_lats), max(all_lons)]
        # Add slight padding if it's a point
        if south_west == north_east:
            south_west = [south_west[0] - 0.001, south_west[1] - 0.001]
            north_east = [north_east[0] + 0.001, north_east[1] + 0.001]
        m.fit_bounds([south_west, north_east], padding=[20, 20])

    # Save to HTML if requested
    if output_html_path:
        out_path = Path(output_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))

    return m
