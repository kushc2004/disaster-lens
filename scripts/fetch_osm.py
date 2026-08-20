#!/usr/bin/env python3
"""Acquire real OSM roads and response facilities for one event.

Use roads_source= and facilities_source= for local extracts, or bbox=min_lon,
min_lat,max_lon,max_lat to query an Overpass endpoint.  Empty or malformed
responses fail; no substitute graph is generated.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, shape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERPASS = "https://overpass-api.de/api/interpreter"


def parse(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value, received {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_local(roads_path: Path, facilities_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads = gpd.read_file(roads_path)
    facilities = gpd.read_file(facilities_path)
    if roads.crs is None or facilities.crs is None:
        raise ValueError("Local OSM-derived sources must contain CRS metadata")
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    facilities = facilities[facilities.geometry.notna() & ~facilities.geometry.is_empty].copy()
    if roads.empty or facilities.empty:
        raise ValueError("Local OSM sources must contain real roads and response facilities")
    facilities.geometry = facilities.geometry.representative_point()
    return roads.to_crs(4326), facilities.to_crs(4326)


def overpass_query(bbox: tuple[float, float, float, float], endpoint: str) -> dict[str, Any]:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be west,south,east,north in WGS84")
    box = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:180];
    (
      way[highway]({box});
      node[amenity~"hospital|clinic|doctors|fire_station"]({box});
      way[amenity~"hospital|clinic|doctors|fire_station"]({box});
      relation[amenity~"hospital|clinic|doctors|fire_station"]({box});
      node[emergency]({box});
      way[emergency]({box});
    );
    out tags center geom;
    """
    payload = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"User-Agent": "DisasterLens/0.1", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"Overpass returned HTTP {response.status}")
        document = json.load(response)
    if not isinstance(document.get("elements"), list):
        raise ValueError("Overpass response lacks an elements array")
    return document


def parse_overpass(document: dict[str, Any]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    road_rows: list[dict[str, Any]] = []
    facility_rows: list[dict[str, Any]] = []
    facility_tags = {"hospital", "clinic", "doctors", "fire_station"}
    for element in document["elements"]:
        tags = element.get("tags") or {}
        geometry = element.get("geometry") or []
        if "highway" in tags and len(geometry) >= 2:
            line = LineString([(float(node["lon"]), float(node["lat"])) for node in geometry])
            if not line.is_empty and line.length:
                road_rows.append(
                    {
                        "osm_id": f"{element.get('type', 'way')}/{element['id']}",
                        "highway": tags.get("highway"),
                        "name": tags.get("name"),
                        "geometry": line,
                    }
                )
        amenity = tags.get("amenity")
        if amenity in facility_tags or "emergency" in tags:
            point: Point | None = None
            if "lat" in element and "lon" in element:
                point = Point(float(element["lon"]), float(element["lat"]))
            elif element.get("center"):
                point = Point(float(element["center"]["lon"]), float(element["center"]["lat"]))
            elif len(geometry) >= 3:
                point = shape(
                    {
                        "type": "Polygon",
                        "coordinates": [[(float(node["lon"]), float(node["lat"])) for node in geometry]],
                    }
                ).representative_point()
            if point is not None:
                facility_rows.append(
                    {
                        "osm_id": f"{element.get('type', 'node')}/{element['id']}",
                        "facility_type": amenity or tags.get("emergency"),
                        "name": tags.get("name"),
                        "geometry": point,
                    }
                )
    if not road_rows:
        raise ValueError("Overpass response contains no usable highway geometry")
    if not facility_rows:
        raise ValueError("Overpass response contains no hospital/clinic/emergency facilities")
    return (
        gpd.GeoDataFrame(road_rows, geometry="geometry", crs=4326),
        gpd.GeoDataFrame(facility_rows, geometry="geometry", crs=4326),
    )


def graph_from_roads(roads: gpd.GeoDataFrame) -> nx.Graph:
    projected = roads.estimate_utm_crs()
    metric = roads.to_crs(projected)
    graph = nx.Graph()
    for row in metric.itertuples():
        geometries = row.geometry.geoms if row.geometry.geom_type == "MultiLineString" else [row.geometry]
        for line in geometries:
            coordinates = list(line.coords)
            for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
                a = f"{start[0]:.3f},{start[1]:.3f}"
                b = f"{end[0]:.3f},{end[1]:.3f}"
                length = float(LineString((start, end)).length)
                if length > 0:
                    graph.add_edge(a, b, length_m=length, highway=str(getattr(row, "highway", "unknown")))
    if graph.number_of_edges() == 0:
        raise ValueError("OSM roads could not form a non-empty graph")
    graph.graph["crs"] = str(projected)
    graph.graph["source"] = "OpenStreetMap"
    return graph


def main() -> None:
    options = parse(sys.argv[1:])
    event_id = options.get("event_id")
    if not event_id:
        raise ValueError("event_id=<audited BRIGHT event> is required")
    local_mode = bool(options.get("roads_source") or options.get("facilities_source"))
    query_mode = bool(options.get("bbox"))
    if local_mode == query_mode:
        raise ValueError(
            "Supply both roads_source= and facilities_source=, or bbox=west,south,east,north"
        )
    if local_mode:
        if not options.get("roads_source") or not options.get("facilities_source"):
            raise ValueError("Local mode requires both roads_source= and facilities_source=")
        roads_source = local_path(options["roads_source"])
        facilities_source = local_path(options["facilities_source"])
        roads, facilities = normalize_local(roads_source, facilities_source)
        source: dict[str, Any] = {
            "mode": "local_osm_extract",
            "roads": str(roads_source.resolve()),
            "roads_sha256": digest(roads_source),
            "facilities": str(facilities_source.resolve()),
            "facilities_sha256": digest(facilities_source),
        }
    else:
        bbox_values = tuple(float(value) for value in options["bbox"].split(","))
        if len(bbox_values) != 4:
            raise ValueError("bbox requires four comma-separated WGS84 values")
        endpoint = options.get("overpass_url", DEFAULT_OVERPASS)
        document = overpass_query(bbox_values, endpoint)
        roads, facilities = parse_overpass(document)
        source = {"mode": "overpass", "endpoint": endpoint, "bbox": bbox_values}

    output = Path(options.get("output_dir", f"data/external/osm/{event_id}"))
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    roads_path, facilities_path = output / "roads.gpkg", output / "facilities.gpkg"
    roads.to_file(roads_path, layer="roads", driver="GPKG")
    facilities.to_file(facilities_path, layer="facilities", driver="GPKG")
    graph = graph_from_roads(roads)
    nx.write_graphml(graph, output / "graph.graphml")
    metadata = {
        "event_id": event_id,
        "dataset": "OpenStreetMap",
        "retrieved_or_copied_at": datetime.now(UTC).isoformat(),
        "source": source,
        "road_features": len(roads),
        "facility_features": len(facilities),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "roads_output_crs": str(roads.crs),
        "facilities_output_crs": str(facilities.crs),
        "graph_metric_crs": graph.graph["crs"],
        "license_notice": "OpenStreetMap data is available under ODbL; attribution required.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[osm] validated real OSM context: {output}", flush=True)


if __name__ == "__main__":
    main()
