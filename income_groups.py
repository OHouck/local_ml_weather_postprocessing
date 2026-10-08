"""
World Bank income groups for countries, grid cells and stations.

Countries are classified with the World Bank's 2024 income groups (fiscal year
FY25, announced July 2024 from 2023 GNI). The paper compares high-income
countries against everyone else, so the low, lower-middle and upper-middle
categories are collapsed into "Low and middle income" (LMIC).

Points (grid cells or stations) are assigned to a country with a spatial join
onto Natural Earth 1:50m country polygons. Both source files are downloaded
once into raw/income_groups/ and read from there afterwards.

Used by the gridded figures (figure 1 and the income map) and by the station
selection, which also takes each station's continent from the same polygons.
"""

import os
import shutil
import urllib.request
import warnings
import zipfile

import geopandas
import numpy as np
import pandas as pd

WORLD_BANK_CLASSIFICATION_URL = ("https://datacatalogfiles.worldbank.org/"
                                 "ddh-published/0037712/DR0090754/OGHIST.xlsx")
WORLD_BANK_FISCAL_YEAR = "FY25"
NATURAL_EARTH_COUNTRIES_URL = ("https://naciscdn.org/naturalearth/50m/cultural/"
                               "ne_50m_admin_0_countries.zip")

# World Bank one-letter codes -> the two groups the paper compares.
WORLD_BANK_CODE_TO_INCOME_GROUP = {
    "L": "Low and middle income",
    "LM": "Low and middle income",
    "UM": "Low and middle income",
    "H": "High income",
}

# The dictionary order is the bar and legend order used in every figure.
INCOME_GROUP_COLORS = {
    "Low and middle income": "#d7191c",
    "High income": "#2c7bb6",
}
INCOME_GROUP_ORDER = list(INCOME_GROUP_COLORS)

# Points are matched to the nearest country polygon within one ERA5 grid cell.
# The 1:50m coastlines are simplified enough that some coastal points (e.g.
# Mumbai's islands) sit a few km offshore; one cell absorbs them without
# pulling open-ocean points onto small island nations.
NEAREST_COUNTRY_DEGREES = 0.25


def load_income_group_countries(directories):
    """
    Load country polygons labelled with their World Bank income group.

    Downloads the World Bank classification spreadsheet and the Natural Earth
    shapefile on first use.

    Inputs:
        directories (dict): output of common.setup_directories().

    Returns:
        geopandas.GeoDataFrame in EPSG:4326 with columns 'iso3', 'name',
        'continent' (Natural Earth's continent name), 'income_group' and
        'geometry'. Disputed territories and countries without a FY25
        classification are dropped, so points there stay unlabelled.
    """
    cache_directory = os.path.join(directories["raw"], "income_groups")
    os.makedirs(cache_directory, exist_ok=True)

    # Download both source files the first time they are needed
    spreadsheet_path = os.path.join(cache_directory, "OGHIST.xlsx")
    zip_path = os.path.join(cache_directory, "ne_50m_admin_0_countries.zip")
    shapefile_path = os.path.join(cache_directory, "ne_50m_admin_0_countries.shp")
    downloads_needed = []
    if not os.path.exists(spreadsheet_path):
        downloads_needed.append((WORLD_BANK_CLASSIFICATION_URL, spreadsheet_path))
    if not os.path.exists(shapefile_path):
        downloads_needed.append((NATURAL_EARTH_COUNTRIES_URL, zip_path))
    for url, destination in downloads_needed:
        print(f"  Downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as response, \
                open(destination, "wb") as output_file:
            shutil.copyfileobj(response, output_file)
    if not os.path.exists(shapefile_path):
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(cache_directory)

    # Read the World Bank classification: row 4 holds the fiscal-year headers
    # and the countries start on row 11, with their ISO3 code in column 0
    spreadsheet = pd.read_excel(spreadsheet_path, sheet_name="Country Analytical History",
                                header=None)
    fiscal_year_headers = spreadsheet.iloc[4].astype(str).tolist()
    fiscal_year_column = fiscal_year_headers.index(WORLD_BANK_FISCAL_YEAR)
    iso3_to_income_group = {}
    for _, row in spreadsheet.iloc[11:].iterrows():
        iso3 = str(row.iloc[0]).strip()
        code = str(row.iloc[fiscal_year_column]).strip()
        if iso3 not in ("nan", "") and code in WORLD_BANK_CODE_TO_INCOME_GROUP:
            iso3_to_income_group[iso3] = WORLD_BANK_CODE_TO_INCOME_GROUP[code]

    # Read the country polygons, dropping disputed territories
    countries = geopandas.read_file(shapefile_path)
    countries = countries[~countries["TYPE"].isin(["Disputed", "Indeterminate"])]
    countries = countries.rename(columns={"ADM0_A3": "iso3", "NAME": "name",
                                          "CONTINENT": "continent"})
    countries = countries[["iso3", "name", "continent", "geometry"]].reset_index(drop=True)

    # Attach the income group and drop countries the World Bank does not classify
    countries["income_group"] = countries["iso3"].map(iso3_to_income_group)
    return countries.dropna(subset=["income_group"]).reset_index(drop=True)


def match_points_to_countries(latitudes, longitudes, countries, columns):
    """
    Look up attributes of the country each point falls in.

    Inputs:
        latitudes, longitudes (array-like): point coordinates in degrees.
            Longitudes may be in either the 0..360 or the -180..180 convention.
        countries (geopandas.GeoDataFrame): output of load_income_group_countries().
        columns (list of str): country columns to return, e.g. ['income_group'].

    Returns:
        pandas.DataFrame with one row per point (in input order) and the
        requested columns. Points more than NEAREST_COUNTRY_DEGREES from any
        classified country (ocean, Antarctica) get missing values.
    """
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)

    points = geopandas.GeoDataFrame({"point_index": np.arange(len(latitudes))},
                                    geometry=geopandas.points_from_xy(longitudes, latitudes),
                                    crs="EPSG:4326")
    # Distances are in degrees on purpose (the cap is one grid cell), so
    # geopandas' warning about measuring distance in a geographic CRS is silenced
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        matched = geopandas.sjoin_nearest(points, countries[columns + ["geometry"]], how="left",
                                          max_distance=NEAREST_COUNTRY_DEGREES)

    # A point equidistant from two countries is matched twice; keep the first
    matched = matched.drop_duplicates(subset="point_index", keep="first")
    return matched.sort_values("point_index")[columns].reset_index(drop=True)
