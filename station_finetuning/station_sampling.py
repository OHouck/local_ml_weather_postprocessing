"""Continent and income-group classification of stations, and sampling from them.

The station draw in station_observations.py answers "which stations exist". This
module answers "which of them should we actually run", which is a different
question once the candidate pool is thirteen thousand stations and the machine
doing the work is a laptop.

Two ideas shape it.

The sample is balanced on income group and left proportional across continents.
Balancing income is the point of the analysis: the rich/poor gap is the quantity
being measured, and HadISD has roughly twice as many high-income stations as
low-and-middle-income ones, so an unbalanced draw would measure the gap with
lopsided precision. Balancing continents is not possible in the same way — there
are no high-income HadISD stations in Africa at all — so continents are left in
their natural proportions and simply reported.

Stations are tagged with a continent purely so the expensive stages can run one
chunk at a time. A continent is a processing unit here, not a stratum: no quota
is applied to it and the chunks are deliberately unequal, because that is what
the worldwide draw produces. The split exists so the forecast neighbourhoods for
one chunk fit comfortably on a laptop and a run can be stopped and resumed
between chunks.
"""

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetuning.erl_figures import _load_income_geodataframe  # noqa: E402

# Natural Earth's continent names mapped to the slugs the rest of the project
# already uses for continents (see helper_funcs.load_all_continent_patches).
# Antarctica and the open-ocean rows are deliberately absent: no station that
# matters to this analysis falls in them.
CONTINENT_SLUGS = {
    "Africa": "africa",
    "Asia": "asia",
    "Europe": "europe",
    "North America": "north_america",
    "South America": "south_america",
    "Oceania": "oceania",
}

# Same cap as _classify_pixels_by_income: one ERA5 cell. The 1:50m coastlines
# are simplified enough that a genuinely coastal station can sit a few km
# offshore of its own country polygon, and one cell absorbs that without
# dragging open-ocean points onto small island nations.
NEAREST_COUNTRY_DEGREES = 0.25


def load_country_attributes(dirs):
    """Load country polygons carrying both income group and continent.

    Both attributes come off the same polygons, so they are matched to stations
    in a single spatial join rather than two.

    Inputs:
        dirs (dict): output of helper_funcs.setup_directories(), used to find
            the cached Natural Earth shapefile and World Bank classification.

    Returns:
        gpd.GeoDataFrame: columns 'income_group', 'continent', 'iso3' and
            'geometry'.
            Countries with no World Bank classification are dropped, as they
            are for the paper's figures, so a station there stays unlabelled.
            'continent' is the project's slug, or None for the handful of
            Natural Earth rows that are not one of the six continents.
    """
    countries = _load_income_geodataframe(dirs).copy()
    countries["continent"] = countries["continent"].map(CONTINENT_SLUGS)
    return countries[["income_group", "continent", "iso3", "geometry"]]


def classify_stations(stations, dirs):
    """Tag each station with its World Bank income group, continent and country.

    Inputs:
        stations (pd.DataFrame): must carry 'latitude' and 'longitude' in
            degrees. Longitude may be in either 0..360 or -180..180.
        dirs (dict): output of helper_funcs.setup_directories().

    Returns:
        pd.DataFrame: a copy of stations with 'income_group', 'continent' and
            'iso3' (ISO 3166 alpha-3 country code) columns added. All are None
            for a station that matches no
            classified country, which is mostly small islands and a handful of
            genuinely offshore platforms.
    """
    countries = load_country_attributes(dirs)

    longitudes = np.asarray(stations["longitude"], dtype=float)
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    points = gpd.GeoDataFrame(
        {"_position": np.arange(len(stations))},
        geometry=gpd.points_from_xy(longitudes,
                                    np.asarray(stations["latitude"], dtype=float)),
        crs="EPSG:4326")

    joined = gpd.sjoin_nearest(points, countries, how="left",
                               max_distance=NEAREST_COUNTRY_DEGREES)
    # A point equidistant from two polygons matches both; keep one row per
    # station so the result stays aligned with the input.
    joined = joined.drop_duplicates(subset="_position").sort_values("_position")

    # A left join keeps one row per station and the sort restores the input
    # order, so the columns can be attached directly. Unmatched stations carry
    # NaN, which every consumer here tests for with notna()/isna().
    classified = stations.copy()
    classified["income_group"] = joined["income_group"].to_numpy()
    classified["continent"] = joined["continent"].to_numpy()
    classified["iso3"] = joined["iso3"].to_numpy()
    return classified


def select_income_balanced_sample(stations, stations_per_income_group,
                                  random_seed=58):
    """Draw an equal number of stations from each income group.

    Within a group the draw is uniform, so continents keep their natural
    proportions. That is deliberate: HadISD's high-income stations are almost all
    in Europe and North America, and there are none at all in Africa, so any
    attempt to balance continents within the high-income group would either
    fail or silently reweight towards a handful of unusual sites.

    Inputs:
        stations (pd.DataFrame): candidates, already filtered to those that
            clear the coverage bar. Must carry 'income_group'.
        stations_per_income_group (int): how many to take from each group. A
            group with fewer than this contributes all of them.
        random_seed (int): seed for the draw, so a re-run selects the same
            stations.

    Returns:
        pd.DataFrame: the selected stations, ordered by income group then by
            their original order. Stations with no income group are excluded,
            since every result in this folder is reported by income group.
    """
    selected_frames = []
    for income_group, group_rows in stations[
            stations["income_group"].notna()].groupby("income_group"):
        if len(group_rows) > stations_per_income_group:
            group_rows = group_rows.sample(n=stations_per_income_group,
                                           random_state=random_seed)
        selected_frames.append(group_rows.sort_index())

    if not selected_frames:
        return stations.iloc[[]].copy()
    return pd.concat(selected_frames).reset_index(drop=True)


def in_regions(stations, continents, excluded_countries=()):
    """Flag stations inside the given continents, minus some countries.

    Inputs:
        stations (pd.DataFrame): output of classify_stations, carrying
            'continent', 'iso3' and 'income_group'.
        continents (list[str]): continent slugs to include, e.g. ['africa'].
        excluded_countries (list[str]): ISO alpha-3 codes inside those
            continents to leave out, e.g. ['ARG'].

    Returns:
        pd.Series: boolean mask aligned with stations. Stations with no income
            group are always False, since every result is reported by income
            group.
    """
    return (stations["continent"].isin(continents)
            & ~stations["iso3"].isin(excluded_countries)
            & stations["income_group"].notna())
