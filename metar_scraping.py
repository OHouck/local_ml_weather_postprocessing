#!/usr/bin/env python3
import os
import time
import argparse
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

# Set the base URL of the archive. (Update this with the actual domain as needed.)
BASE_URL = "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"

def get_soup(url):
    """
    Fetches the given URL and returns a BeautifulSoup object.
    """
    print(f"Fetching: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def get_directory_links(url):
    """
    Returns a list of directory names (strings ending with '/') from the index page.
    It skips the "Parent Directory" link.
    """
    soup = get_soup(url)
    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and href not in ("../", "/") and href.endswith("/"):
            links.append(href)
    return links

def download_file(url, local_path):
    """
    Downloads the file from url to local_path.
    If the file already exists, it skips the download.
    """
    if os.path.exists(local_path):
        print(f"File already exists: {local_path} -- skipping.")
        return
    print(f"Downloading {url} to {local_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

def main(test_mode, output_dir):

    output_dir = os.path.expanduser(output_dir)
    # Determine which years to process.
    if test_mode:
        # Only January 2024 in test mode.
        years = ["2024/"]
        print("Running in test mode: only processing January 2024.")
    else:
        # Full mode: years 2001 through 2025 (each as string with trailing slash).
        years = [f"{year}/" for year in range(2006, 2026)]

    for year in years:
        year_url = urljoin(BASE_URL, year)
        print(f"\nProcessing year: {year.strip('/')}")
        if test_mode:
            months = ["01/"]
        else:
            months = get_directory_links(year_url)
        
        for month in months:
            month_url = urljoin(year_url, month)
            print(f"  Processing month: {month.strip('/')}")
            day_links = get_directory_links(month_url)
            for day in day_links:
                day_url = urljoin(month_url, day)
                print(f"    Processing day: {day.strip('/')}")
                day_subdirs = get_directory_links(day_url)
                if "point/" not in day_subdirs:
                    print(f"      'point/' folder not found at {day_url} -- skipping day.")
                    continue
                point_url = urljoin(day_url, "point/")
                point_subdirs = get_directory_links(point_url)
                if "metar/" not in point_subdirs:
                    print(f"      'metar/' folder not found at {point_url} -- skipping day.")
                    continue
                metar_url = urljoin(point_url, "metar/")
                metar_subdirs = get_directory_links(metar_url)
                if "netcdf/" not in metar_subdirs:
                    print(f"      'netcdf/' folder not found at {metar_url} -- skipping day.")
                    continue
                netcdf_url = urljoin(metar_url, "netcdf/")
                
                # Parse the netcdf directory page for .gz files.
                soup = get_soup(netcdf_url)
                file_links = [a.get("href") for a in soup.find_all("a")
                              if a.get("href") and a.get("href").endswith(".gz")]
                if not file_links:
                    print(f"      No .gz files found in {netcdf_url}.")
                    continue
                for file_link in file_links:
                    file_url = urljoin(netcdf_url, file_link)
                    # Build a local file path using the user-specified output_dir.
                    local_path = os.path.join(output_dir,
                                              year.strip("/"),
                                              month.strip("/"),
                                              day.strip("/"),
                                              "point", "metar", "netcdf",
                                              file_link)
                    download_file(file_url, local_path)
                    time.sleep(0.5)
                time.sleep(0.5)
            time.sleep(1)
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download METAR weather observation data from the archive."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only download January 2024 data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Volumes/wd_external_hd/noaa_data/metar_data",
        help="Output directory to save downloaded files (default: 'data')."
    )
    args = parser.parse_args()
    main(args.test, args.output)
