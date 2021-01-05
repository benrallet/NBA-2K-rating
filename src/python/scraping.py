import argparse
import logging
import requests
import os
import pandas as pd

# Logging configuration
MSG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=MSG_FORMAT, datefmt=DATETIME_FORMAT)
logger = logging.getLogger("2K RATING")
logger.setLevel(logging.INFO)

# Global variables
root = os.path.abspath(os.path.join(os.curdir, "../.."))

cols_2K = ["Player", "Rating", "Year"]

cols_stats = [
    "Rk", "Player", "Pos", "Age", "Tm", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA",
    "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK",
    "TOV", "PF", "PTS", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%",
    "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP", "Year"
]

urls_2K = ["https://hoopshype.com/nba2k/{}-{}/"]

urls_stats = [
    "https://www.basketball-reference.com/leagues/NBA_{}_totals.html",
    "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html"
]


def intersection(l1, l2):
    """
    Return the intersection of two lists

    :params:
        l1 - list 1
        l2 - list 2

    :return: array with the intersection of l1 and l2

    :tests:
    >>> l1 = [1, 5, 8, 9]
    >>> l2 = [2, 1, 4, 9]
    >>> intersection(l1, l2)
    [1, 9]
    """
    return [item for item in l1 if item in l2]


def scraping(urls, cols, year_min, year_max):
    """
    Retrieve HCP and selected demographic and consent informations

    :params:
        urls     - array of the urls to scrap
        cols     - list of columns to keep in final dataframe
        year_min - start year
        year_max - end year

    :return: dataframe

    :tests:
    >>> urls = ["https://hoopshype.com/nba2k/{}-{}/"]
    >>> cols = ["Player", "Year"]
    >>> year_min = 2013
    >>> year_max = 2014
    >>> res = scraping(urls, cols, year_min, year_max)
    >>> res.shape[0]
    468
    >>> res["Year"].unique()
    array([2014], dtype=object)
    """

    df = pd.DataFrame(columns=cols)

    for i in range(year_min, year_max):
        res = {}
        for j, url in enumerate(urls):
            # get page content
            html = requests.get(url.format(*[i, i+1])).content
            df_list = pd.read_html(html)
            res[j] = df_list[-1]

        if len(urls) > 1:
            join = intersection(res[0].columns, res[1].columns)
            temp = res[0].merge(res[1], how="inner", on=join)
        else:
            temp = res[0].rename(
                {"{}/{}".format(i, str(i+1)[2:]): "Rating"},
                axis=1
            )

        temp["Year"] = i+1
        df = df.append(temp[cols], ignore_index=True)  # append data

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_year")
    parser.add_argument("--max_year")
    parser.add_argument("--source")

    args = parser.parse_known_args()[0]

    if args.source == "2K":
        cols = cols_2K
        urls = urls_2K
    elif args.source == "stats":
        cols = cols_stats
        urls = urls_stats
    else:
        # raise Exception
        cols = None
        urls = None

    logger.info(">>> Scrap data from specified source")
    df = scraping(
        urls,
        cols,
        int(args.min_year),
        int(args.max_year)
    )

    path = "{}/data/raw/{}.csv".format(root, args.source)
    logger.info(">>> Save output to {}".format(path))
    df.to_csv(path, index=False)
