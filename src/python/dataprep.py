import argparse
import logging
import os
import pandas as pd
import numpy as np
import unicodedata

# Logging configuration
MSG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=MSG_FORMAT, datefmt=DATETIME_FORMAT)
logger = logging.getLogger("2K RATING")
logger.setLevel(logging.INFO)

# Global variables
root = os.path.abspath(os.path.join(os.curdir, "../.."))


def clean_name(df):
    """
    Return the string without accents

    :params:
        s - string
        l2 - list 2

    :return: string

    :tests:
    >>> data = [
    ...     {'Player':'Christapher Besuk'},
    ...     {'Player':'Tony Parker'},
    ...     {'Player':'Kobe Bryant*'},
    ...     {'Player':'Théo Maledon'}
    ... ]
    >>> df = pd.DataFrame.from_dict(data, orient='columns')
    >>> clean_name(df)
             Player
    0   Chris Besuk
    1   Tony Parker
    2   Kobe Bryant
    3  Theo Maledon
    """
    df["Player"] = (df["Player"]
                    .apply(lambda x: _strip_accent(x))
                    .apply(lambda x: _replace_name(x))
                    .str
                    .replace("*", "")
                   )

    return df[df["Player"] != "Player"]


def _replace_name(s):
    """
    Return the string with specific name replacements

    :params:
        s - string

    :return: string

    :tests:
    >>> s = "Christapher Besuk"
    >>> _replace_name(s)
    'Chris Besuk'
    """
    return s.replace("'", "")\
            .replace(".", "")\
            .replace(" Jr", "")\
            .replace("Christapher", "Chris")\
            .replace("Schroeder", "Schroder")\
            .replace("Ishmael", "Ish")


def _strip_accent(s):
    """
    Return the string without accents

    :params:
        s - string

    :return: string

    :tests:
    >>> s = "Ömer Aşık"
    >>> _strip_accent(s)
    'Omer Asık'
    """
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s


def valid_period(train, test, val):
    """
    Return the intersection of two lists

    :params:
        train - array with start and end of the training period
        test - array with start and end of the test period
        val - array with start and end of the validation period

    :return: bool

    :tests:
    >>> valid_period([2014, 2012], [2016, 2018], [2019, 2019])
    The end of the period is before the start of the period.
    False
    >>> valid_period([2012, 2017], [2016, 2018], [2019, 2019])
    The training set must be strictly before the test set, that is striclty before the validation set.
    False
    >>> valid_period([2012, 2014], [2016, 2018], [2019, 2019])
    True
    """
    for p in [train, test, val]:
        # test if first number is always under second number
        if p[0] > p[1]:
            print("The end of the period is before the start of the period.")
            return False

    if train[1] < test[0] and test[1] < val[0]:
        return True

    print("The training set must be strictly before the test set, that is striclty before the validation set.")
    return False


def string_to_list(s):
    """
    Transform a string with int delimited by comma to a list of int

    :params:
        s - string

    :return: array of int

    :tests:
    >>> s = "2018,2010"
    >>> string_to_list(s)
    [2018, 2010]
    """
    return [int(a) for a in s.split(',')]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--val")

    args = parser.parse_known_args()[0]

    logger.info(">>> Read raw data")
    ratings = pd.read_csv("{}/data/raw/2K.csv".format(root))
    stats = pd.read_csv("{}/data/raw/stats.csv".format(root))

    logger.info(">>> Clean player names")
    ratings_cln = clean_name(ratings)
    stats_cln = clean_name(stats)

    logger.info(">>> Merge the two dataframes")
    df = (stats_cln
          .merge(ratings_cln, how="inner", on=["Player", "Year"])
          .drop_duplicates(subset=["Player", "Year"])
          .fillna(0)
         )

    logger.info(">>> Feature engineering")
    # Position binarization
    for pos in ["SG", "PF", "PG", "C", "SF"]:
        df[pos] = np.where(df["Pos"].str.find(pos) != -1, 1, 0)

    # Normalization by game played
    df["%GS"] = df["GS"].astype(int)/df["G"].astype(int)
    df["%MPPG"] = df["MP"].astype(int)/df["G"].astype(int)

    # Normalization by minute played
    for c in ['FG', '3P', '2P', 'FT', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']:
        name = c + "PM"
        df[name] = df[c].astype(int)/df["MP"].astype(int)

    logger.info(">>> Split data into train, test and validation datasets and save outputs")
    train = string_to_list(args.train)
    test = string_to_list(args.test)
    val = string_to_list(args.val)

    if valid_period(train, test, val):
        for p in ["train", "test", "val"]:
            temp = df[(df["Year"] >= eval(p)[0]) & (df["Year"] <= eval(p)[1])]
            temp.to_csv("{}/data/input/df_{}.csv".format(root, p), index=False)

    df.to_csv("{}/data/input/df.csv".format(root), index=False)
