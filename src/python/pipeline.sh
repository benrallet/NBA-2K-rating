MINYEAR="2013"
MAXYEAR="2021"

# SCRAPING
python scraping.py \
  --source "2K" \
  --min_year "$MINYEAR" \
  --max_year "$MAXYEAR"

python scraping.py \
  --source "stats" \
  --min_year "$MINYEAR" \
  --max_year "$MAXYEAR"

# DATA PREPARATION
python dataprep.py \
  --train "2015,2019" \
  --test "2020,2020" \
  --val "2021,2021"

# MODELING
python modeling.py
