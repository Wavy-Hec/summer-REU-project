# Dataset

The analysis runs on **Highway-Rail Grade Crossing Accident Data** (FRA Form
6180.57) — every reported impact between railroad on-track equipment and a
highway user at a US highway-rail grade crossing.

The CSV is not committed here. It is a large, freely available public dataset,
and keeping it out of version control keeps the repository small and the data
current.

## Download

Get the human-readable export from the US DOT open data portal:

**<https://data.transportation.gov/Railroads/Highway-Rail-Grade-Crossing-Accident-Data/7wn6-i5b9>**

Export as CSV and save it here as:

```
data/Highway-Rail_Grade_Crossing_Accident_Data.csv
```

If you would rather keep it elsewhere, point `REU_DATA_PATH` at the file:

```bash
export REU_DATA_PATH=/path/to/Highway-Rail_Grade_Crossing_Accident_Data.csv
```

`src/config.py` checks the environment variable first, then falls back to the
path above.

## Columns the scripts rely on

| Column | Used by |
| --- | --- |
| `Railroad Code` | every script — it is the node identity throughout |
| `County Code` | the county co-occurrence graphs and the bipartite view |
| `Report Year` / `Incident Year` | the yearly EDA and the PageRank accident keys |
| `Weather Condition` / `Weather Condition Code` | the weather view |
| `Vehicle Damage Cost` | the weighted damage-cost graph and both clusterings |
| `Total Killed Form 57`, `Total Killed Form 55A`, `Total Injured Form 57`, `Total Injured Form 55A` | the casualty graph |

The official data dictionary for Form 57 is linked from the dataset page above.

## Related sources

- [FRA Safety Data portal](https://safetydata.fra.dot.gov/) — full datasets and PDF reports
- [Raw Form 57 source table](https://data.transportation.gov/dataset/Form57-Source-Table/icqf-xf4w) — unlabelled raw codes
