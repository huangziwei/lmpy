# Doing Linear Models with Python

An exercise of implementing R's `lm` interface in Python .

## Dependencies

- Python >=3.10.0
- numpy, scipy, matplotlib, pandas, polars
- formulaic 

## Usage

```
from lmpy import lm, data

gala = data('gala', package='faraway')
m = lm('Species ~ Area + Adjacent + Elevation + Nearest + Scruz', gala)
m.summary()
```

See more [examples](./example/)