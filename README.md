# Doing Linear Models with Python

An exercise of implementing R's `lm`, `lme`, `glm` and `gam` in Python .

## Usage

```python
from lmpy import lm, data

gala = data('gala', package='faraway')
m = lm('Species ~ Area + Adjacent + Elevation + Nearest + Scruz', gala)
m.summary()
```

See more [examples](./example/).