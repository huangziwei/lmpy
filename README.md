# hea: Linear Models with Python

> hea v. (Cantonese) 
> 1. to kill time; to hang around  
> 2. to do something without putting much care or effort into it. 
> 
> This project started as an laid-back exercise of implementing R's `lm`, `lme`, `glm` and `gam` in Python, and later evolved into my private benchmark for coding agents, and it finally kind of working after trying with Opus 4.7.

## Usage

```python
from hea import lm, data

gala = data('gala', package='faraway')
m = lm('Species ~ Area + Adjacent + Elevation + Nearest + Scruz', gala)
m.summary()
```

See more [examples](./example/).