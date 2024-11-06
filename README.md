# Dynamic-Emergence-Plate-Great-EQ

Code for our paper:

Dynamic Emergence of Plate Motions and Great Megathrust Earthquakes Across Length and Time Scales, by Jiaqi Fang, Michael Gurnis and Nadia Lapusta.

## Dependencies

* Python 3.7
* Underworld 2.10.0b
* Shapely 2.0.2

We expect the code to work well for other recent versions.

## How to run

2D model:
```
mpirun -n 64 python main.py
```

For the 2.5D model without along-strike resistance, change `yres` in `main.py` from 0 to 2, and run:
```
mpirun -n 16 python main.py
```

For the 2.5D model with along-strike resistance controlled by characteristic length scale `L`, we start from an initial condition from the 2.5D model without along-strike resistance (before a rupture event). Set the value of `L` (e.g., 1e10) and the smoothing factor $\phi$ (e.g., 0.5) in `along_strike_resist`, and run:
```
mpirun -n 16 python along_strike_resist.py 1e10 0.5
```

With model results, all the figures in the paper can be reproduced using `plot_figures.py`.
