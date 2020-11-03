# plot basic vertical meltwater distribution

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', size = 14)

# get meltwater data (change filename appropriately)
ds1 = xr.open_dataset("filename.nc")
mw_sim = ds1.meltwater.isel(time=1).mean(dim=("xC","yC")) 
z = mw_sim.zC

## PLOT

plt.figure()
plt.plot(mw_sim,z,linewidth=2)
plt.ylim(-400,-0)
plt.xlabel('Meltwater')

plt.show()
