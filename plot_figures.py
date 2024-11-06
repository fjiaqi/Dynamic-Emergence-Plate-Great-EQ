# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('./model/')
sys.path.append('./utils/')
from diag_utils import *

params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'RdYlBu',
    'savefig.dpi': 300,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False
}
mpl.rcParams.update(params)

figure_dir = Path('/home1/08524/jqfang/scratch/model_2411/figures')
figure_dir.mkdir(exist_ok=True)

dir_2d = Path('/home1/08524/jqfang/scratch/model_2411/2d')
with h5py.File(dir_2d / 'basic/mesh_0.h5', 'r') as f:
    mesh_2d = f['vertices'][...]

dir_3d = Path('/home1/08524/jqfang/scratch/model_2411/2.5d')
with h5py.File(dir_3d / 'basic/mesh_0.h5', 'r') as f:
    mesh_3d = f['vertices'][...]

core_left, core_right = -100, 200
core_bottom, core_top = -100, 0

dt = 5

# %%
with h5py.File(dir_2d / 'basic/temperature_0.h5', 'r') as f:
    temp_2d = f['data'][...]

fig, ax = plt.subplots(figsize=(16, 3))

ax.set_aspect('equal')
mappable = ax.tricontourf(l_km(mesh_2d[:, 0]), l_km(mesh_2d[:, -1]), T_K(temp_2d.ravel()), 
                          levels=np.linspace(299.9, 1700.1, 101), cmap='inferno')
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_label(r'$T$ (K)')
cbar.set_ticks(np.arange(300, 1701, 350))

ax.plot([core_left, core_right, core_right, core_left, core_left],
        [core_bottom, core_bottom, core_top, core_top, core_bottom], 
        'k', lw=2)

ax.set_xlim(l_km(mesh_2d[:, 0]).min(), l_km(mesh_2d[:, 0]).max())
ax.set_ylim(l_km(mesh_2d[:, -1]).min(), l_km(mesh_2d[:, -1]).max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$z$ (km)')
ax.minorticks_on()

plt.tight_layout()
plt.savefig(figure_dir / 'fig1a.png')

# %%
npz = np.load(dir_2d / 'basic/shear_zone.npz')
top_coords = npz['top']
bottom_coords = npz['bottom']

fig, ax = plt.subplots(figsize=(9, 3))

core_buffer = 5
arg_core = np.where((core_left-core_buffer <= l_km(mesh_2d[:, 0])) & 
                    (l_km(mesh_2d[:, 0]) <= core_right+core_buffer) &
                    (core_bottom-core_buffer <= l_km(mesh_2d[:, -1])) & 
                    (l_km(mesh_2d[:, -1]) <= core_top+core_buffer))[0]
mappable = ax.tricontourf(l_km(mesh_2d[arg_core, 0]), l_km(mesh_2d[arg_core, -1]), T_K(temp_2d[arg_core, 0]), 
                          levels=np.linspace(299.9, 1700.1, 101), cmap='inferno')
ax.set_aspect('equal')

ax.plot(l_km(top_coords[:, 0]), l_km(top_coords[:, 1]), c='silver', ls='--', lw=2)
ax.plot(l_km(bottom_coords[:, 0]), l_km(bottom_coords[:, 1]), c='silver', ls='--', lw=2)

cs = ax.tricontour(l_km(mesh_2d[arg_core, 0]), l_km(mesh_2d[arg_core, -1]), T_K(temp_2d[arg_core, 0]), 
              colors='w', levels=[373., 423., 623., 723.])
plt.clabel(cs, fontsize=10, fmt='%d K', inline_spacing=-5)

ax.set_xlim(core_left, core_right)
ax.set_ylim(core_bottom, core_top)

ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$z$ (km)')
ax.minorticks_on()

plt.tight_layout()
plt.savefig(figure_dir / 'fig1b.png')

#%%
npz = np.load(dir_2d / f'basic/vars_0.npz')
loc = npz['loc']
vars = npz['vars']
material = vars[:, 0]

material_idx = 3
arg_sz = np.where(material == material_idx)[0]

bins = -np.arange(0, 101, 2.5)
var_arr = np.zeros(len(bins)-1)

bin_var = T_K(vars[arg_sz, 1])

for i in range(len(bins)-1):
    arg_bin = np.where((l_km(loc[arg_sz, -1]) > bins[i+1]) & (l_km(loc[arg_sz, -1]) <= bins[i]))[0]
    var_arr[i] = bin_var[arg_bin].mean()

fig, ax = plt.subplots(figsize=(6, 3))

ax.plot([300, 1000], [0.6, 0.6], label=r'$\mu_{st}$', lw=3)
ax.plot([300, 373., 423., 623., 723., 1000.], [0.8, 0.8, 0.2, 0.2, 0.8, 0.8], label=r'$\mu_{dyn}$', ls='--', lw=3)

ax.set_xlim(300, 900)
ax.set_yticks([0.3, 0.6])
ax.set_xlabel(r'$T$ (K)')
ax.set_ylabel(r'$\mu$')
ax.legend(loc='lower right', fontsize=12)
ax.minorticks_on()
ax.set_yticks([0.2, 0.4, 0.6, 0.8])

def forward(T):
    return -np.interp(T, var_arr, -(bins[:-1]+bins[1:])/2)

def inverse(z):
    return np.interp(-z, -(bins[:-1]+bins[1:])/2, var_arr)

ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))
ax2.set_xlabel(r"$z$ (km)")
ax2.set_ticks(np.arange(-80, 1, 20))
ax2.minorticks_on()

plt.tight_layout()
plt.savefig(figure_dir / 'fig1c.png')

# %%
npz = np.load(dir_2d / f'viscosity/viscosity.npz')
loc = npz['loc']
vars = npz['vars']

fig, ax = plt.subplots(figsize=(16, 3))

mappable = ax.tricontourf(l_km(loc[:, 0]), l_km(loc[:, -1]), logeta_Pa_s(vars[:, 1]),
                          levels=np.linspace(16.99, 24.01, 101), cmap='RdYlBu')
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_label(r'$\log_{10} \eta$ (Pa s)')
cbar.set_ticks(np.arange(18, 24.01, 2))
ax.set_aspect('equal')

ax.set_xlim(l_km(mesh_2d[:, 0]).min(), l_km(mesh_2d[:, 0]).max())
ax.set_ylim(l_km(mesh_2d[:, -1]).min(), l_km(mesh_2d[:, -1]).max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$z$ (km)')
ax.minorticks_on()

plt.tight_layout()
plt.savefig(figure_dir / 'fig1d.png')

# %%
step_start = 100
step_end = 501
nsteps = step_end - step_start
step_arr = np.arange(step_start, step_end)
time_arr = (step_arr - step_start) * dt
slip_vel_arr = np.zeros(nsteps)

bins = -np.arange(0, 101, 5)
var_z_interp = -np.arange(0, 100.1, 0.5)
bin_slipv_mean_arr = np.zeros((len(step_arr), len(bins)-1))
bin_slipv_mean_arr_interp = np.zeros((len(step_arr), len(var_z_interp)))
bin_slipv_max_arr = np.zeros((len(step_arr), len(bins)-1))
bin_slipv_max_arr_interp = np.zeros((len(step_arr), len(var_z_interp)))
bin_stress_arr = np.zeros((len(step_arr), len(bins)-1))
bin_stress_arr_interp = np.zeros((len(step_arr), len(var_z_interp)))

for istep in tqdm(range(len(step_arr))):
    npz = np.load(dir_2d / f'slipv_stress/vars_s{step_arr[istep]}.npz')
    loc = npz['loc']
    vars = npz['vars']

    bin_slipv = v_cm_yr(vars[:, 0])
    bin_stress = tau_MPa(vars[:, 1])

    slip_vel_arr[istep] = np.mean(bin_slipv) / 100.

    for i in range(len(bins)-1):
        arg_bin = np.where((l_km(loc[:, -1]) > bins[i+1]) & (l_km(loc[:, -1]) <= bins[i]))[0]
        bin_slipv_mean_arr[istep, i] = bin_slipv[arg_bin].mean()
        bin_slipv_max_arr[istep, i] = bin_slipv[arg_bin].max()
        bin_stress_arr[istep, i] = bin_stress[arg_bin].mean()

    bin_slipv_mean_arr_interp[istep, :] = np.interp(-var_z_interp, -(bins[:-1]+bins[1:])/2, bin_slipv_mean_arr[istep, :])
    bin_slipv_max_arr_interp[istep, :] = np.interp(-var_z_interp, -(bins[:-1]+bins[1:])/2, bin_slipv_max_arr[istep, :])
    bin_stress_arr_interp[istep, :] = np.interp(-var_z_interp, -(bins[:-1]+bins[1:])/2, bin_stress_arr[istep, :])

arg_surf = np.where(np.isclose(mesh_2d[:, -1], 0))[0]

vel_arr = np.zeros((len(step_arr), len(arg_surf)))
for istep in tqdm(range(len(step_arr))):
    with h5py.File(dir_2d / f'velocity/velocity_s{step_arr[istep]}.h5', 'r') as f:
        vel_2d = f['data'][...]

    vel_arr[istep, :] = v_cm_yr(vel_2d[arg_surf, 0])
disp_arr = np.cumsum(vel_arr, axis=0) * dt/1e2

# %%
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

ax = axes[0]

step_list = [102, 117, 120]
label_arr = ['Interseismic', 'Coseismic', 'Postseismic']

steplim1, steplim2 = step_start, step_end

ax.plot(l_km(mesh_2d[arg_surf, 0]), vel_arr[step_list[0]-step_start], 
        c='b', lw=3, label=label_arr[0])
ax.plot([], [], c='r', lw=3, label=label_arr[1])

ax.plot(l_km(mesh_2d[arg_surf, 0]), 
        np.sum(vel_arr[steplim1-step_start:steplim2-step_start], axis=0)/(steplim2-steplim1-1),
        c='k', ls='--', lw=3, label='Average')

axt = ax.twinx()
axt.plot(l_km(mesh_2d[arg_surf, 0]), vel_arr[step_list[1]-step_start], c='r', lw=3)

ax.set_zorder(axt.get_zorder()+1)
ax.set_frame_on(False)
ax.set_xlim(l_km(mesh_2d[arg_surf, 0]).min(), l_km(mesh_2d[arg_surf, 0]).max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='b')
ax.legend(loc='lower left', fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax.minorticks_on()
axt.set_ylabel(r'$v_x$ (cm/yr)', c='r')
axt.tick_params(axis='y', labelcolor='r')
axt.minorticks_on()

ax = axes[1]

ax.plot(var_z_interp, bin_slipv_mean_arr_interp[step_list[0]-step_start], label=label_arr[0], c='b', lw=3)
ax.plot([], [], label=label_arr[1], c='r', lw=3)

axt = ax.twinx()
axt.plot(var_z_interp, bin_slipv_mean_arr_interp[step_list[1]-step_start]*dt/1e2, c='r', lw=3)

ax.set_zorder(axt.get_zorder()+1)
ax.set_frame_on(False)
ax.set_xlim(-2.5, -80)
ax.set_ylim(0, 6)
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$V$ (cm/yr)', c='b')
ax.tick_params(axis='y', labelcolor='b')
ax.minorticks_on()
axt.set_ylabel(r'$U$ (m)', c='r')
axt.tick_params(axis='y', labelcolor='r')
axt.minorticks_on()

ax = axes[2]

ax.plot(time_arr, slip_vel_arr, lw=3, c='teal')
ax.set_ylabel(r'$V_{mean}$ (m/yr)', c='teal')
ax.set_xlim(time_arr.min(), time_arr.max())
ax.minorticks_on()
ax.set_xlabel(r'$t$ (year)')
ax.set_xticks(np.arange(0, 2001, 250))

axt = ax.twinx()
obs_loc = np.array([-100, 100])
arg_upper = np.where(l_km(mesh_2d[arg_surf, 0]) > 0.)[0]
arg_lower = np.where(l_km(mesh_2d[arg_surf, 0]) <= 0.)[0]
arg_obs = np.argmin(np.abs(l_km(mesh_2d[arg_surf, 0]) - obs_loc[0]), axis=0)
axt.plot(time_arr, disp_arr[:, arg_obs]-np.mean(disp_arr[:, arg_lower], axis=1), 
        'm', label=f'Subduction', lw=3, alpha=0.75)
arg_obs = np.argmin(np.abs(l_km(mesh_2d[arg_surf, 0]) - obs_loc[1]), axis=0)
axt.plot(time_arr, disp_arr[:, arg_obs]-np.mean(disp_arr[:, arg_upper], axis=1), 
        'm--', label=f'Overriding', lw=3, alpha=0.75)
axt.set_ylabel(r'$\Delta u_x$ (m)', c='m')
axt.minorticks_on()

ax = axes[3]

mappable = ax.pcolormesh(time_arr, var_z_interp, (bin_stress_arr_interp-np.mean(bin_stress_arr_interp, axis=0)).T,
                         cmap='RdBu', shading='auto', vmin=-5, vmax=5)
ax.set_xticks(np.arange(0, 2001, 250))
ax.set_xlabel(r'$t$ (year)')
ax.set_ylabel(r'$z$ (km)')
ax.minorticks_on()

cbax = fig.add_axes([0.92, 0.07, 0.01, 0.17])
cbar = fig.colorbar(mappable, cax=cbax, label=r'$\Delta \tau_{II}$ (MPa)')

plt.tight_layout()
plt.savefig(figure_dir / 'fig2.png')

# %%
def extract_surf_vel(step, l=''):
    if(step == 0):
        with h5py.File(Path(dir_3d) / f'velocity/velocity_s{step}.h5', 'r') as f:
            vel_3d = f['data'][...]
    else:
        with h5py.File(Path(dir_3d) / f'velocity/velocity_s{step}_L{l}.h5', 'r') as f:
            vel_3d = f['data'][...]

    arg_surf = np.where(np.isclose(mesh_3d[:, -1], 0.0) &
                        np.isclose(mesh_3d[:, 1], 0))[0]
    surf_loc = mesh_3d[arg_surf, 0]
    surf_vel = vel_3d[arg_surf, 0]

    return surf_loc, surf_vel

def extract_slip_vel(step, l=''):
    if(step == 0):
        npz = np.load(Path(dir_3d)  / f'slipv_stress/vars_s{step}.npz')
    else:
        npz = np.load(Path(dir_3d)  / f'slipv_stress/vars_s{step}_L{l}.npz')
    loc = npz['loc']
    vars = npz['vars']

    yres = 2
    domain_length = mesh_3d[:, 1].max()
    wall_y = 0./yres * domain_length
    arg_wall = np.where(np.isclose(loc[:, 1], wall_y, atol=0.5/yres*domain_length))[0]

    bins = -np.arange(0, 101, 5)
    var_arr = np.zeros(len(bins)-1)

    for j in range(len(bins)-1):
        arg_bin = np.where((l_km(loc[arg_wall, -1]) > bins[j+1]) & (l_km(loc[arg_wall, -1]) <= bins[j]))[0]
        var_arr[j] = v_mm_yr(vars[arg_wall, 0][arg_bin]).mean()

    return bins, var_arr

dL = 5.0

def L2Ltilde(L):
    return np.sqrt(dL*L)

def Ltilde2L(Ltilde):
    return Ltilde**2/dL

# %%
fmt = mticker.LogFormatterSciNotation(base=10)

l_list = ['1e+50', '2e+08', '2e+07', '2e+06', '2e+05', '2e+04']
color_list = ['C5', 'C6', 'C4', 'C3', 'C1', 'C0']
label_list = [f'{L2Ltilde(float(l_list[i])/1e3):.0f}' for i in range(len(l_list))]
label_list[0] = r'+$\infty$'
dt = 5

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

ax = axes[0]

for i in range(len(l_list)):
    surf_loc, surf_vel = extract_surf_vel(1, l_list[i])
    ax.plot(l_km(surf_loc), v_mm_yr(surf_vel)*dt/1e3, label=r'$\tilde{L}$'+f' = {label_list[i]:s} km', lw=2, c=color_list[i])

ax.set_xlim(l_km(surf_loc[0]), l_km(surf_loc[-1]))
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$u_x$ (m)')
ax.grid()
ax.minorticks_on()

ax = axes[1]

for i in tqdm(range(len(l_list))):
    bins, var_arr = extract_slip_vel(1, l_list[i])
    ax.plot((bins[:-1]+bins[1:])/2, var_arr*dt/1e3, label=r'$\tilde{L}$'+f' = {label_list[i]:s} km', lw=2, c=color_list[i])

ax.set_xlim(-2.5, -80)
ax.grid()
ax.minorticks_on()

ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(figure_dir / 'fig3ab.png')

# %%
l_list = ['1e+50', '2e+08', '2e+07', '2e+06', '2e+05', '2e+04']
title_list = ['C', 'D', 'E', 'F', 'G', 'H']
label_list = [f'{L2Ltilde(float(l_list[i])/1e3):.0f}' for i in range(len(l_list))]
label_list[0] = r'+$\infty$'

vlim = 5

fig, axes = plt.subplots(2, 3, figsize=(12, 3), sharex=True, sharey=True)

for i in range(6):
    ax = axes.flatten()[i]
    ax.set_aspect('equal')
    with h5py.File(dir_3d / f'velocity/velocity_s1_L{l_list[i]}.h5', 'r') as f:
        vel_3d = f['data'][...]
        
    core_buffer = 5
    arg_frontcore = np.where((np.isclose(mesh_3d[:, 1], 0.)) &
                         (core_left-core_buffer <= l_km(mesh_3d[:, 0])) & 
                        (l_km(mesh_3d[:, 0]) <= core_right+core_buffer) &
                        (core_bottom-core_buffer <= l_km(mesh_3d[:, -1])) & 
                        (l_km(mesh_3d[:, -1]) <= core_top+core_buffer))[0]

    mappable = ax.tricontourf(l_km(mesh_3d[arg_frontcore, 0]), l_km(mesh_3d[arg_frontcore, -1]), 
                            np.linalg.norm(v_mm_yr(vel_3d[arg_frontcore]), axis=1)*dt/1e3, 
                            np.linspace(0, vlim, 101), cmap='RdYlBu_r', extend='max')

    ax.plot(l_km(top_coords[:, 0]), l_km(top_coords[:, 1]), c='silver', ls='--', lw=2)
    ax.plot(l_km(bottom_coords[:, 0]), l_km(bottom_coords[:, 1]), c='silver', ls='--', lw=2)
    
    ax.text(-90, -90, r'$\tilde{L}$'+f' = {label_list[i]:s} km', fontsize=12,
            ha='left', va='bottom', c='k',
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, boxstyle='round'))

for ax in axes[:, 0]:
    ax.set_ylabel(r'$z$ (km)')
for ax in axes[-1, :]:
    ax.set_xlabel(r'$x$ (km)')
ax.set_xlim(core_left, core_right)
ax.set_ylim(core_bottom, core_top)
ax.minorticks_on()

cbax = fig.add_axes([0.92, 0.15, 0.01, 0.65])
cbar = fig.colorbar(mappable, cax=cbax, aspect=30)
cbar.ax.set_title(r'$u$ (m)', loc='left')
cbar.set_ticks(np.arange(0, vlim+0.01, 1))

plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.2, wspace=0.2)
plt.savefig(figure_dir / 'fig3c.png')

# %%
fmt = mticker.LogFormatterSciNotation(base=10)

l_list = ['1e+10', '5e+09', '2e+09', '1e+09', '5e+08', '2e+08', '1e+08', '5e+07', '2e+07', '1e+07', '5e+06', '2e+06', '1e+06', '5e+05', '2e+05', '1e+05', '5e+04', '2e+04', '1e+04']
l_arr = np.zeros(len(l_list))
max_surf_vel_arr = np.zeros(len(l_list))
max_slip_vel_arr = np.zeros(len(l_list))
dt = 5

for i in tqdm(range(len(l_list))):
    surf_loc, surf_vel = extract_surf_vel(1, l_list[i])
    l_arr[i] = float(l_list[i])/1e3
    max_surf_vel_arr[i] = v_mm_yr(surf_vel)[(l_km(surf_loc)<-20)&(l_km(surf_loc)>-100)].max()
    bins, var_arr = extract_slip_vel(1, l_list[i])
    max_slip_vel_arr[i] = var_arr.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

ax = axes[0]
ax.plot(L2Ltilde(l_arr), max_surf_vel_arr*dt/1e3, '-o', lw=3)

surf_loc, surf_vel = extract_surf_vel(0)
ax.axhline(v_mm_yr(surf_vel)[(l_km(surf_loc)<-20)&(l_km(surf_loc)>-100)].max()*dt/1e3, c='k', ls='--', lw=2)
ax.text(L2Ltilde(l_arr[0]), 0.02+v_mm_yr(surf_vel)[(l_km(surf_loc)<-20)&(l_km(surf_loc)>-100)].max()*dt/1e3, r'Before the event', ha='right', va='bottom')
surf_loc, surf_vel = extract_surf_vel(1, '1e+50')
ax.axhline(v_mm_yr(surf_vel)[(l_km(surf_loc)<-20)&(l_km(surf_loc)>-100)].max()*dt/1e3, c='k', ls='--', lw=2)
ax.text(L2Ltilde(l_arr[-1]), -0.03+v_mm_yr(surf_vel)[(l_km(surf_loc)<-20)&(l_km(surf_loc)>-100)].max()*dt/1e3, r'$\tilde{L}$$\rightarrow$+$\infty$', ha='left', va='top')

ax.set_xlabel(r'$\tilde{L}$ (km)')
ax.set_ylabel(r'$u_{x}$ (m)')
ax.set_xscale('log')
ax.minorticks_on()

ax = axes[1]
ax.plot(L2Ltilde(l_arr), max_slip_vel_arr*dt/1e3, '-o', lw=3)

bins, var_arr = extract_slip_vel(0)
ax.axhline(var_arr.max()*dt/1e3, c='k', ls='--', lw=2)
ax.text(L2Ltilde(l_arr[0]), var_arr.max()*dt/1e3+0.1, r'Before the event', ha='right', va='bottom')
bins, var_arr = extract_slip_vel(1, '1e+50')
ax.axhline(var_arr.max()*dt/1e3, c='k', ls='--', lw=2)
ax.text(L2Ltilde(l_arr[-1]), var_arr.max()*dt/1e3-0.15, r'$\tilde{L}$$\rightarrow$+$\infty$', ha='left', va='top')

ax.set_xlabel(r'$\tilde{L}$ (km)')
ax.set_ylabel(r'$U_{max}$ (m)')
ax.set_xscale('log')
ax.minorticks_on()

plt.tight_layout()
plt.savefig(figure_dir / 'fig4.png')
# %%
