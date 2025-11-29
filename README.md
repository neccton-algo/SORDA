🚀 SORDA
Spectral Ocean Reflectance Data Assimilation

A NECCTON WP4 demonstration of data assimilation with new satellite ocean surface reflectance observations

📘 Overview

SORDA is a reproducible demonstration of an Ensemble Kalman Filter (EnKF) data assimilation workflow using the OAK library applied to Black Sea biogeochemistry. This demonstration kit contains 2016 data from an free-running ensemble of NEMO+BAMHBI+RADTRANS model configuration. 

The system assimilates satellite-derived ocean reflectance (RRS) into biogeochemical model fields (BAMHBI PFTs, POC, Zoo-plakton and gelatinious) and evaluates the impacts using MAE, CRPS, and increment diagnostics.

This project is part of NECCTON WP4, showcasing the assimilation of novel observational datasets.

## 📦 Dependencies

This project contains two major components with different dependencies:

1. **Python workflow**  
2. **Fortran (OAK) executable**

Both must be available for running the full SORDA pipeline.

---

## 1. Python Environment

All Python code relies on a minimal scientific stack and can be installed via:

```bash
pip install -r requirements.txt
```

### Required Python packages (tested versions)

```text
python >= 3.10
numpy >= 1.26
xarray >= 2023.6
matplotlib >= 3.7
cartopy >= 0.22
netCDF4 >= 1.6
pandas >= 2.0
properscoring >= 0.1
tqdm >= 4.65
```

*All plotting and post-processing routines are pure Python and work on Linux/MacOS.*

---

## 2. Fortran (OAK) Environment

OAK is a **Fortran + MPI ensemble Kalman filter** system.  
It compiles cleanly on most HPC clusters but requires several scientific libraries.

### Essential toolchain

```text
GFortran >= 10
GCC >= 10
MPI implementation (OpenMPI >= 4 or MPICH)
```

### Required scientific libraries

```text
netCDF-Fortran (>= 4.5)   – reading/writing model input and output
BLAS / LAPACK             – linear algebra backend
ScaLAPACK                 – distributed matrix operations (EnKF update)
UDUNITS                   – physical unit handling
X11                       – optional diagnostics / plotting
SuiteSparse               – optional but recommended
Boost                     – optional; some builds require it
```

### Version flexibility

OAK has been successfully built with **multiple module stacks**.  
Both of these example setups are valid:

#### Example modern HPC stack (2023)

```text
netCDF-Fortran/4.6
OpenBLAS/0.3
ScaLAPACK/2.2
UDUNITS/2.2
Boost/1.82        (optional)
OpenMPI/4.1
GCC/GFortran 12.x
```

#### Example older HPC stack (2020)

```text
netCDF-Fortran/4.5
SuiteSparse
Perl/5.32
Boost/1.74
GCC/GFortran 10.x
OpenMPI 4.0
```

These combinations were **verified** with the SORDA workflow.

---

## 3. Executable Location

The assimilation driver expects:

```
current_day/assim-gfortran-double-mpi
```

This must be built from the OAK repository and placed (or symlinked) inside `current_day/` before running the pipeline.


# Data Sources

In this demonstration kit SORDA uses only model-produced data.  
All inputs are external and not produced within this repository.  
They are required to reproduce the Black Sea assimilation demonstration for February 2016.

## 1. Satellite Observations (REF)
Daily S3-like reflectance fields derived from deterministic model run.  
Used as the observation vector in the assimilation system.

- Directory: `data/dynamic/obs/`
- File pattern:
  
  CREF_yYYYYmMMdDD.nc

- Variables (2D or 3D depending on band):
  - `RRS_412nm`, `RRS_443nm`, `RRS_490nm`, `RRS_510nm`, `RRS_555nm`, `RRS_670nm`
- Dimensions: `(y, x)` or `(depth, y, x)`


## 2. Ensemble Model Optical Fields (“model” folder)
These are **ensemble-simulated reflectances and optical quantities**, used to
estimate **model–observation error statistics**.

- Directory: `data/dynamic/model/`
- File pattern:

  C0XX_yYYYYmMMdDD.nc

- Variables (example):
  - `RRS412`, `RRS443`, `RRS490`, `RRS510`, `RRS555`, `RRS670`
- Dimensions: `(ensemble_member, y, x)` or `(ensemble_member, depth, y, x)`
- Role in SORDA:
  - Provides the **observation-space ensemble** for the EnKF


## 3. PTRC Biogeochemical Ensemble (“ptrc” folder)
These are the **actual biogeochemical model state variables**, i.e. the
quantities that the EnKF **updates**.  
These *do* contain CDI, CEM, CFL, MES, MIC, GEL, POC, etc.

- Directory: `data/dynamic/ptrc/`
- File pattern:

  PC0XX_yYYYYmMMdDD.nc

- Variables:
  - `CDI`, `CEM`, `CFL`, `MES`, `MIC`, `GEL`, `POC`
- Dimensions: `(ensemble, depth, y, x)`
- Role in SORDA:
  - Provides the **background (forecast)** ensemble for the EnKF update
  - Produces updated analysis ensemble after assimilation

## 4. Static Model Files

Essential grid geometry and mask files required by OAK.

- `domain_cfg.nc`
- `mesh_mask.nc`
- `gridZ.nc`
- `domain_part.nc`
- `REF_stds.nc` or `REF_stds_YYYYMM.nc`

These are contained in the main working directory (`current_day/`) and are not to be removed or moved. 

## Data Access: Zenodo + demo summaries to plot on GitHub

All input and output NetCDF files required to reproduce the full SORDA
demonstration are archived on Zenodo:

> **DOI:** `10.5281/zenodo.17727180`

User can download the data manually from the Zenodo web interface, or via
the command line. 
In the ./data/dynamic folder download raw input data to try the assimilation:

```bash
wget https://zenodo.org/records/17727180/files/model.tar.gz
wget https://zenodo.org/records/17727180/files/obs.tar.gz
wget https://zenodo.org/records/17727180/files/ptrc.tar.gz

tar -xvzf *.tar.gz
```
To use pre-computed assimilation analysis, forecast and increment files for plotting, 
download SORDA output data to the root directory (./):
```bash
wget https://zenodo.org/records/17727180/files/SORDA_output.tar.zst

tar --use-compress-program=unzstd -xf SORDA_output.tar.zst
```

### MAE and CRPS summaries are available to be used for plotting in ./demo_summaries/ folder (see below for SORDA_summaries_plotting.ipynb). 

## Repository layout and data locations

After cloning the repo and data download a typical directory structure is:

```text
SORDA/
├── README.md
├── environment.yml
├── requirements.txt
├── assim.date
├── clean_all.sh
├── WP4_demo_SORDA.ipynb
├── SORDA_summaries_plotting.ipynb
├── src/
│   └── python/
│       ├── prep.py
│       ├── month_loop.py
│       ├── job.py
│       ├── plotting.py
│       └── summaries.py
├── data/
│   ├── static/              # small, versioned grid/config files
│   │   ├── domain_cfg.nc
│   │   ├── domain_part.nc
│   │   ├── gridZ.nc
│   │   ├── mesh_mask.nc
│   │   └── REF_stds.nc
│   └── dynamic/             # initially empty (only .gitkeep)
│       ├── model/           # unpack model.tar.gz here
│       ├── obs/             # unpack obs.tar.gz here
│       └── ptrc/            # unpack ptrc.tar.gz here
├── demo_summaries/          # small pre-computed summaries (tracked)
│   ├── mae_summary.nc
│   ├── crps_summary.nc
│   └── increments_combined.nc
├── analysis/                # runtime analysis outputs (NetCDF) – ignored by git
│   └── .gitkeep
├── forecast/                # runtime forecast outputs (NetCDF) – ignored by git
│   └── .gitkeep
├── current_day/             # working directory for daily OAK runs – ignored by git
│   └── .gitkeep
└── SORDA_output/            # optional: unpack SORDA_output.tar.zst here
    └── .gitkeep
```

# 🔧 Reproducible Workflow (Jupyter Notebook)

A complete, end-to-end demonstration of the SORDA system is provided in the Jupyter notebook:

**`WP4_demo_RRSDA.ipynb`**  

**Before running the assimilation pipeline,you need to fix path to SORDA in the assim.date file!**

```text
Config.exec = '/path/to/SORDA/current_day/'
ErrorSpace.path  = '/path/to/SORDA/current_day/'
Zones.path = '/path/to/SORDA/current_day/'
Obs001.path      = '/path/to/SORDA/current_day/'

``` 
This notebook allows the user to run the entire February 2016 Black Sea demonstration experiment, including:

- Preparing daily observation and ensemble input files  
- Executing the OAK Ensemble Kalman Filter cycle  
- Producing combined forecast, analysis, increment, and reference files  
- Visualising diagnostics:  
  - MAE Hovmöller diagrams  
  - Global MAE metrics  
  - CRPS time series  
  - CRPS at selected depths  
  - Spatial and vertical-section increment plots  
- Inspecting and analysing the skill of the assimilation system  

The notebook is fully reproducible, self-contained, and can be adapted to new date ranges, domains, or biogeochemical variables.

### ℹ About OAK (Ensemble Kalman Filter system)

The data assimilation step uses the **OAK Ensemble Kalman Filter**, originally developed at GHER of ULiege.  
Its public source code is available here:

👉 **https://github.com/gher-uliege/OAK**

In this repository, OAK is used as the core of EnKF update mechanism.

---

# 📊 Metrics for Validation

To evaluate the performance of the assimilation system, SORDA computes several widely used metrics for ensemble-based ocean data assimilation. These metrics quantify the improvement of the analysis compared to the free forecast.

### **1. Mean Absolute Error (MAE)**

MAE measures the average magnitude of model–observation differences:

$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |x_i - y_i|$

In this project, MAE is computed in several forms:

- **MAE(time, depth)** — Hovmöller diagrams of error evolution  
- **MAE(time)** — domain- and depth-averaged error  
- **MAE(depth)** — mean error profiles over the entire period

These help diagnose whether assimilation reduces biases at specific depths or time periods.

---

### **2. CRPS (Continuous Ranked Probability Score)**

CRPS is a probabilistic metric designed for ensemble systems.  
It measures how well the forecast distribution matches the observed value:

- **Lower CRPS is better**  
- Reduces to MAE when the ensemble degenerates to a single member  
- Sensitive to ensemble spread → evaluates uncertainty

We compute:

- **CRPS(time)** — domain-averaged forecast vs analysis comparison  
- **CRPS(time @ depth)** — CRPS at selected biogeochemical depth levels

This metric is more informative than MAE when dealing with ensembles.

---

### **3. Assimilation Increments**

Although not a metric, increments provide a **diagnostic tool**:

$\delta x = x^\text{analysis} - x^\text{forecast}$

Spatial maps and vertical sections of increments show:

- where the data strongly influence the state  
- whether corrections are physically coherent  
- whether persistent biases exist

---

### Summary

| Metric | Purpose | Output |
|--------|---------|--------|
| **MAE** | Accuracy of deterministic forecast/analysis | Hovmöller, profiles, time series |
| **CRPS** | Probabilistic skill of ensemble forecast | Time series, depth-specific CRPS |
| **Increments** | Diagnose assimilation impact | Maps and sections |

These metrics provide a comprehensive view of assimilation performance for biogeochemical variables in the Black Sea.

## Summary Diagnostics Notebook: `SORDA_summaries_plotting.ipynb`

This notebook provides a **lightweight**, **pre-computed** view of the SORDA
assimilation performance. It is intended for users who want to explore
diagnostics *without* running the full EnKF/OAK pipeline.

### Inputs

The notebook reads summary NetCDF files stored in:

```text
demo_summaries/
├── mae_summary.nc          # MAE Hovmöller, global MAE, vertical profiles
├── crps_summary.nc         # CRPS time series summaries
└── increments_combined.nc  # (optional) increments for maps/sections

These files are produced by save_mae_summary and save_crps_summary functions of ./src/python/summaries.py indode thewhole pipeline demo notebook. 

### Notebook capabilities

For each biogeochemical variable:

- `CDI`
- `CEM`
- `CFL`
- `POC`
- `MES`
- `MIC`
- `GEL`

the notebook produces:

- **MAE Hovmöller diagrams** (time–depth) for forecast vs analysis
- **Global MAE time series** (domain- and depth-averaged)
- **MAE depth profiles** (time-mean, depth-dependent error)
- **CRPS time series** (forecast vs analysis)
- **Increment maps** (optional, if increments are available)

