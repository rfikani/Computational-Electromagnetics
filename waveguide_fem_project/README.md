# 2D FEM eigenmode solver for PEC Waveguides

<img width="1860" height="540" alt="mesh_plots" src="https://github.com/user-attachments/assets/d126b2eb-03ce-451d-b501-483e109981f0" />


## Files

- `waveguide_fem/constants.py` — physical constants
- `waveguide_fem/models.py` — mesh/mode dictionary builders
- `waveguide_fem/mesh.py` — rectangular, circular, and double-ridged mesh generation
- `waveguide_fem/fem.py` — local element matrices, assembly, normalization helpers
- `waveguide_fem/solver.py` — TE/TM generalized eigenvalue solves
- `waveguide_fem/references.py` — analytical references and target mode shapes
- `waveguide_fem/utils.py` — eigenspace grouping, projection, dispersion helper
- `waveguide_fem/matching.py` —mode matching / extraction
- `waveguide_fem/pipeline.py` —full solve for one mesh
- `waveguide_fem/plotting.py` — mode and dispersion plotting
- `run_all.py` —  runs everything

## How to run

### 1. Create and activate a virtual environment

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python3 -m venv .venv
.venv\Scripts\Activate.ps1
```
### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```
### 3. Run the project

```bash
python3 run_all.py
```
## Output files

Running `run_all.py` produces:

- `rect_waveguide_modes_fixed.png`
- `circ_waveguide_modes_fixed.png`
- `double_ridged_waveguide_modes_fixed.png`
- `rect_waveguide_dispersion_fixed.png`
- `circ_waveguide_dispersion_fixed.png`
- `double_ridged_waveguide_dispersion_fixed.png`
