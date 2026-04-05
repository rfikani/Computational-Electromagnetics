# 2D FEM eigenmode solver for PEC Waveguides

<img width="1860" height="540" alt="mesh_plots" src="https://github.com/user-attachments/assets/d126b2eb-03ce-451d-b501-483e109981f0" />


<img width="2358" height="1280" alt="rect_waveguide_modes_fixed" src="https://github.com/user-attachments/assets/5ef6b152-501e-46d4-bc90-178ee7ca5547" />


<img width="1460" height="1100" alt="rect_waveguide_dispersion_fixed" src="https://github.com/user-attachments/assets/cc643dbe-14b3-48fb-b337-23a6c6dabbd8" />


<img width="2361" height="1275" alt="circ_waveguide_modes_fixed" src="https://github.com/user-attachments/assets/d11d907d-bb3b-4c06-a995-ff37a636f342" />


<img width="1460" height="1100" alt="circ_waveguide_dispersion_fixed" src="https://github.com/user-attachments/assets/831a760b-1be0-4027-9fe4-d7dac3d5daba" />


<img width="2358" height="1280" alt="double_ridged_waveguide_modes_fixed" src="https://github.com/user-attachments/assets/4b5d4cb7-b372-4b18-9a51-2b6e1cad900f" />


<img width="1460" height="1100" alt="double_ridged_waveguide_dispersion_fixed" src="https://github.com/user-attachments/assets/4cf23eb3-991a-4767-ba85-a0efb3e39eca" />


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
