# 2D TMz FDTD solver


## Files

- `main.py` runs everything
- `constants.py` has the physical constants and global settings
- `mesh.py` builds the grid, material arrays, PML arrays, and field arrays
- `profiles.py` builds the graded PML conductivity profile
- `source.py` has the impressed source waveform
- `solver.py` has the `steptmz(...)` update step
- `cases.py` contains the three study blocks
- `plotting.py` contains the plotting code



## How to run

1. Open the project folder in your terminal.
2. Install the needed packages:

```bash
pip3 install -r requirements.txt
```

3. Run the code:

```bash
python3 main.py
```

