# Seam Carving Project
## Running
Method 1: command line: `python3 <input_filename> <output#columns> <output#rows> <energy_type> <output_filename>  opt(optional)`
examples are included in `run.sh`

Method 2: command line: `sh run.sh`

## Files
* `run.sh`
* `seam_main.py`
* `sc.py`: seam carving functions
* `part1_energy.py`: energy function including RGB-value differences and local entropy.
* `forward_energy.py`: 'forward' energy function
* `network_energy.py`: energy map coming from VGG-19
* `combined_energy.py`: combination of RGB, entropy and network energy
* `input`: contains some input images
* `output`: contains some output images