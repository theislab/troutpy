# This file is a template for paths.py. paths.py is listed in .gitignore since different users might want to put in
# custom pathes. If you add a new path variable to paths.py make sure to also add it to paths_template.py. 
# As a reminder we run a test that checks if paths.py and paths_template.py have the same variables.


from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Define paths
PROJECT_DIR = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/")
RESULTS_DIR = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/results")
#FIG_DIR = ROOT / "figures"

xenium_path_cropped = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/example_datasets/mousebrain_prime_crop_communication.zarr")

# Check if path variables in paths.py and paths_template.py are the same
if Path(__file__).name == "paths.py":
    import paths_template
    pvars_ext = set(
        [key for key in dir(paths_template) if not key.startswith("_") and (key not in ["paths","paths_template"])]
    )
    pvars_int = set(
        [key for key in dir() if (not key.startswith("_")) and (key not in ["paths","pvars_ext","paths_template"])]
    )
    if not (pvars_ext == pvars_int):
        raise ImportError("Please update paths_template.py and paths.py to have the same path variables.\n"
                          f"Variable mismatch: {pvars_ext.symmetric_difference(pvars_int)}.")    


