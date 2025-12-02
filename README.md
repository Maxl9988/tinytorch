# TinyTorch (without Tito)

This repository contains a version of **TinyTorch** without the original Tito Guide. It is structured so that you can work through the assignments, implement missing components, and optionally build your own TinyTorch implementation alongside the provided modules.

---

## Quick Start Guide
Install uv: [Astral uv](https://docs.astral.sh/uv/getting-started/installation/)  
```
git clone https://github.com/Maxl9988/tinytorch.git
cd tinytorch
uv sync

# Open VS-Code and choose the kernel tiny-torch
```
---

## Repository Structure

- **assignments/**  
  Contains all Jupyter notebooks (`.ipynb`) where you must write your solutions.  
  All code between `BEGIN SOLUTION` and `END SOLUTION` has been removed and must be implemented by you.  
  Each notebook contains tests at the end to verify correctness.

- **core/**  
  Contains the default import modules used by the assignments.  
  These files provide the structure required by the notebooks but do not contain your personal implementations.

- **core_own/**  
  Intended for your **own** Python modules.  
  If you want your own implementations to be reused in later assignments, copy your code from the `.ipynb` files into new `.py` files in this folder and update the imports accordingly.

- **solutions/**  
  Contains all completed solution notebooks.  
  Every solution notebook has been executed, and each one reports a successful module completion in the final test.

---



## Using Your Own Implementations in Later Assignments

If later assignments should use your written code instead of the default modules in `core/`, follow these steps:

1. Implement the solution inside the assignment notebook.  
2. Copy your working implementation from the notebook.  
3. Paste it into a new `.py` file inside `core_own/` (e.g., `core_own/tensor.py`).  
4. Change the imports in the notebook from  
   `from core.xyz import Something`  
   to  
   `from core_own.xyz import Something`.

This allows your TinyTorch implementation to grow across modules like a real framework.


 
Many thanks to [TinyTorch](https://github.com/MLSysBook/TinyTorch)  for the outstanding course.  
Unfortunately, I am not a fan of the Tito module!  
This repository was created while TinyTorch was still under construction.
