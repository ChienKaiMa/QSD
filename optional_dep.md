# Optional Dependencies
## qclib fix
Due to floating-point precision issues in qclib's matrix checks (specifically, strict `np.allclose` comparisons without explicit tolerance), you may encounter the following error during circuit compilation. This affects the orthonormal columns validation in isometry decomposition.

**Example Error Traceback:**
```plaintext
Traceback (most recent call last):
  File "/home/tacas/QSD/quick_test_tacas.py", line 31, in <module>
    povm_ckts.build_circuit()
  File "/home/tacas/QSD/flow/build_circuits.py", line 122, in build_circuit
    qc_iso = qclib.isometry.decompose(unitary_gate, scheme=scheme)
  File "/home/tacas/QSD/venv/lib/python3.12/site-packages/qclib/isometry.py", line 58, in decompose
    check_isometry(iso, log_lines, log_cols)
  File "/home/tacas/QSD/venv/lib/python3.12/site-packages/qclib/isometry.py", line 85, in check_isometry
    raise ValueError("The input matrix has non orthonormal columns.")
ValueError: The input matrix has non orthonormal columns.
```

The installation script (`install.sh`) automatically applies a single-line sed command to relax the tolerance in the relevant `np.allclose` call to `atol=1e-3`. After running the script, re-compile your circuits. If issues persist, try a larger tolerance manually or update qclib.

## qclib fix
If you see the following error during circuit compilation, please manually
modify the qclib code for the circuit compilation. (TBA)

File location: `.venv/lib/python3.12/site-packages/qclib/isometry.py#L90`
```python
return np.allclose(identity, np.eye(int(2**log_cols)))
```
to
```python
return np.allclose(identity, np.eye(int(2**log_cols)), atol=1e-5)
```

## SCS with oneMKL acceleration

### Install oneMKL
You can find the same commands in the references with more details, but it is convenient to copy and paste the whole thing if you understand what these commands will do.

References:
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt
- https://api.projectchrono.org/module_mkl_installation.html
- https://stackoverflow.com/questions/78274730/symbol-lookup-error-with-intel-mkl-on-wsl

```bash
sudo apt update
sudo apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-mkl -y
sudo apt install intel-oneapi-mkl-devel -y
sudo apt install libomp-dev -y
```

### Install SCS with MKL Pardiso interface
Reference:
- https://www.cvxgrp.org/scs/install/python.html#python-install

Assume that you are at your project root folder.
```bash
# Sync the dependencies in pyproject.toml
uv sync
# Activate the environment
source .venv/bin/activate
```
Then you should modify the pyproject.toml first (Code TBA).
Then,
```bash
git clone --recursive https://github.com/bodono/scs-python.git
cd scs-python

# Build the wheel for SCS with MKL interface.
# pyproject.toml will find the wheel when syncing.
uv build --wheel -Csetup-args=-Dlink_mkl=true .
# uv pip install --verbose -Csetup-args=-Dlink_mkl=true .
```
```bash
# Verify installation
uv pip install pytest
python -m pytest .
cd ../
uv sync
```

