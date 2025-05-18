# QSD
Implementation of the quantum state discrimination in quantum circuits

## Install SCS with oneMKL acceleration

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

```bash
 git clone --recursive https://github.com/bodono/scs-python.git
 cd scs-python
 uv pip install --verbose -Csetup-args=-Dlink_mkl=true .
 uv pip install pytest
 python -m pytest .
```
