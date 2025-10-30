#!/usr/bin/env bash
# Usage: source install.sh to persist venv activation; or bash install.sh for deps only

# Install uv if not already installed
if ! command -v uv > /dev/null 2>&1; then
    echo "Installing uv..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    # Reload profile to make uv available in this session
    source ~/.bashrc
fi

# Sync the dependencies in pyproject.toml; creates .venv if needed
uv sync

# Activate the environment; only persists if script is sourced
source .venv/bin/activate

# Loosen qclib np.allclose tolerance; atol=1e-5
# Dynamically find the isometry.py path for portability across Python versions
isometry_file=$(python -c "import qclib; print(qclib.__file__.rsplit('/', 1)[0] + '/isometry.py')")
sed -i.bak "s/return np.allclose(identity, np.eye(int(2\*\*log_cols)))/return np.allclose(identity, np.eye(int(2\*\*log_cols)), atol=1e-5)/" "$isometry_file"

# Check if the fix was applied
if grep -q "atol=1e-5" "$isometry_file"; then
    echo "qclib fix applied successfully."
else
    echo "qclib fix not detected - check $isometry_file manually."
fi

echo "Installation complete! Venv is ready. If you ran with bash install.sh, manually activate with source .venv/bin/activate."