# Install uv if uv is not installed yet
wget -qO- https://astral.sh/uv/install.sh | sh

# Sync the dependencies in pyproject.toml
uv sync
# Activate the environment
source .venv/bin/activate
