# Build environment

# Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

```bash
uv init test_qsd
cd test_qsd
uv python pin 3.10
uv sync
source .venv/bin/activate
```
```bash
uv run python ...
```