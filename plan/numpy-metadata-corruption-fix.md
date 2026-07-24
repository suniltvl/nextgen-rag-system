# NumPy Metadata Corruption Fix

**Date:** 2026-07-19  
**Project:** nextgen-rag-system  
**Affected area:** `.venv`, imports via `transformers` / LangChain

---

## Summary

NumPy **imports and runs correctly**, but Python cannot read its **installed package version** from metadata. Libraries such as `transformers` (pulled in by LangChain) fail at import time with:

```text
ValueError: Unable to compare versions for numpy>=1.17: need=1.17 found=None.
This is unusual. Consider reinstalling numpy.
```

This is a **broken package install**, not a missing NumPy dependency.

---

## Symptoms

- `import numpy` succeeds and reports a version (e.g. `2.4.6`)
- `from importlib.metadata import version; version("numpy")` returns `None`
- `import transformers` or `from langchain_text_splitters import ...` fails with the error above
- `uv sync` may warn about missing `RECORD` files under `numpy-*.dist-info`
- `uv pip show numpy` may list **two versions** (e.g. `1.26.4` and `2.4.6`) in the same environment

---

## Root Cause

The virtual environment has a **partially applied NumPy upgrade**:

| Component | State |
|-----------|--------|
| `numpy/` package code | Present and importable |
| `numpy-1.26.4.dist-info/` | Leftover from an older install |
| `numpy-2.4.6.dist-info/` | Incomplete (missing `METADATA`, `RECORD`) |
| `importlib.metadata` | Registers **zero** NumPy distributions |

This typically happens when `uv sync` or `pip install` tries to reinstall NumPy while another process holds open NumPy DLLs on Windows:

```text
Access is denied (os error 5)
failed to rename ... numpy\linalg\_umath_linalg.cp312-win_amd64.pyd
```

Common lockers:

- Running Jupyter kernels
- Python REPLs or notebooks using the same `.venv`
- IDE-integrated Python sessions

The upgrade writes new code but leaves metadata incomplete, so version checks fail even though `import numpy` works.

---

## How Transformers Triggers the Failure

On import, `transformers` validates dependencies using `importlib.metadata`, not `numpy.__version__`:

```python
from importlib.metadata import version
version("numpy")  # → None when metadata is corrupted
```

That `None` value causes the `ValueError` before any application code runs.

---

## Where This Appeared in the Project

In `src/questionDB.ipynb`, Cell 0 imported unused LangChain packages:

- `langchain_text_splitters` → `transformers` → NumPy version check → crash

The notebook logic only needed `datasets`, `dotenv`, `os`, and `psycopg`. Removing unused LangChain imports avoids the broken chain for that notebook, but **the underlying venv issue remains** for other notebooks (`Vectorization.ipynb`, `Generation.ipynb`, etc.).

---

## Fix Procedure

### Step 1 — Stop processes using the venv

Close:

- All Jupyter notebook kernels
- Terminals running Python from `.venv`
- Any IDE Python sessions bound to this environment

### Step 2 — Reinstall dependencies

```powershell
cd F:\Github\Capstone\nextgen-rag-system
uv sync
```

### Step 3 — Verify metadata is restored

```powershell
.venv\Scripts\python.exe -c "from importlib.metadata import version; import numpy; print('metadata:', version('numpy')); print('import:', numpy.__version__)"
```

**Expected:** both values match (e.g. `2.4.6`), not `metadata: None`.

Also confirm:

```powershell
.venv\Scripts\python.exe -c "import transformers; print('transformers ok')"
```

### Step 4 — If `uv sync` still fails (full reset)

```powershell
# Ensure all kernels/processes are closed first
Remove-Item -Recurse -Force .venv
uv sync
```

Then rerun the verification commands from Step 3.

---

## Prevention

1. **Close Jupyter kernels** before running `uv sync` or upgrading NumPy-heavy stacks.
2. **Avoid duplicate NumPy installs** — let `uv` manage a single pinned version via `pyproject.toml` / `uv.lock`.
3. After any failed sync, **verify metadata** before assuming the environment is healthy:

   ```powershell
   .venv\Scripts\python.exe -c "from importlib.metadata import version; print(version('numpy'))"
   ```

4. Prefer **minimal imports** in notebooks so unrelated dependency chains (LangChain → transformers → NumPy check) do not block simple scripts.

---

## Related Fixes (Same Investigation)

These were separate issues found while debugging `questionDB.ipynb`; they do not replace the NumPy metadata fix above.

| Issue | Cause | Fix |
|-------|--------|-----|
| Unused LangChain imports in `questionDB.ipynb` | Pulled in broken `transformers` chain | Trim imports to `datasets`, `dotenv`, `os`, `psycopg` |
| `psycopg` import failure on Windows | Plain `psycopg` needs system `libpq` | Use `psycopg[binary]>=3.3.4` in `pyproject.toml` |
| `DATABASE_URL` connection error | Truncated `sslmode=requir` in `.env` | Correct to `sslmode=require` |
| `cs_questions` table missing | Table never created in Neon Postgres | Created table matching `nextgenrag_questions` schema |
| Dataset typo | `emanaul` invalid config name | Corrected to `emanual` |

---

## Quick Reference

| Check | Healthy | Broken |
|-------|---------|--------|
| `import numpy` | OK | OK (misleading) |
| `version("numpy")` | e.g. `2.4.6` | `None` |
| `import transformers` | OK | `ValueError` |
| `numpy-*.dist-info/RECORD` | Present | Missing |
| `uv pip show numpy` | One entry | Multiple entries |
