# Result Migration

This document tracks the migration of experiment results from the source server to this workspace.

## Source Details
- **Path:** `/po1/yan/afpi/`
- **Folders:** `cfg1`, `cfg3`, `cfg5`, `cfg7`
- **Target Sub-directories:** `aidi`, `exact`, `fpi`, `spd`, `ddim`
- **File Types:** `*.pt`, `*.png`

## Migration Command
To synchronize new results while preserving the directory structure and filtering out unwanted files, use the provided `Makefile` target:

```bash
make migrate
```

## Configuration
The `rsync` command is configured to:
1. Include only `cfg*` top-level directories.
2. Include only the specified sub-directories (`aidi`, `exact`, etc.).
3. Include only `*.png` and `*.pt` files within those directories.
4. Exclude everything else (e.g., `.ipynb`, large logs, or other subfolders like `spd_real`).
