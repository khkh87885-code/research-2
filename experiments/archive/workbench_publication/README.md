# Legacy workbench: deidentified numeric technical-pilot records

This bundle contains every final task row from the three legacy local workbench experiments, including the entire failed run. No new experiment or patient result was invented. These records are separate from the later IACS_v2 development study.

The JSONL files preserve nested numeric measurements; CSV stores nested objects as JSON cells. Source summaries and deidentified configuration summaries retain processing recipes, source-code/model checksums, feature definitions, numeric metadata, method descriptions and run settings. Descriptive metric tables retain planned/valid/missing-or-failed counts. They do not establish clinical acceptability or noninferiority. Upload times are simulated, not measured network performance. The source dataset inventory is aggregate-only.

Case and image keys are stable HMAC-SHA256 identifiers (24 hex characters), using the existing pilot's `dataset + ':' + raw identifier` convention. The external secret key and raw-to-pseudonym correspondence are not included. Case/source identifiers are not automatically confirmed independent patients.

Excluded: all original images, masks, derived pixel images, existing figures, SQLite databases, pathology, raw identities, source-image paths, per-image content hashes and free-form failure messages. Original error types and failure states are retained. Per-experiment failure summaries provide fixed, non-identifying context labels when a recognized phrase is present; omitting free-form text does not remove failure rows.

`SOURCE_SHA256.json` identifies the read-only input numeric/configuration files by relative experiment label and SHA-256. `EXPORT_AUDIT.json` records full row counts, numeric-value checks and before/after source hashes. `ARTIFACT_SHA256.json` verifies this export's files.

## Repeat locally

The included script uses only Python's standard library:

```powershell
python -I export_legacy_numeric.py --source-root '<local experiments directory>' --output '<new separate export directory>' --key-file '<external secret file>' --inventory '<optional local inventory.json>'
```

The key-file option is optional; omitting it creates a temporary key that is never exported and produces new pseudonyms on the next invocation. Supply the same external key to reproduce stable identities across exports. Never put the secret key or original correspondence data in a public repository. The destination must not exist; existing exports and input files are not overwritten.
