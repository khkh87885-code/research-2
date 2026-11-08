# Observed research export

Records: 102; successful: 102; failed or incomplete: 0.
Patient identifiers missing: 0 records. Missing patient identities are not treated as separate patients.

All image records, including failures, are retained. Patient summaries average valid finite values within each dataset, split, condition and prompt strategy. Condition summaries give each eligible patient equal weight.

No confidence intervals or hypothesis tests are inferred by this export module. Additional supplied statistical results are stored separately. Upload times, if present elsewhere, are simulations rather than network measurements.

Figures contain aggregate results only. This export includes no original image paths or medical images.

## Run metadata

```json
{
  "purpose": "development_technical_smoke",
  "started_at": "2026-11-08T04:21:52.077414+00:00",
  "fingerprint_sha256": "879a77cdb3f4d1a85c5e27006a1aab7e083d9c9787372ca30986bae76bb3acfd",
  "runtime": {
    "python": "3.8.20 (default, Oct  3 2024, 15:19:54) [MSC v.1929 64 bit (AMD64)]",
    "platform": "Windows-10-10.0.26200-SP0",
    "torch": "2.4.1+cu118",
    "numpy": "1.24.4",
    "pillow": "10.4.0"
  },
  "limitations": [
    "Technical pilot only unless separately prespecified",
    "Distances in pixels",
    "Full-frame application prompt is unvalidated",
    "IACS performance unvalidated"
  ]
}
```
