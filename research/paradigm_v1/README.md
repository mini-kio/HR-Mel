# HR-Mel Paradigm Studies (v1)

This folder contains step-by-step studies that move HR-Mel from a fixed recipe to a design-space paradigm.

## Run

```bash
python research/paradigm_v1/run_research.py --input . --output-dir research/paradigm_v1/results
```

Optional:

```bash
python research/paradigm_v1/run_research.py --input . --max-files 2
```

## Outputs

- `study_01_design_space_per_file.csv`
- `study_01_design_space_aggregate.csv`
- `study_01_best_config.json`
- `study_02_rate_distortion_per_file.csv`
- `study_02_rate_distortion_aggregate.csv`
- `study_03_multirate_evidence.csv`
- `study_03_multirate_summary.json`
- `study_04_psycho_allocator.csv`
- `study_04_psycho_allocator_summary.json`
- `study_05_content_adaptive.csv`
- `study_05_content_adaptive_summary.json`
- `study_06_standard_api_spec.json`
- `SUMMARY.md`
- `run_meta.json`

## Study Scope

1. Design space search:
   - Band boundaries (`f1`, `f2`)
   - Bin allocations (`n1`, `n2`, `n3`)
   - High-band nonlinearity (`log1p`, `sqrt_log1p`, `pow075`)
2. Rate-distortion:
   - Uniform scalar quantization at multiple bit-widths
   - Compare `mel96_log` and best `hr` config under equal token rate formula
3. Multi-rate evidence:
   - High-band temporal contrast gain with shorter hop
   - Low-band peak jitter reduction with longer FFT
4. Psychoacoustic allocation:
   - A-weighted energy driven 3-band bin assignment
5. Content-adaptive profiles:
   - Rule-based profile selection from centroid/flatness/onset
6. Standard API draft:
   - JSON schema for profile-driven `HRMelSpec`
