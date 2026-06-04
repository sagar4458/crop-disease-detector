# Data — Crop Disease Detector

## Files

- `sample_images_metadata.csv` — metadata for 300 labelled leaf images (image IDs, disease labels, capture conditions)
- `disease_stats.json` — class distribution and collection summary

## Source

Images sourced from open agricultural datasets and supplemented with locally collected samples.
All disease labels were cross-verified with an agronomist before being used for training.

## Note

Actual image files are not included in this repo due to size (avg 2MB each, ~600MB total).
The metadata CSV maps image IDs to their storage location and ground-truth labels.
To reproduce training, download the images from the dataset links in `model/train.py` and place
them in `data/images/` following the folder structure in `sample_images_metadata.csv`.
