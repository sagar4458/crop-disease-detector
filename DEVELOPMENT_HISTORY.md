# development notes — crop-disease-detector

Started this as a computer vision learning project. The specific motivation was watching farmers apply fungicide for what turned out to be a viral disease — completely the wrong treatment — because there was no easy way to get a quick diagnosis.

First version used only colour channel analysis and was embarrassingly bad. Kept getting healthy plants flagged because I hadn't normalised the input properly. That took two evenings to debug. Added texture variance, edge density, and multi-scale colour features later, which made a significant difference for distinguishing diseases that have similar colour profiles but different spread patterns.

Ended up with a 45-feature pipeline — RGB statistics at full, half, and quarter resolution, grayscale histogram bins, percentile values, and channel difference ratios. Each feature addition was motivated by what agronomists actually look for visually, not just throwing things at the model.

Trained on the PlantVillage dataset — 19,000 samples across 38 disease classes. Random Forest with 500 trees hit 84.8% CV accuracy. Not state of the art, but runs on any machine without a GPU and gives results in under a second.

## what I'd do differently

- image quality check before inference — currently accepts blurry or dark images and gives unreliable results without warning
- the feature extraction function has some redundant transformations that should be cleaned up
- would add a confidence threshold below which the model returns "inconclusive" rather than forcing a low-confidence prediction
- a fine-tuned MobileNetV2 would push accuracy to 95%+ — the feature engineering approach has a ceiling

## known gaps

Treatment recommendations use the raw PlantVillage class names which are ugly and inconsistent. Display cleaning is partial — needs a proper name mapping for all 38 classes.

---
*Published September 2025 after training on the full PlantVillage dataset...*