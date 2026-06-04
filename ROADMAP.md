# roadmap — crop-disease-detector

## where it is now

45-feature Random Forest pipeline trained on PlantVillage (19,000 samples, 38 classes). CV accuracy 84.8%. Runs locally, no GPU needed, results in under a second.

## short term

- clean up the class name display — raw PlantVillage names like `Corn_(maize)___Common_rust_` need a proper mapping to readable names
- image quality check before inference — reject blurry or dark images rather than returning a low-confidence result silently
- confidence threshold — return "inconclusive" below 40% rather than forcing a prediction
- batch upload — process a folder of images and export results as CSV

## medium term

- replace the feature engineering pipeline with a fine-tuned MobileNetV2 — target 93%+ accuracy with a model small enough to run on a phone
- ONNX export for offline inference — no backend required after initial model download
- multi-leaf detection in a single image using a lightweight object detection layer
- geolocation tagging so reports can be tied to a specific farm location and tracked over time

## ai integration ideas

- use Gemini Vision or GPT-4o as a secondary validation layer when primary model confidence is below 50% — the LLM examines the image independently and explains its reasoning
- natural language query interface: "what diseases affect tomatoes in humid conditions?" — LLM with a curated agronomic knowledge base
- treatment report generation in Telugu, Hindi, and Tamil — most of the farmers this is actually built for don't read English

## infrastructure

- model retraining pipeline triggered when users submit corrected labels — active learning loop
- REST API versioning so the mobile app and web frontend can be updated independently