# Artificial Intelligence for Structural Health Monitoring (SHM) and Autonomous Robotic Inspection

## Project Overview

This project is a Streamlit-based, explainable rule-based computer-vision prototype for preliminary screening of inspection images from pipelines, bridges, tunnels, and industrial structures. It is designed as the vision-software component of a future autonomous robotic inspection system.

The prototype provides a structured workflow for uploading inspection images, applying preprocessing and segmentation techniques, reviewing candidate regions with quantitative metrics, and generating a downloadable plain-text inspection report. All outputs are screening-level candidates intended for review by qualified engineers — not confirmed defect classifications.

## Problem Statement

Civil and industrial infrastructure — bridges, tunnels, pipelines, and industrial facilities — deteriorates over time due to environmental exposure, material fatigue, and operational loads. Traditional manual inspection presents significant limitations:

- **Cost** — Deploying qualified inspection teams and specialized NDT equipment to multiple sites is expensive, particularly for geographically distributed infrastructure networks.
- **Time** — Manual visual inspection of large structures is slow, limiting the frequency and coverage of assessment cycles.
- **Accessibility** — Many structural elements (elevated bridge components, pipeline segments, tunnel linings) are physically difficult or dangerous to reach on foot.
- **Safety** — Inspectors working in confined spaces, at height, or near live infrastructure face real occupational hazards.

There is a growing need for automated vision tools that can support engineers by rapidly triaging inspection imagery and highlighting areas that warrant closer investigation.

## Implemented Features

- **Image upload** — JPG, JPEG, and PNG formats supported via a Streamlit file uploader.
- **Automatic RGB/RGBA normalization** — Uploaded images are converted to three-channel RGB regardless of source format, ensuring consistent downstream processing.
- **Structure-type selection** — Sidebar control for choosing the infrastructure category (Pipeline, Bridge, Tunnel, Industrial Structure).
- **CLAHE contrast enhancement** — Contrast Limited Adaptive Histogram Equalization improves local contrast in the grayscale image without over-amplifying noise.
- **Adjustable Canny edge detection** — User-tunable lower and upper hysteresis thresholds for edge-map generation.
- **Explainable HSV and RGB corrosion-colour candidate segmentation** — A fused dual-criteria approach flags pixels as corrosion candidates only when they satisfy both an HSV rust-colour range and an RGB red–brown dominance test.
- **Morphological mask cleanup** — Elliptical opening removes small noise specks; closing fills small holes in the corrosion candidate mask.
- **Corrosion candidate visualization and region metrics** — Orange-tinted overlay and contour outlines on the annotated image; candidate pixel count, area percentage, and accepted region count displayed in a metrics table.
- **Preliminary corrosion-extent bands** — Candidate area classified as Minimal (below 2%), Localized (2%–10%), Moderate (10%–25%), or Extensive (25% and above).
- **Experimental dark linear-anomaly screening** — Rule-based pipeline combining black-hat filtering, Otsu thresholding, Canny-edge intersection, and morphological operations to highlight dark elongated features, with adjustable minimum contour area, elongation ratio, and maximum candidate area controls.
- **Processing and candidate metrics** — Image dimensions, edge-pixel statistics, corrosion metrics, and linear-anomaly metrics presented in structured tables.
- **Corrosion-based review priority and recommended action** — A follow-up scheduling hint derived from the preliminary corrosion extent only; linear-anomaly results are deliberately excluded.
- **Downloadable timestamped plain-text inspection report** — A comprehensive report file containing the inspection settings, processing metrics, corrosion and linear-anomaly results, review priority, recommended action, method limitations, and engineer-verification disclaimer.
- **Responsible-use warnings and engineering/NDT verification requirements** — On-screen disclaimers and report text clearly stating that all outputs require verification by qualified engineers and certified NDT personnel.

## How It Works

The prototype follows a sequential processing pipeline:

1. **Uploaded image** — The user selects a structure type and uploads an inspection image (JPG, JPEG, or PNG).
2. **RGB normalization** — The image is converted to a guaranteed three-channel RGB array, normalizing any RGBA or palette-based source.
3. **Preprocessing and edge map** — The image is resized (if wider than 1200 px), converted to grayscale, enhanced with CLAHE, smoothed with a Gaussian blur, and processed through a Canny edge detector with user-defined thresholds.
4. **Corrosion-colour candidate segmentation** — The RGB image is converted to HSV; pixels within the user-defined hue, saturation, and brightness window are combined with an RGB red–brown dominance mask using logical AND. The fused mask is cleaned with morphological opening and closing, and external contours above a minimum area threshold are retained.
5. **Experimental linear-anomaly screening** — A black-hat filter highlights dark features, which are then thresholded with Otsu's method, intersected with dilated Canny edges, morphologically closed, and filtered by minimum area, elongation ratio, and maximum area percentage.
6. **Metrics and preliminary summary** — Processing metrics, corrosion metrics, linear-anomaly metrics, preliminary corrosion extent, review priority, and recommended next action are compiled into an on-screen Preliminary Inspection Summary.
7. **Downloadable report** — A timestamped plain-text report containing all settings, metrics, results, limitations, and disclaimers is generated and offered for download.

> **Note:** Linear-anomaly screening results are experimental. They are displayed for research purposes only and are explicitly excluded from the corrosion-based review priority and recommended action.

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web-based interactive user interface |
| **OpenCV** | Image processing and computer-vision operations |
| **Pillow** | Image loading and format normalization |
| **NumPy** | Numerical array operations |

This prototype was developed with assistance from Qoder.

## Installation

### Windows

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

If PowerShell script execution is blocked, run the following command in the same PowerShell session before activating:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Running the Application

```powershell
python -m streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Using the Prototype

1. **Select a structure type** — Use the sidebar dropdown to choose the infrastructure category (Pipeline, Bridge, Tunnel, or Industrial Structure).
2. **Upload an image** — Click the file uploader and select a JPG, JPEG, or PNG inspection image.
3. **Adjust thresholds** — Use the sidebar sliders to tune Canny edge detection thresholds, HSV corrosion colour ranges (hue, saturation, brightness), and experimental linear-anomaly parameters (minimum contour area, elongation ratio, maximum candidate area).
4. **Automatic processing** — The application reruns the processing pipeline automatically after an image is uploaded or a setting is changed.
5. **Review masks and metrics** — Examine the original image, contrast-enhanced grayscale, edge map, corrosion candidate mask, annotated corrosion overlay, dark linear-feature mask, and annotated linear-anomaly candidates alongside their respective metric tables.
6. **Read the summary** — The Preliminary Inspection Summary consolidates all key results, the corrosion-based review priority, and the recommended next action.
7. **Download the report** — Click the download button to save a timestamped plain-text inspection report for your records.

## Interpretation and Limitations

- **Outputs are candidates, not confirmed defects.** Every flagged region is a screening-level indication that requires verification by a qualified engineer using appropriate inspection methods.
- **Corrosion results are colour-based.** The HSV and RGB segmentation may produce false positives due to lighting conditions, paint colour, soil, surface contamination, or image quality, and may miss corrosion that is not rust-coloured.
- **The prototype does not measure** corrosion depth, wall-thickness loss, structural loading, remaining service life, or failure probability. It has no access to ultrasonic, radiographic, or other NDT sensor data.
- **Linear anomalies cannot reliably distinguish** cracks from joints, scratches, shadows, material boundaries, surface texture, or image noise.
- **This is not a trained or validated defect-diagnosis model.** The prototype uses explainable, rule-based computer-vision techniques only. No machine-learning or deep-learning inference is performed.
- **Results require qualified engineer and certified NDT verification** before any maintenance, repair, or safety decisions are made.

## Future Development

- Robotic camera integration for autonomous image acquisition in the field.
- Calibrated NDT sensor integration (ultrasonic, radiographic, or other modalities) for quantitative defect characterization.
- Dataset collection and annotation from real-world inspection campaigns.
- Trained and validated defect-detection models using supervised deep learning.
- Localization of detected defects to structural coordinates for asset management integration.
- Field testing on operational infrastructure to evaluate performance under real-world conditions.

## Repository Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application — contains the complete user interface, image preprocessing, corrosion-colour candidate segmentation, experimental linear-anomaly screening, preliminary inspection summary, and downloadable report generation. |
| `requirements.txt` | Python package dependencies (Streamlit, OpenCV, Pillow, NumPy). |
| `README.md` | Project documentation (this file). |
| `.gitignore` | Specifies files and directories excluded from version control (virtual environment, Python cache, etc.). |

## Team

- **Unzila Zeb**
- **Momina Rizwan**
- **Naba Raheel**

Institution: NUST College of Electrical & Mechanical Engineering (CEME)

## Responsible-Use Disclaimer

This prototype supports preliminary screening only and does not replace inspection by qualified engineers or certified NDT personnel. It is an academic project developed for a university AI hackathon. All candidate findings must be verified on site by a qualified engineer using appropriate inspection methods before any safety-critical or maintenance decisions are made.
