# AI Structural Health Monitoring and Autonomous Robotic Inspection

## Problem Statement

Civil and industrial infrastructure (bridges, tunnels, pipelines, and industrial facilities) deteriorates over time due to environmental exposure, material fatigue, and operational loads. Traditional manual inspection is labour-intensive, time-consuming, and sometimes hazardous. There is a growing need for automated, AI-assisted tools that can help engineers quickly identify potential structural defects from images captured during routine or robotic inspections.

## Current Solution

This project develops a Python-based prototype that:

1. Accepts inspection images uploaded by the user (or eventually captured by an autonomous inspection robot).
2. Provides a structured interface for selecting the type of infrastructure being inspected.
3. Will, in future iterations, use computer-vision and deep-learning techniques to detect and localise common defects such as cracks and corrosion.

## Current Prototype Features (v1)

- Streamlit web interface with wide layout.
- Sidebar for selecting the structure type (Pipeline, Bridge, Tunnel, Industrial Structure).
- Image upload supporting JPG, JPEG, and PNG formats.
- Display of the uploaded image with filename and pixel dimensions.
- Clear disclaimer regarding responsible use.

Defect detection, AI inference, and robotic integration are not yet implemented. They are planned for subsequent versions.

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501.

## Team

- Unzila Zeb
- Momina Rizwan
- Naba Raheel

## Responsible-Use Disclaimer

This prototype is an academic project developed for a university AI hackathon. It is intended to support engineers in preliminary visual assessment and does not replace certified Non-Destructive Testing (NDT) inspection performed by qualified professionals. Any outputs produced by this tool should be verified by domain experts before being used in safety-critical decisions.
