# Clinical RAG Active Verifiability Framework

**Creator:** Anthony Onoja  
**Affiliation:** School of Health Sciences, University of Surrey, UK  
**Contact:** a.onoja@surrey.ac.uk  

This repository contains the **Clinical RAG Active Verifiability Framework** (AV-RAG), a clinical research decision-support tool focused on providing high epistemic transparency through constraint-aware retrieval, claim-level attestation, dynamic guideline anchoring, and a comprehensive Human-in-the-loop (HITL) architecture.

## Overview

Unlike standard RAG systems that provide a single authoritative generation, this app utilizes multiple Agents (Agent A: Evidence Analyser, Agent B: Guideline Comparator) to surface contradicting evidence and map it against accepted medical protocols.

The framework supports flexible domains ranging from oncology and immunotherapy to specialized vestibular neuro-physiotherapy (e.g., Persistent Postural-Perceptual Dizziness - PPPD).

### Key Features
- **Contradiction-aware retrieval**: Surfaces risk and benefit signals.
- **Guideline Anchoring**: Support for NICE protocols, localized NHS protocols, and Physio-pedia.
- **Human-in-the-Loop Validation**: An interactive feedback UI allowing clinicians to validate agent reasoning, logging data back for continuous review.
- **Visuals**: Incorporates PyVis evidence networks and radar charts.

## Setup Instructions

### Local Development

1. Clone this repository to your local machine.
2. Ensure you have Python 3.9+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables if necessary (e.g., Gemini API keys if you wish to bypass the UI input). 
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment Instructions

### Option 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the fastest way to host this app directly from your GitHub repository.

1. Commit and push all your files to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**.
4. Select your repository, the main branch, and specify `app.py` as the Main file path.
5. In the **Advanced Settings**, ensure you add any secret keys your app requires (e.g., `GEMINI_API_KEY`).
6. Click **Deploy!** 

### Option 2: Render.com

You can also host this via Render using their web service offering.

1. Go to [Render](https://render.com/) and create a new **Web Service**.
2. Connect your GitHub repository.
3. Specify the Environment as **Python**.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Click **Create Web Service**.

## Human-in-the-loop Logging

The application maintains several log files for auditability:
- `crts_log.csv` & `crts_log.jsonl`: Audit metric logs per query.
- `human_feedback_log.jsonl`: Clinician feedback tracking AI output fidelity.

**Note**: These are ignored by Git to prevent unintentional data leakage.

## Scientific Documentation

Refer to the included [Model Card](docs/model_card.md) for full architectural design and scope overview.
