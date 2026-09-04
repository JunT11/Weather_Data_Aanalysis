# Weather Analysis – Cloud Deployment

This folder contains the optimized Streamlit application for classroom use.

## Required application files

Place these files under `app/`:

- `Weather_Analysis.py`
- `model_wo_power.sav`
- `ai_chara.png`

The current package already contains `Weather_Analysis.py`. Copy the model and character image into the same `app/` directory before deployment.

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this folder.
3. In Streamlit Community Cloud, select the repository.
4. Set the main file to:
   `app/Weather_Analysis.py`
5. Deploy.

`requirements.txt` is used automatically.

## Google Cloud Run

From this folder:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/weather-analysis

gcloud run deploy weather-analysis \
  --image gcr.io/YOUR_PROJECT_ID/weather-analysis \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --port 8080
```

For a classroom deployment, start with 1 minimum instance and test with 10, 20, then 30 simultaneous users. Increase CPU/memory or maximum instances if needed.

## Important

This package improves application-side resource usage, but the folder itself cannot guarantee 30 simultaneous users. Actual capacity depends on CSV size, chart workload, model prediction time, Cloud Run/Streamlit resources, and the number of simultaneous sessions.

## File layout

```text
Weather_Analysis_Cloud_Deploy/
├─ app/
│  ├─ Weather_Analysis.py
│  ├─ model_wo_power.sav
│  └─ ai_chara.png
├─ .streamlit/
│  └─ config.toml
├─ .dockerignore
├─ .gitignore
├─ Dockerfile
├─ packages.txt
├─ requirements.txt
└─ README.md
```
