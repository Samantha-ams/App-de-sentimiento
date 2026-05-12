# Sentiment Analysis App - FilingLens AI

A Flask web app for sentiment analysis on SEC financial filings such as 10-K, 10-Q, and 8-K reports. It can analyze uploaded filing files or fetch filings from SEC EDGAR by ticker.

## Live Preview

Static GitHub Pages preview: https://samantha-ams.github.io/App-de-sentimiento/

> GitHub Pages only serves static files. It cannot run Python, Flask, uploads, or the local AI model. Use the local setup below for the full analysis workflow.

## Screenshot

![App screenshot](static/preview.png)

## What Is Included

Only the files needed for the app and the GitHub Pages preview are included:

- `app.py` - Flask backend and sentiment analysis logic.
- `requirements.txt` - Python dependencies.
- `templates/index.html` - Flask page used when running locally.
- `static/style.css` - App styling.
- `static/script.js` - Frontend behavior.
- `static/preview.png` - Screenshot shown in this README.
- `index.html` - Static preview for GitHub Pages.
- `uploads/.gitkeep` - Keeps the local upload folder available without uploading user files.
- `.gitignore` - Prevents temporary files, ZIPs, caches, and uploads from being committed.

## Run Locally

```bash
git clone https://github.com/Samantha-ams/App-de-sentimiento.git
cd App-de-sentimiento
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## Features

- Sentiment classification: Positive, Neutral, or Negative.
- File upload support for `.txt`, `.html`, and `.htm` filings.
- SEC EDGAR filing lookup by ticker.
- Chunked processing for long documents.
- Flask backend with a vanilla HTML, CSS, and JavaScript frontend.

## Tech Stack

- Python
- Flask
- HuggingFace Transformers
- BeautifulSoup
- HTML, CSS, and JavaScript
- GitHub Pages for the static preview
