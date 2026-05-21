# 🎯 Filipino ASR Evaluation Tool — AI Aligner

A web-based tool for evaluating Filipino ASR (Automatic Speech Recognition) results with AI-powered translation alignment using Google Gemini.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **ASR Alignment** — Compare True Text vs ASR Result with word-level diff highlighting
- **ASR Splitter** — Drag & drop `.txt` files to auto-split into ASR and Translation halves
- **AI Translation Alignment** — Use Google Gemini to align English translations to ASR segments
- **Real-time Progress** — Live log panel with progress bar during AI processing
- **Editable Results** — Click any cell in the results table to edit inline
- **Re-evaluate** — Recalculate diffs and WER after editing
- **Excel Export** — Export results with color-coded differences to `.xlsx`
- **Dark Mode** — Toggle between light and dark themes
- **Line Numbers** — Side-by-side line numbers for easy reference

## 🚀 Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/masusbilla4/asr_web_aligner.git
cd asr-eval

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → Sign up with GitHub
3. Click **New** → **Web Service** → Connect your repo
4. Render auto-detects settings from `render.yaml`
5. Click **Deploy** — you'll get a live URL like `https://asr-eval.onrender.com`

## 📖 How to Use

1. **Paste** True Text (reference) and ASR Result (hypothesis) into the input areas
2. **Click** "▶ Align ASR" to run word-level diff alignment
3. **Review** results in the table — WER%, differences highlighted in red/blue
4. **Optional**: Paste English Translation and click "🌐 Align Translation (AI)" to align translations using Gemini
5. **Edit** any cell by clicking on it — then click "🔄 Re-evaluate" to recalculate
6. **Export** to Excel with color-coded differences

### ASR Splitter

Drop `.txt` files into the splitter zone — files are sorted A→Z, first half becomes ASR lines, second half becomes Translation lines.

## 🔑 API Key

AI features require a **Google Gemini API key**. Enter it in the modal when prompted.

Get one at: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## 🛠 Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML/CSS/JavaScript (no framework, single-page)
- **AI**: Google Gemini (via `google-genai`)
- **Export**: openpyxl (Excel with Rich Text formatting)

## 📁 Project Structure

```
asr_web_app/
├── app.py              # Flask server & API routes
├── alignment_engine.py # ASR alignment & diff logic
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── .gitignore
├── README.md
├── templates/
│   └── index.html      # Single-page web UI
└── exports/            # Generated Excel/JSON exports (gitignored)
```

## 📊 Scoring Guide

| Score | Description |
|-------|-------------|
| 3 | No wrong/missing/additional word. Perfect sentence. |
| 2.5 | Some errors but no problem understanding the sentence. |
| 2 | Some errors + grammar issues, but overall meaning is clear. |
| 1.5 | Partially able to guess and understand subject/details. |
| 1 | Unable to understand or no text transcribed. |

## 📄 License

MIT License — feel free to use and modify.
