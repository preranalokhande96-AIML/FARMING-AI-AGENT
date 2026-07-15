# 🌾 Smart Farming Agent

> AI-powered farming advisory powered by **IBM Watsonx.ai (Granite)** and **Flask**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![IBM Watsonx.ai](https://img.shields.io/badge/IBM-Watsonx.ai-0f62fe.svg)](https://www.ibm.com/watsonx)

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [IBM Watsonx.ai Setup](#ibm-watsonxai-setup)
- [Agent Customisation](#agent-customisation)
- [Running Locally](#running-locally)
- [Deployment (Production)](#deployment-production)
- [API Reference](#api-reference)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Chat Agent** | Multilingual farming advisory (English / Hindi / Marathi) |
| 🌱 **Crop Advisory** | Sowing, growth stage, harvesting guidance for 10+ Indian crops |
| 💊 **Fertiliser Schedule** | Soil & season-aware NPK recommendations |
| 💧 **Irrigation Plan** | Water requirement calculator by crop & weather |
| 🐛 **Pest & Disease ID** | Upload a crop photo for identification |
| 🌡️ **Weather-based Tips** | Temperature & rainfall-aware farming advice |
| 📊 **MSP/Market Prices** | Minimum Support Price awareness for Indian mandis |
| 🏛️ **Govt. Schemes** | PM-KISAN, PMFBY, and other agricultural scheme info |
| 🎙️ **Voice Input** | Web Speech API — speak in Hindi, Marathi, or English |
| 🌙 **Dark Mode** | System-aware + manual toggle |

---

## 📁 Project Structure

```
SmartFarmingAgent/
├── app.py                  ← Main Flask backend + AGENT_INSTRUCTIONS
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
├── .env                    ← Your secrets (gitignored)
├── templates/
│   └── index.html          ← Full SPA frontend
└── static/
    ├── css/
    │   └── style.css       ← Custom styles + dark mode
    └── js/
        └── main.js         ← Frontend logic, chat, API calls
```

---

## 🔧 Prerequisites

- Python **3.10** or higher
- `pip` package manager
- IBM Cloud account with **Watsonx.ai** access
- (Optional) OpenWeatherMap API key for live weather

---

## 🚀 Quick Start

```bash
# 1. Clone or download the project
cd SmartFarmingAgent

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env        # Windows
cp .env.example .env          # macOS/Linux

# 5. Edit .env with your IBM API credentials (see below)
# 6. Run the application
python app.py

# 7. Open browser
# http://localhost:5000
```

---

## 🔑 IBM Watsonx.ai Setup

### Step 1 — Create IBM Cloud Account
1. Go to [cloud.ibm.com](https://cloud.ibm.com) and sign up (free tier available)
2. Navigate to **Catalog → AI / Machine Learning → Watsonx.ai**
3. Create a Watsonx.ai instance

### Step 2 — Get API Key
1. Go to **Manage → Access (IAM) → API keys**
2. Click **Create an IBM Cloud API key**
3. Copy the key immediately (shown only once)

### Step 3 — Get Project ID
1. In Watsonx.ai Studio, create a new **Project**
2. Go to **Project Settings → General**
3. Copy the **Project ID**

### Step 4 — Update `.env`
```env
IBM_API_KEY=your_actual_api_key_here
IBM_PROJECT_ID=your_actual_project_id_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=some-random-secure-string
```

> ⚠️ **Never commit your `.env` file to version control.**  
> The `.gitignore` already excludes it.

### Supported Models
| Model ID | Description |
|----------|-------------|
| `ibm/granite-13b-instruct-v2` | Default — best balance (recommended) |
| `ibm/granite-3-8b-instruct` | Faster, lighter model |
| `ibm/granite-20b-multilingual` | Best for Hindi/Marathi responses |

Change `MODEL_ID` in `app.py` under **AGENT_INSTRUCTIONS**.

---

## 🎛️ Agent Customisation

Open `app.py` and find the `AGENT_INSTRUCTIONS` block:

```python
# ── AGENT INSTRUCTIONS — EDIT THIS BLOCK ──────────────────────────────────

AGENT_TONE = "friendly"
# "formal" | "friendly" | "concise"

CROP_SPECIALIZATIONS = [
    "cotton", "wheat", "sugarcane", "rice",
    "soybean", "onion", "tur dal", "chickpea"
]

LANGUAGE_SUPPORT = "English"
# "English" | "Hindi" | "Marathi"

REGION = "Maharashtra, India"
# Change to: "Punjab", "Uttar Pradesh", "Rajasthan", etc.

SAFETY_RULES = {
    "never_dose_without_disclaimer": True,   # ⚠️ Pesticide safety disclaimer
    "always_kvk_for_critical":       True,   # KVK recommendation for critical issues
    "no_financial_advice":           True,   # No guaranteed market price advice
}

MAX_TOKENS  = 1024      # Longer = more detailed responses
TEMPERATURE = 0.7       # 0.0=deterministic, 1.0=creative
MODEL_ID    = "ibm/granite-13b-instruct-v2"
```

### Examples

**For Punjab wheat farmer (Hindi):**
```python
CROP_SPECIALIZATIONS = ["wheat", "rice", "maize", "sunflower"]
LANGUAGE_SUPPORT     = "Hindi"
REGION               = "Punjab, India"
AGENT_TONE           = "friendly"
```

**For Maharashtra cotton farmer (Marathi):**
```python
CROP_SPECIALIZATIONS = ["cotton", "soybean", "tur dal", "onion"]
LANGUAGE_SUPPORT     = "Marathi"
REGION               = "Vidarbha, Maharashtra"
```

---

## 🖥️ Running Locally

```bash
# Development mode
python app.py

# With hot-reload
FLASK_DEBUG=True python app.py
```

Access at: `http://localhost:5000`

---

## 🌐 Deployment (Production)

### Option 1 — Gunicorn + Nginx (Linux VPS)

```bash
# Install
pip install gunicorn

# Run
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or with timeout for LLM calls
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
```

**Nginx config snippet:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

### Option 2 — Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

```bash
docker build -t smart-farming-agent .
docker run -p 5000:5000 --env-file .env smart-farming-agent
```

### Option 3 — IBM Code Engine / Cloud Foundry

```bash
# Install IBM Cloud CLI
# ibmcloud login
# ibmcloud target --cf

# Push to Cloud Foundry
ibmcloud cf push smart-farming-agent -m 512M
```

### Option 4 — Render / Railway (Free tier)

1. Push code to GitHub (ensure `.env` is gitignored)
2. Connect repo to [render.com](https://render.com) or [railway.app](https://railway.app)
3. Set environment variables in the platform dashboard
4. Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT --timeout 120 app:app`

---

## 📡 API Reference

### `POST /api/chat`
Send a message to the farming agent.

**Request:**
```json
{
  "message": "What fertilizer for cotton?",
  "image": "data:image/jpeg;base64,...",
  "farm_context": {
    "crop": "cotton",
    "soil_type": "black cotton soil",
    "soil_ph": "6.8",
    "weather": "38°C",
    "location": "Yavatmal"
  }
}
```

**Response:**
```json
{
  "response": "For cotton on black soil...",
  "mode": "live",
  "turn_count": 3
}
```

---

### `POST /api/quick-advice`
Get instant advisory for a category.

| category | Description |
|----------|-------------|
| `fertilizer` | Fertiliser schedule |
| `irrigation` | Irrigation plan |
| `pest` | Pest & disease guide |
| `weather` | Weather-based tips |
| `market` | MSP & mandi prices |
| `schemes` | Government schemes |

---

### `POST /api/update-farm-context`
Update farm profile (persisted in session).

### `POST /api/clear-chat`
Clear conversation history.

### `GET /api/status`
Health check & configuration info.

---

## 🔒 Security Notes

- API keys are loaded via `python-dotenv` from `.env` — never hardcoded
- `.env` is excluded via `.gitignore`
- Session data is stored server-side (Flask sessions)
- No user data is stored persistently without authentication

---

## 🤝 Support

For IBM Watsonx.ai issues: [IBM Cloud Support](https://cloud.ibm.com/unifiedsupport/supportcenter)  
For crop/agriculture queries: Contact your local **Krishi Vigyan Kendra (KVK)**  
ICAR KVK Directory: [kvk.icar.gov.in](https://kvk.icar.gov.in)

---

*Built with ❤️ for Indian farmers | Powered by IBM Watsonx.ai*
