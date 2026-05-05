# 🚀 CitySafe – AI Smart Infrastructure Monitoring System

> *"Transforming reactive infrastructure management into proactive urban intelligence."*

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/ML-KNN%20%7C%20Scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Leaflet](https://img.shields.io/badge/Map-Leaflet.js-brightgreen?logo=leaflet)](https://leafletjs.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

CitySafe is an AI-powered smart city solution that detects, analyzes, and prioritizes infrastructure vulnerabilities in real time using citizen-reported images and live environmental data.

It transforms traditional **reactive maintenance** into a **proactive, intelligent decision-support system** for urban governance.

---

## 📑 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Our Solution](#-our-solution)
3. [Key Features](#️-key-features)
4. [AI & Analytics](#-ai--analytics)
5. [Risk Scoring Formula](#-risk-scoring-formula)
6. [System Architecture](#-system-architecture)
7. [Tech Stack](#️-tech-stack)
8. [Project Structure](#-project-structure)
9. [Dataset](#-dataset)
10. [API Reference](#-api-reference)
11. [Data Schema](#-data-schema)
12. [Getting Started](#-getting-started)
13. [Deployment](#-deployment)
14. [Environment & Configuration](#-environment--configuration)
15. [Known Limitations](#️-known-limitations)
16. [Future Scope](#-future-scope)
17. [Impact](#-impact)
18. [Team](#-team)
19. [Project Type](#-project-type)

---

## 🧠 Problem Statement

Urban infrastructure failures — road cracks, drainage blockages, structural damage — often go unnoticed until they become critical and costly.

There is no unified platform that:
- Collects real-time ground data from citizens
- Analyzes infrastructure conditions intelligently
- Prioritizes issues for authorities
- Predicts potential failures before they occur

---

## 💡 Our Solution

CitySafe provides a full end-to-end pipeline:

| Step | What Happens |
|------|-------------|
| 1 | Citizen uploads an image of damaged infrastructure |
| 2 | AI model (KNN) classifies the image as **Damaged** or **Safe** |
| 3 | Live weather (rainfall) and time-based traffic are factored in |
| 4 | A **dynamic risk score** (0–100) is computed |
| 5 | Failure probability and estimated timeline are predicted |
| 6 | The report is saved and surfaced to government authorities |
| 7 | Authorities triage, resolve, and log all actions |

---

## ⚙️ Key Features

### 👤 Citizen Dashboard (`/frontend/user/`)

| Feature | Detail |
|---------|--------|
| 📸 Image Upload / Camera Capture | Supports JPG, PNG, WebP; uses `capture="environment"` for mobile camera |
| 📍 GPS Location | `navigator.geolocation` API with live map centering |
| 🗺️ Map Click Selection | Interactive Leaflet.js map; click anywhere to set location |
| 🔍 Location Search with Autocomplete | Debounced Nominatim search with live dropdown suggestions |
| 🌧️ Auto Rainfall Fetch | Open-Meteo API (`precipitation`, `rain` fields); falls back to random value if API is unavailable |
| 🚗 Smart Traffic Estimation | Time-of-day logic (morning/evening peak hours) + latitude-based urban boost |
| 🤖 AI Damage Analysis | Sends image + rainfall + traffic to FastAPI; receives full prediction payload |
| 📊 Risk Score Display | Visual score badge with High / Moderate / Low priority label |
| 🔮 Failure Probability Bar | Animated progress bar color-coded by severity |
| 💡 Explainable AI Output | Human-readable explanation + recommended action + impact score |
| 💾 Save Report | One-click save to backend; includes **duplicate detection** within 50m radius |
| 🔄 Re-Analyze | Re-run AI analysis on a past report from history view |
| 📜 Report History | All submitted reports with timestamps, status badges |
| 🔔 Resolution Notification | Toast notification when any report is marked Resolved (checked per session) |
| 🌙 Dark / Light Theme Toggle | Persisted via `localStorage` |
| 💬 Guided Tooltips | First-visit onboarding tooltips that auto-dismiss after 7 seconds |

---

### 🏛️ Government Dashboard (`/frontend/admin/`) – Command Center

| Feature | Detail |
|---------|--------|
| 📊 Summary Analytics | Live counters: Total, High Risk, Moderate, Low Risk reports |
| 🗺️ Interactive Map | Color-coded SVG markers (red/amber/green); custom icons per risk level |
| 🎯 Map ↔ Queue Linking | Clicking a map marker scrolls and highlights its queue card |
| 🔥 Priority Queue | Sorted by risk score (High → Low); Resolved reports pushed to bottom |
| ⚠️ Urgent Badge | "Immediate Action Required" badge on High-risk unresolved reports |
| 🧠 Smart Filtering | Filter by risk level, status (Pending/Resolved), and free-text search |
| 🔔 Live Alerts Panel | Lists all reports with risk score > 80 with reverse-geocoded location names |
| 📈 Trend Chart | Chart.js line chart grouping reports by date; today's point highlighted in red |
| 📋 Action Log | Per-report audit trail: Created → Viewed → Marked as Resolved (with timestamps) |
| ✅ Mark as Resolved | One-click resolution; auto-refreshes queue with backend confirmation |
| 📄 PDF Report Generation | jsPDF-powered export: executive summary, critical alerts, top-5 risk table |
| 🔄 Auto-Refresh | Polls backend every **10 seconds** for live data updates |

---

## 🔬 AI & Analytics

### Model: K-Nearest Neighbors (KNN)

- **Training script:** `backend/train_knn.py`
- **Model file:** `backend/model/concrete_knn_model.pkl` (~1.2 MB)
- **Input:** 20×20 grayscale flattened image → 400-dimensional feature vector
- **Output:** Binary classification — `0` (Negative / Safe) or `1` (Positive / Damaged)
- **Hyperparameter:** `n_neighbors=7`
- **Train/Test split:** 80 / 20 (stratified with `random_state=42`)

### Image Preprocessing Pipeline

```
Raw Image (any format)
    → Convert to RGB (PIL)
    → Convert to Grayscale (skimage.color.rgb2gray)
    → Resize to 20×20 pixels (skimage.transform.resize)
    → Flatten to 1D array of 400 values (np.ravel)
    → Feed to KNN model
```

### 🔮 Predictive Intelligence

| Output | Method |
|--------|--------|
| **Failure Probability** | `min(100, risk_score + random(0, 10))` |
| **Estimated Failure Days** | Risk > 80 → 7 days; Risk > 60 → 15 days; else → 30 days |
| **Explanation** | Rule-based text based on prediction class |
| **Recommended Action** | Rule-based based on risk tier |
| **Impact Score** | High / Moderate / Low based on risk tier |

---

## 📐 Risk Scoring Formula

The dynamic risk score is computed on the backend for every analysis:

```
base_score = 80  (if prediction == "Damaged")
           = 30  (if prediction == "Safe")

risk_score = base_score + (rainfall × 0.3) + (traffic × 0.2)
risk_score = min(risk_score, 100)   ← capped at 100
```

**Priority Tiers:**

| Risk Score | Priority | Action |
|-----------|----------|--------|
| > 80 | 🔴 High | Immediate inspection required |
| 51 – 80 | 🟡 Moderate | Schedule maintenance |
| ≤ 50 | 🟢 Low | Monitor condition |

---

## 🌐 System Architecture

```
┌─────────────────────────────────┐
│          User (Browser)         │
│  frontend/index.html (Portal)   │
└──────────┬──────────────────────┘
           │
    ┌──────┴──────────────────┐
    │                         │
┌───▼────────────┐   ┌────────▼──────────┐
│  Citizen View  │   │  Government View  │
│ frontend/user/ │   │ frontend/admin/   │
│ (Leaflet, Open │   │ (Leaflet, Chart.js│
│  Meteo, Nomin.)│   │  jsPDF, Nomin.)   │
└───────┬────────┘   └────────┬──────────┘
        │                     │
        └────────┬────────────┘
                 │  HTTP (REST)
        ┌────────▼────────────┐
        │   FastAPI Backend   │
        │   backend/server.py │
        │   (Uvicorn ASGI)    │
        └──────┬──────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼──────┐  ┌─────▼──────────────┐
│  KNN Model  │  │   JSON Database    │
│ model/*.pkl │  │ database/reports.  │
│ (Scikit-    │  │ json               │
│  learn)     │  │ (Flat file store)  │
└─────────────┘  └────────────────────┘
```

---

## 🧰 Tech Stack

### Frontend

| Technology | Purpose |
|-----------|---------|
| HTML5 | Semantic structure |
| CSS3 (Glassmorphism) | Dark UI with glass effect panels |
| Vanilla JavaScript (ES2022) | All business logic, API calls |
| [Leaflet.js v1.9.4](https://leafletjs.com/) | Interactive maps, markers, popups |
| [Chart.js](https://www.chartjs.org/) | Trend analysis line chart with gradient fill |
| [jsPDF v2.5.1](https://github.com/parallax/jsPDF) | Client-side PDF generation |
| [Google Fonts – Outfit](https://fonts.google.com/specimen/Outfit) | Typography (300–700 weight) |

### Backend

| Technology | Purpose |
|-----------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | REST API framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [python-multipart](https://pypi.org/project/python-multipart/) | Multipart form / file upload handling |
| [Pydantic](https://docs.pydantic.dev/) | Request body validation (BaseModel) |

### Machine Learning

| Technology | Purpose |
|-----------|---------|
| [Scikit-learn](https://scikit-learn.org/) | KNN classifier, train/test split, evaluation |
| [NumPy](https://numpy.org/) | Array manipulation, feature flattening |
| [Pillow (PIL)](https://pillow.readthedocs.io/) | Image decoding and RGB conversion |
| [scikit-image](https://scikit-image.org/) | Grayscale conversion, image resizing |

### External APIs

| API | Used For |
|-----|---------|
| [Open-Meteo](https://open-meteo.com/) | Real-time rainfall / precipitation data |
| [Nominatim (OpenStreetMap)](https://nominatim.org/) | Forward geocoding (search) + reverse geocoding (location names) |

---

## 📦 Project Structure

```
CitySafe-Dashboard/
│
├── 📄 README.md                     ← This file
├── 📄 requirements.txt              ← Python dependencies
├── 📄 runtime.txt                   ← Python version pin (python-3.10)
├── 📄 start.sh                      ← Production start command (Render)
│
├── 🗂️ backend/
│   ├── server.py                    ← FastAPI app with all API routes
│   ├── train_knn.py                 ← KNN model training script
│   ├── test_img.png                 ← Sample grayscale test image
│   ├── test_rgb.jpg                 ← Sample RGB test image
│   │
│   ├── model/
│   │   └── concrete_knn_model.pkl   ← Pre-trained KNN model (~1.2 MB)
│   │
│   └── database/
│       └── reports.json             ← Persistent flat-file report store
│
├── 🗂️ frontend/
│   ├── index.html                   ← Landing / Portal page (User vs Admin)
│   ├── style.css                    ← Landing page styles (glassmorphism)
│   │
│   ├── user/
│   │   ├── index.html               ← Citizen Reporting Dashboard
│   │   ├── app.js                   ← All citizen-side logic (~727 lines)
│   │   └── styles.css               ← Citizen dashboard styles (~19k)
│   │
│   └── admin/
│       ├── index.html               ← Government Command Center
│       ├── app.js                   ← All admin-side logic (~560 lines)
│       └── styles.css               ← Admin dashboard styles (~11k)
│
└── 🗂️ dataset/
    ├── Positive/                    ← Damaged concrete images (label: 1)
    └── Negative/                    ← Intact concrete images (label: 0)
```

---

## 🗃️ Dataset

The KNN model was trained on a binary image dataset of concrete surfaces:

| Class | Folder | Label | Description |
|-------|--------|-------|-------------|
| Damaged | `dataset/Positive/` | `1` | Cracked / deteriorated concrete |
| Safe | `dataset/Negative/` | `0` | Intact / undamaged concrete |

**Preprocessing applied during training:**
- RGB → Grayscale conversion (`skimage.color.rgb2gray`)
- Resize to **20 × 20** pixels
- Flatten to **400-dimensional** feature vector

**To retrain the model:**
```bash
# Run from the project root
python backend/train_knn.py
# Move the generated model file
move concrete_knn_model.pkl backend/model/
```

---

## 📡 API Reference

Base URL (local): `http://127.0.0.1:8000`

### `POST /api/analyze`

Analyze an infrastructure image and compute risk score.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File (image) | ✅ | JPG, PNG, or WebP image |
| `rainfall` | float | ✅ | Current rainfall in mm |
| `traffic` | float | ✅ | Estimated traffic volume (0–100) |

**Response:** `application/json`

```json
{
  "prediction": "Damaged",
  "confidence": 0.857,
  "risk_score": 100,
  "priority": "High",
  "failure_probability": 103,
  "estimated_failure_days": 7,
  "explanation": "Structural damage detected under current environmental stress.",
  "recommended_action": "Immediate inspection required",
  "impact_score": "High (dense impact area)"
}
```

---

### `POST /api/report/save`

Save a completed analysis as a permanent report.

**Request:** `application/json`

```json
{
  "image_name": "crack_photo.jpg",
  "prediction": "Damaged",
  "risk_score": 95,
  "latitude": 12.9716,
  "longitude": 77.5946,
  "failure_probability": 99,
  "estimated_failure_days": 7,
  "explanation": "Structural damage detected...",
  "recommended_action": "Immediate inspection required",
  "impact_score": "High (dense impact area)"
}
```

**Response:** `application/json`

```json
{
  "status": "success",
  "report": {
    "id": "uuid-v4",
    "status": "Pending",
    "timestamp": "2026-05-05T10:00:00.000000",
    "action_log": [{ "action": "Created", "timestamp": "..." }]
  }
}
```

---

### `GET /api/reports`

Fetch all submitted reports.

**Response:** JSON array of all report objects (see [Data Schema](#-data-schema)).

---

### `PUT /api/report/update/{id}`

Update a report's status (e.g., mark as Resolved).

**Request:** `application/json`

```json
{ "status": "Resolved" }
```

**Response:** Updated report object with appended action log entry.

---

### `PUT /api/report/log_view/{id}`

Log a "Viewed" event in a report's action log (prevents duplicate spam).

**Response:** `{ "status": "success" }`

---

### `GET /api/notifications`

Fetch all high-risk reports (risk score > 80) for the alert panel.

**Response:** JSON array of matching report objects.

---

## 🗄️ Data Schema

Each report stored in `database/reports.json`:

```json
{
  "id": "uuid-v4-string",
  "image_name": "filename.jpg",
  "prediction": "Damaged | Safe",
  "risk_score": 0,
  "latitude": 0.0,
  "longitude": 0.0,
  "failure_probability": 0,
  "estimated_failure_days": 0,
  "explanation": "...",
  "recommended_action": "...",
  "impact_score": "High | Moderate | Low",
  "timestamp": "ISO-8601 datetime",
  "status": "Pending | Resolved",
  "action_log": [
    { "action": "Created",          "timestamp": "..." },
    { "action": "Viewed",           "timestamp": "..." },
    { "action": "Marked as Resolved", "timestamp": "..." }
  ]
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- A modern browser (Chrome, Firefox, Edge)
- [Live Server extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) (VS Code) — recommended for frontend

---

### 🔧 Backend Setup

```bash
# 1. Navigate to the backend directory
cd backend

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the development server
uvicorn server:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

Interactive API docs: `http://127.0.0.1:8000/docs` (Swagger UI)

---

### 🌐 Frontend Setup

Open the frontend using **Live Server** in VS Code:

```
Right-click frontend/index.html → "Open with Live Server"
```

Or serve directly:

```bash
# From the frontend directory, using Python's built-in server
cd frontend
python -m http.server 5500
# Open: http://localhost:5500
```

**Portal Entry Points:**

| Page | Path |
|------|------|
| 🏠 Landing Portal | `frontend/index.html` |
| 👤 Citizen Dashboard | `frontend/user/index.html` |
| 🏛️ Admin Command Center | `frontend/admin/index.html` |

---

### 🧪 Re-train the Model (Optional)

To train a fresh KNN model on your own dataset:

```bash
# 1. Populate the dataset folders
# Place damaged images in:  dataset/Positive/
# Place intact images in:   dataset/Negative/

# 2. Run training from the project root
python backend/train_knn.py

# 3. Move the output model to the correct location
move concrete_knn_model.pkl backend/model/
```

Expected training output:
```
Initializing Training Pipeline...
Loading and preprocessing images...
Successfully loaded N images.
Training set size: ...
Testing set size:  ...
Training the K-Nearest Neighbors model...
Evaluating model...
              precision    recall  f1-score  ...
Success! Model saved locally as 'concrete_knn_model.pkl'
```

---

## 🌍 Deployment

| Layer | Platform | Config File |
|-------|----------|-------------|
| Frontend | [Vercel](https://vercel.com/) | — |
| Backend | [Render](https://render.com/) | `start.sh`, `runtime.txt` |
| Database | JSON flat file | `backend/database/reports.json` |

### Render Deployment (`start.sh`)

```bash
uvicorn server:app --host 0.0.0.0 --port 10000
```

### Python Version (`runtime.txt`)

```
python-3.10
```

> **Note:** After deploying the backend to Render, update the API base URL in both `frontend/user/app.js` and `frontend/admin/app.js` from `http://127.0.0.1:8000` to your Render service URL.

---

## 🔧 Environment & Configuration

No `.env` file is required for the prototype. Key configuration values are set directly in the source:

| Setting | Location | Default |
|---------|----------|---------|
| Backend API URL (citizen) | `frontend/user/app.js` line 336 | `http://127.0.0.1:8000` |
| Backend API URL (admin) | `frontend/admin/app.js` line 98 | `http://127.0.0.1:8000` |
| Reports file path | `backend/server.py` line 19 | `database/reports.json` |
| Model file path | `backend/server.py` line 20 | `model/concrete_knn_model.pkl` |
| Admin auto-refresh interval | `frontend/admin/app.js` line 558 | `10000` ms (10 s) |
| Duplicate detection radius | `frontend/user/app.js` line 389 | `0.05` km (50 m) |
| KNN neighbors | `backend/train_knn.py` line 62 | `7` |
| Image resize dimensions | `backend/train_knn.py` line 15–16 | `20 × 20` px |
| Notification threshold | `backend/server.py` line 228 | Risk score `> 80` |

---

## ⚠️ Known Limitations

- **Flat-file database** — `reports.json` is a single file; not safe for concurrent writes under heavy load
- **No authentication** — both dashboards are open access (no login / session management)
- **No image storage** — images are analyzed but not persisted; re-analysis uses a mock canvas blob
- **CORS wildcard** — backend accepts all origins (`allow_origins=["*"]`); not recommended for production
- **Traffic estimation is approximate** — time-of-day heuristic, not real traffic data
- **Single-node deployment** — not horizontally scalable in current form

---

## 🔮 Future Scope

- 🧠 **Deep Learning model** — CNN / MobileNet for higher accuracy and richer feature extraction
- 🗄️ **Production database** — PostgreSQL / Supabase with proper schema and migrations
- 🔐 **Authentication** — Role-based access (Citizen / Authority) with JWT sessions
- 📱 **Mobile application** — React Native or Flutter client
- 🌡️ **IoT sensor integration** — Structural health monitoring feeds
- 🚦 **Real traffic data** — Integration with Google Maps Traffic or HERE API
- 🌍 **Multi-language support** — Localization for regional languages
- 📊 **Advanced analytics** — Heatmaps, predictive maintenance scheduling
- 🔄 **Real-time sync** — WebSocket-based live updates instead of polling

---

## 🏆 Impact

CitySafe enables:

- ⚡ **Faster response** to infrastructure failures before they become critical
- 📊 **Data-driven decision making** for urban authorities
- 💰 **Reduced maintenance cost** through early detection
- 🛡️ **Improved public safety** in high-density urban areas
- 📋 **Transparent accountability** through auditable action logs

---

## 👨‍💻 Team

| Name | Role |
|------|------|
| Sathya R V | Developer / AI Integration |
| S Shayden | Developer |
| Sudharsan S M | Developer |
| Tajmal Hussain S | Developer |

---

## 🎯 Project Type

**Smart India Hackathon (SIH)**

| Field | Value |
|-------|-------|
| 🏷️ Theme | Disaster Management – Risk Mitigation and Planning |
| 💻 Category | Software |
| 🏙️ Domain | Smart Cities / Urban Infrastructure |