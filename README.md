# 🚀 CitySafe – AI Smart Infrastructure Monitoring System

> *"Transforming reactive infrastructure management into proactive urban intelligence."*

CitySafe is an AI-powered smart city solution designed to detect, analyze, and prioritize infrastructure vulnerabilities in real time using citizen-reported images and live environmental data.

It transforms traditional **reactive maintenance** into a **proactive, intelligent decision-support system** for urban governance.

---

## 🧠 Problem Statement

Urban infrastructure failures such as road cracks, drainage blockages, and structural damage often go unnoticed until they become critical.

There is no unified platform that:
- Collects real-time ground data from citizens
- Analyzes infrastructure conditions intelligently
- Prioritizes issues for authorities
- Predicts potential failures before they occur

---

## 💡 Our Solution

CitySafe provides an end-to-end system that:

- Enables citizens to report infrastructure issues using images
- Uses AI to classify damage severity (Damaged / Safe)
- Computes a dynamic risk score using live rainfall and traffic data
- Predicts failure timelines and explains AI decisions in plain language
- Provides a command dashboard for authorities to triage and resolve issues

---

## ⚙️ Key Features

### 👤 Citizen Dashboard
- 📸 Image upload / camera capture (JPG, PNG, WebP)
- 📍 GPS, map-click, and search-based location selection with autocomplete
- 🌧️ Real-time rainfall auto-fetch via Open-Meteo API
- 🚗 Smart traffic estimation based on time-of-day
- 🤖 AI-based damage detection (KNN model)
- 📊 Dynamic risk score with High / Moderate / Low priority badge
- 🔮 Failure probability bar + estimated failure timeline
- 💡 Explainable AI — human-readable explanation & recommended action
- 💾 Duplicate detection before saving (checks within 50m radius)
- 🔄 Re-analyze option from report history
- 📜 Report history with status tracking
- 🔔 In-app notification when an issue is resolved
- 🌙 Dark / Light theme toggle (persisted via localStorage)

---

### 🏛️ Government Dashboard (Command Center)
- 📊 Live summary analytics — Total, High, Moderate, Low risk counts
- 🗺️ Interactive map with color-coded risk markers (red / amber / green)
- 🎯 Map marker click scrolls and highlights the matching queue card
- 🔥 Priority queue sorted by risk score (Resolved pushed to bottom)
- ⚠️ "Immediate Action Required" badge on unresolved high-risk reports
- 🧠 Smart filtering by risk level, status, and free-text search
- 🔔 Live Alerts panel listing all high-risk reports (score > 80)
- 📈 Trend chart (Chart.js) — reports over time, today's point highlighted
- 📋 Per-report action log audit trail (Created → Viewed → Resolved)
- ✅ One-click mark as Resolved with backend confirmation
- 📄 PDF report generation (executive summary + top-5 risk table via jsPDF)
- 🔄 Auto-refresh every 10 seconds

---

## 🔬 AI & Analytics

### Model — K-Nearest Neighbors (KNN)
- **Input:** Image resized to 20×20 grayscale → flattened to a 400-dimensional vector
- **Output:** `Damaged` (1) or `Safe` (0), with confidence probability
- **Hyperparameter:** `n_neighbors = 7`, trained with 80/20 train-test split
- **Model file:** `backend/model/concrete_knn_model.pkl` (~1.2 MB)

### Dynamic Risk Score Formula

```
base_score  = 80  (if Damaged)  |  30  (if Safe)
risk_score  = base_score + (rainfall × 0.3) + (traffic × 0.2)
risk_score  = min(risk_score, 100)
```

| Risk Score | Priority | Recommended Action |
|-----------|----------|--------------------|
| > 80 | 🔴 High | Immediate inspection required |
| 51 – 80 | 🟡 Moderate | Schedule maintenance |
| ≤ 50 | 🟢 Low | Monitor condition |

### 🔮 Predictive Intelligence
- **Failure probability** — risk score + a small random factor (capped at 100%)
- **Estimated failure timeline** — High risk: 7 days / Medium: 15 days / Low: 30 days
- **Explainable output** — plain-language explanation + recommended action + impact rating

---

## 🌐 System Architecture

```
User (Browser)
      ↓
Landing Portal  (frontend/index.html)
      ↓                    ↓
Citizen Dashboard     Admin Dashboard
 (frontend/user/)    (frontend/admin/)
  Leaflet, Open-Meteo  Leaflet, Chart.js,
  Nominatim            jsPDF, Nominatim
      ↓                    ↓
      └────── REST API ─────┘
            (HTTP / JSON)
                 ↓
      FastAPI Backend (server.py)
      Uvicorn ASGI Server
           ↓          ↓
      KNN Model    JSON Database
    (model/*.pkl) (database/reports.json)
```

---

## 🧰 Tech Stack

### Frontend
- HTML5, CSS3 (Glassmorphism UI), Vanilla JavaScript
- [Leaflet.js v1.9.4](https://leafletjs.com/) — interactive maps
- [Chart.js](https://www.chartjs.org/) — trend analysis graph
- [jsPDF v2.5.1](https://github.com/parallax/jsPDF) — PDF generation
- [Google Fonts – Outfit](https://fonts.google.com/specimen/Outfit) — typography

### Backend
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- python-multipart, Pydantic (request validation)

### Machine Learning
- Scikit-learn (KNN classifier), NumPy, Pillow, scikit-image

### External APIs
- [Open-Meteo](https://open-meteo.com/) — live rainfall data
- [Nominatim / OpenStreetMap](https://nominatim.org/) — forward + reverse geocoding

---

## 📦 Project Structure

```
CitySafe-Dashboard/
│
├── README.md
├── requirements.txt              ← Python dependencies
├── runtime.txt                   ← Python version (3.10)
├── start.sh                      ← Render deployment command
│
├── backend/
│   ├── server.py                 ← FastAPI app & all API routes
│   ├── train_knn.py              ← KNN model training script
│   ├── model/
│   │   └── concrete_knn_model.pkl
│   └── database/
│       └── reports.json          ← Flat-file report store
│
├── frontend/
│   ├── index.html                ← Landing / portal page
│   ├── style.css
│   ├── user/                     ← Citizen dashboard
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   └── admin/                    ← Government command center
│       ├── index.html
│       ├── app.js
│       └── styles.css
│
└── dataset/
    ├── Positive/                 ← Damaged concrete images (label: 1)
    └── Negative/                 ← Intact concrete images (label: 0)
```

---

## 🚀 Getting Started

### 🔧 Backend Setup

```bash
# Navigate to the backend folder
cd backend

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Start the dev server
uvicorn server:app --reload
```

API runs at: `http://127.0.0.1:8000`  
Swagger docs: `http://127.0.0.1:8000/docs`

### 🌐 Frontend

Open with **Live Server** in VS Code:

```
Right-click frontend/index.html → "Open with Live Server"
```

Or use Python's built-in server:

```bash
cd frontend
python -m http.server 5500
# Open: http://localhost:5500
```

### 🧪 Re-train the Model (Optional)

```bash
# Add images to dataset/Positive/ and dataset/Negative/
python backend/train_knn.py
# Then move the output model file:
move concrete_knn_model.pkl backend/model/
```

---

## 🌍 Deployment

| Layer | Platform | Notes |
|-------|----------|-------|
| Frontend | Vercel | Static deployment |
| Backend | Render | Uses `start.sh` + `runtime.txt` |
| Database | JSON file | Prototype stage only |

> ⚠️ After deploying to Render, update the API base URL in `frontend/user/app.js` and `frontend/admin/app.js` from `http://127.0.0.1:8000` to your Render service URL.

---

## ⚠️ Known Limitations

- JSON flat file is not safe for concurrent writes under heavy load
- No user authentication — both dashboards are open access
- Images are not stored; re-analysis uses a mock canvas placeholder
- Traffic estimation is time-of-day heuristic, not real traffic data
- CORS is set to `allow_origins=["*"]` — not suitable for production

---

## 🔮 Future Scope

- 🧠 Deep Learning model (CNN / MobileNet) for higher accuracy
- 🗄️ PostgreSQL / Supabase for production-grade database
- 🔐 Role-based authentication (Citizen / Authority)
- 📱 Mobile application (React Native or Flutter)
- 🌡️ Real-time IoT sensor data integration
- 🚦 Live traffic data (Google Maps / HERE API)
- 📊 Heatmaps and predictive maintenance scheduling

---

## 🏆 Impact

CitySafe enables:

- Faster response to infrastructure failures before they escalate
- Data-driven decision making for urban authorities
- Reduced maintenance cost through early detection
- Improved public safety in high-density areas
- Full transparency with timestamped action logs

---

## 👨‍💻 Team

| Name | 
|------|
| Sathya R V |
| S Shayden |
| Sudharsan S M |
| Tajmal Hussain S |

---

## 🎯 Project Type

**Smart India Hackathon (SIH)**
- **Theme:** Disaster Management – Risk Mitigation and Planning
- **Category:** Software
- **Domain:** Smart Cities / Urban Infrastructure

---

> 📌 *"Transforming reactive infrastructure management into proactive urban intelligence."*