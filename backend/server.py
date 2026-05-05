import os
import json
import uuid
import pickle
import numpy as np
import random
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import skimage.transform
import skimage.color

os.makedirs("model", exist_ok=True)
os.makedirs("database", exist_ok=True)

REPORTS_FILE = "database/reports.json"
MODEL_FILE = "model/concrete_knn_model.pkl"

if not os.path.exists(REPORTS_FILE):
    with open(REPORTS_FILE, "w") as f:
        json.dump([], f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
try:
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Warning: Model file not found at {MODEL_FILE}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    rainfall: float = Form(...),
    traffic: float = Form(...)
):
    if model is None:
        return {"error": "Model not loaded on server."}

    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return {"error": "Invalid image format"}
            
        img_array = np.array(image)
        gray_image = skimage.color.rgb2gray(img_array)
            
        resized = skimage.transform.resize(gray_image, (20, 20))
        flattened_img = np.asarray(resized).ravel()
        
        prediction_int = int(model.predict([flattened_img])[0])
        try:
            probabilities = model.predict_proba([flattened_img])[0]
            confidence = float(max(probabilities))
        except:
            confidence = 0.8
            
        prediction = "Damaged" if prediction_int == 1 else "Safe"
        
        base_score = 80 if prediction == "Damaged" else 30
        risk_score = int(base_score + (rainfall * 0.3) + (traffic * 0.2))
        risk_score = min(risk_score, 100)
        
        if risk_score > 80:
            priority = "High"
        elif risk_score > 50:
            priority = "Moderate"
        else:
            priority = "Low"
            
        failure_probability = min(100, risk_score + random.randint(0, 10))
        
        if risk_score > 80:
            estimated_failure_days = 7
        elif risk_score > 60:
            estimated_failure_days = 15
        else:
            estimated_failure_days = 30
            
        if prediction == "Damaged":
            explanation = "Structural damage detected under current environmental stress."
        else:
            explanation = "No major structural issues detected, but environmental factors contribute to moderate risk."
            
        if risk_score > 80:
            recommended_action = "Immediate inspection required"
            impact_score = "High (dense impact area)"
        elif risk_score > 50:
            recommended_action = "Schedule maintenance"
            impact_score = "Moderate"
        else:
            recommended_action = "Monitor condition"
            impact_score = "Low"
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_score": risk_score,
            "priority": priority,
            "failure_probability": failure_probability,
            "estimated_failure_days": estimated_failure_days,
            "explanation": explanation,
            "recommended_action": recommended_action,
            "impact_score": impact_score
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

class ReportSave(BaseModel):
    image_name: str
    prediction: str
    risk_score: int
    latitude: float
    longitude: float
    failure_probability: int
    estimated_failure_days: int
    explanation: str
    recommended_action: str = "Monitor condition"
    impact_score: str = "Low"

@app.post("/api/report/save")
async def save_report(report: ReportSave):
    try:
        with open(REPORTS_FILE, "r") as f:
            reports = json.load(f)
            
        new_report = {
            "id": str(uuid.uuid4()),
            "image_name": report.image_name,
            "prediction": report.prediction,
            "risk_score": report.risk_score,
            "latitude": report.latitude,
            "longitude": report.longitude,
            "failure_probability": report.failure_probability,
            "estimated_failure_days": report.estimated_failure_days,
            "explanation": report.explanation,
            "recommended_action": report.recommended_action,
            "impact_score": report.impact_score,
            "timestamp": datetime.now().isoformat(),
            "status": "Pending",
            "action_log": [{"action": "Created", "timestamp": datetime.now().isoformat()}]
        }
        
        reports.append(new_report)
        
        with open(REPORTS_FILE, "w") as f:
            json.dump(reports, f, indent=4)
            
        return {"status": "success", "report": new_report}
    except Exception as e:
        return {"error": f"Failed to save report: {str(e)}"}

@app.get("/api/reports")
async def get_reports():
    try:
        with open(REPORTS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to fetch reports: {str(e)}"}

class UpdateStatus(BaseModel):
    status: str

@app.put("/api/report/update/{id}")
async def update_report(id: str, update: UpdateStatus):
    try:
        with open(REPORTS_FILE, "r") as f:
            reports = json.load(f)
            
        for report in reports:
            if report["id"] == id:
                report["status"] = update.status
                if "action_log" not in report:
                    report["action_log"] = [{"action": "Created", "timestamp": report.get("timestamp", datetime.now().isoformat())}]
                report["action_log"].append({"action": f"Marked as {update.status}", "timestamp": datetime.now().isoformat()})
                with open(REPORTS_FILE, "w") as f:
                    json.dump(reports, f, indent=4)
                return {"status": "success", "report": report}
                
        return {"error": "Report not found"}
    except Exception as e:
        return {"error": f"Failed to update report: {str(e)}"}

@app.put("/api/report/log_view/{id}")
async def log_view(id: str):
    try:
        with open(REPORTS_FILE, "r") as f:
            reports = json.load(f)
            
        for report in reports:
            if report["id"] == id:
                if "action_log" not in report:
                    report["action_log"] = [{"action": "Created", "timestamp": report.get("timestamp", datetime.now().isoformat())}]
                # Prevent spamming Viewed
                if not report["action_log"] or report["action_log"][-1]["action"] != "Viewed":
                    report["action_log"].append({"action": "Viewed", "timestamp": datetime.now().isoformat()})
                    with open(REPORTS_FILE, "w") as f:
                        json.dump(reports, f, indent=4)
                return {"status": "success"}
                
        return {"error": "Report not found"}
    except Exception as e:
        return {"error": f"Failed to log view: {str(e)}"}

@app.get("/api/notifications")
async def get_notifications():
    try:
        with open(REPORTS_FILE, "r") as f:
            reports = json.load(f)
            
        notifications = [r for r in reports if r["risk_score"] > 80]
        return notifications
    except Exception as e:
        return {"error": f"Failed to fetch notifications: {str(e)}"}

# uvicorn server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
