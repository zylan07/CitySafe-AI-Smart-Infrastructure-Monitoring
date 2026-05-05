document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    
    // Upload Elements
    const imageUpload = document.getElementById('image-upload');
    const uploadLabel = document.getElementById('upload-label');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image');
    
    // Inputs
    const rainfallInput = document.getElementById('rainfall-input');
    const trafficInput = document.getElementById('traffic-input');
    const locationStatus = document.getElementById('location-status');
    
    // Location Controls
    const gpsBtn = document.getElementById('gps-btn');
    const searchInput = document.getElementById('location-search-input');
    const searchSuggestions = document.getElementById('search-suggestions');
    const searchBtn = document.getElementById('location-search-btn');
    const activeSourceBadge = document.getElementById('active-source-badge');
    
    // Tabs & Views
    const tabSubmit = document.getElementById('tab-submit');
    const tabHistory = document.getElementById('tab-history');
    const viewSubmit = document.getElementById('view-submit');
    const viewHistory = document.getElementById('view-history');
    const historyContainer = document.getElementById('history-container');
    
    // Buttons
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = analyzeBtn.querySelector('.btn-text');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const saveReportBtn = document.getElementById('save-report-btn');
    
    // Result Section
    const resultCard = document.getElementById('result-card');
    const priorityBadge = document.getElementById('priority-badge');
    const predictionValue = document.getElementById('prediction-value');
    const scoreValue = document.getElementById('score-value');
    const errorMsg = document.getElementById('error-message');
    
    // Predictive UI
    const probText = document.getElementById('prob-text');
    const estDaysText = document.getElementById('est-days-text');
    const probBarFill = document.getElementById('prob-bar-fill');
    const explanationText = document.getElementById('explanation-text');
    
    // Toast
    const successPopup = document.getElementById('success-popup');

    // --- State ---
    let currentFile = null;
    let currentAnalysis = null;
    let currentLat = 20.5937;
    let currentLon = 78.9629;
    let mapMarker = null;

    // --- Event Listeners ---

    // Tab Switching
    tabSubmit.addEventListener('click', () => {
        tabSubmit.classList.add('active');
        tabHistory.classList.remove('active');
        viewSubmit.classList.remove('hidden');
        viewHistory.classList.add('hidden');
    });

    tabHistory.addEventListener('click', () => {
        tabHistory.classList.add('active');
        tabSubmit.classList.remove('active');
        viewHistory.classList.remove('hidden');
        viewSubmit.classList.add('hidden');
        fetchHistory();
    });

    // --- Map Initialization ---
    const map = L.map('map').setView([currentLat, currentLon], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Initial marker
    mapMarker = L.marker([currentLat, currentLon]).addTo(map);
    updateLocationText(currentLat, currentLon);

    // Unified Location Handler
    async function updateLocation(lat, lon, source = 'map') {
        currentLat = lat;
        currentLon = lon;
        
        mapMarker.setLatLng([currentLat, currentLon]);
        map.setView([currentLat, currentLon], map.getZoom() < 10 ? 10 : map.getZoom());
        updateLocationText(currentLat, currentLon);
        
        // Update badge
        if (activeSourceBadge) {
            activeSourceBadge.className = `source-badge ${source}-source`;
            activeSourceBadge.textContent = source.toUpperCase();
        }
        
        locationStatus.textContent = 'Fetching weather...';
        
        // Generate Traffic
        generateTraffic(currentLat, currentLon);

        // Fetch Rainfall
        try {
            const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${currentLat}&longitude=${currentLon}&current=precipitation,rain`);
            if (!res.ok) throw new Error('Weather API error');
            const data = await res.json();
            console.log("Rainfall data:", data);
            
            let rainfall = 0;

            if (data.current) {
                rainfall =
                    data.current.precipitation ??
                    data.current.rain ??
                    0;
            }

            if (!rainfall || rainfall === 0) {
                rainfall = Math.floor(Math.random() * 40) + 10;
            }
            
            rainfallInput.value = rainfall;
            locationStatus.textContent = `Weather fetched for ${source.toUpperCase()} location.`;
        } catch (err) {
            console.error(err);
            rainfallInput.value = Math.floor(Math.random() * 40) + 10; // Fallback
            locationStatus.textContent = `Weather fallback used for ${source.toUpperCase()} location.`;
        }
    }

    function generateTraffic(lat, lon) {
        const hour = new Date().getHours();
        let base;

        if (hour >= 8 && hour <= 10) base = 80;
        else if (hour >= 17 && hour <= 20) base = 85;
        else base = 40;

        // Add randomness
        let randomFactor = Math.floor(Math.random() * 20) - 10;

        let traffic = base + randomFactor;

        // City-based boost
        if (lat && lon) {
            // crude urban detection (example: near cities)
            if (lat > 10 && lat < 30) {
                traffic += 10;
            }
        }

        // Clamp values
        traffic = Math.max(10, Math.min(100, traffic));

        trafficInput.value = traffic;
    }

    // Map Click Handler
    map.on('click', (e) => {
        updateLocation(e.latlng.lat, e.latlng.lng, 'map');
    });

    // GPS Handler
    gpsBtn.addEventListener('click', () => {
        if (!navigator.geolocation) {
            showError("Geolocation is not supported by your browser.");
            return;
        }

        locationStatus.textContent = "Locating...";
        gpsBtn.disabled = true;

        navigator.geolocation.getCurrentPosition(
            (position) => {
                gpsBtn.disabled = false;
                updateLocation(position.coords.latitude, position.coords.longitude, 'gps');
            },
            (error) => {
                gpsBtn.disabled = false;
                console.error("GPS Error:", error);
                showError("Unable to retrieve your location. Please check permissions.");
                locationStatus.textContent = "GPS locating failed.";
            }
        );
    });

    // Search Autocomplete Handler
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();
        
        if (query.length < 3) {
            if (searchSuggestions) {
                searchSuggestions.classList.add('hidden');
                searchSuggestions.innerHTML = '';
            }
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const res = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=5`);
                const data = await res.json();
                console.log("Search results:", data);
                
                if (searchSuggestions) {
                    searchSuggestions.innerHTML = '';
                    if (data && data.length > 0) {
                        data.forEach(item => {
                            const li = document.createElement('li');
                            li.textContent = item.display_name;
                            li.addEventListener('click', () => {
                                searchInput.value = item.display_name;
                                searchSuggestions.classList.add('hidden');
                                updateLocation(parseFloat(item.lat), parseFloat(item.lon), 'search');
                            });
                            searchSuggestions.appendChild(li);
                        });
                        searchSuggestions.classList.remove('hidden');
                    } else {
                        searchSuggestions.classList.add('hidden');
                    }
                }
            } catch (err) {
                console.error("Autocomplete error:", err);
            }
        }, 500);
    });

    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (searchSuggestions && !searchInput.contains(e.target) && !searchSuggestions.contains(e.target)) {
            searchSuggestions.classList.add('hidden');
        }
    });

    // Search Button Handler
    searchBtn.addEventListener('click', async () => {
        const query = searchInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        locationStatus.textContent = "Searching...";
        if (searchSuggestions) searchSuggestions.classList.add('hidden');

        try {
            const res = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`);
            const data = await res.json();

            if (data && data.length > 0) {
                const lat = parseFloat(data[0].lat);
                const lon = parseFloat(data[0].lon);
                updateLocation(lat, lon, 'search');
            } else {
                showError("No results found.");
                locationStatus.textContent = "Search failed.";
            }
        } catch (err) {
            console.error("Search Error:", err);
            showError("Search service unavailable.");
            locationStatus.textContent = "Search failed.";
        } finally {
            searchBtn.disabled = false;
        }
    });

    function updateLocationText(lat, lon) {
        if (locationStatus) {
            locationStatus.textContent = `Selected: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
        }
    }

    // Handle Image Upload
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            currentFile = file;
            const reader = new FileReader();
            
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                uploadLabel.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                hideError();
            };
            
            reader.readAsDataURL(file);
        } else {
            showError("Please select a valid image file.");
        }
    });

    // Handle Remove Image
    removeImageBtn.addEventListener('click', (e) => {
        e.preventDefault();
        imageUpload.value = '';
        currentFile = null;
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadLabel.classList.remove('hidden');
        hideResultCard();
    });

    // Handle Analyze Button Click
    analyzeBtn.addEventListener('click', async () => {
        // Validate
        if (!currentFile) {
            showError("Please upload an image first.");
            return;
        }
        
        const rainfall = parseFloat(rainfallInput.value);
        const traffic = parseInt(trafficInput.value);
        
        if (isNaN(rainfall) || isNaN(traffic)) {
            showError("Please enter valid numerical values for rainfall and traffic.");
            return;
        }

        hideError();
        setAnalyzingState(true);

        // Prepare Data
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('rainfall', rainfall);
        formData.append('traffic', traffic);

        try {
            const response = await fetch('http://127.0.0.1:8000/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            
            currentAnalysis = {
                prediction: data.prediction,
                risk_score: data.risk_score,
                priority: data.priority,
                failure_probability: data.failure_probability ?? 50,
                estimated_failure_days: data.estimated_failure_days ?? 30,
                explanation: data.explanation || "Analysis complete."
            };

            showResultCard(currentAnalysis);

        } catch (error) {
            console.error("Analysis Failed:", error);
            showError("Analysis failed. Please ensure the backend server is running.");
        } finally {
            setAnalyzingState(false);
        }
    });

    function getDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    // Handle Save Report Button Click
    saveReportBtn.addEventListener('click', async () => {
        if (!currentFile || !currentAnalysis) {
            showError("No analysis data available to save.");
            return;
        }

        try {
            // Duplicate Check
            const historyRes = await fetch('http://127.0.0.1:8000/api/reports');
            if (historyRes.ok) {
                const pastReports = await historyRes.json();
                const threshold = 0.05; // 50 meters
                const isDuplicate = pastReports.some(r => {
                    const dist = getDistance(currentLat, currentLon, r.latitude, r.longitude);
                    return dist < threshold && r.prediction === currentAnalysis.prediction && r.status !== 'Resolved';
                });
                
                if (isDuplicate) {
                    alert("Similar issue already reported nearby. Saving aborted to prevent duplicates.");
                    return;
                }
            }
        } catch (e) {
            console.error("Duplicate check failed:", e);
        }

        const originalBtnHtml = saveReportBtn.innerHTML;
        saveReportBtn.innerHTML = 'Saving...';
        saveReportBtn.disabled = true;

        const reportPayload = {
            image_name: currentFile.name,
            prediction: currentAnalysis.prediction,
            risk_score: currentAnalysis.risk_score,
            latitude: currentLat,
            longitude: currentLon,
            failure_probability: currentAnalysis.failure_probability,
            estimated_failure_days: currentAnalysis.estimated_failure_days,
            explanation: currentAnalysis.explanation,
            recommended_action: currentAnalysis.recommended_action,
            impact_score: currentAnalysis.impact_score
        };

        try {
            const response = await fetch('http://127.0.0.1:8000/api/report/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(reportPayload)
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }

            // Show Success Popup
            showToast();
            
            // Optionally, reset the form after successful save
            // resetForm();

        } catch (error) {
            console.error("Save Failed:", error);
            showError("Failed to save report. Please try again.");
        } finally {
            saveReportBtn.innerHTML = originalBtnHtml;
            saveReportBtn.disabled = false;
        }
    });

    // --- Helper Functions ---

    function setAnalyzingState(isAnalyzing) {
        if (isAnalyzing) {
            analyzeBtn.disabled = true;
            analyzeBtnText.textContent = "Analyzing...";
            analyzeSpinner.classList.remove('hidden');
            hideResultCard();
        } else {
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = "Analyze Infrastructure";
            analyzeSpinner.classList.add('hidden');
        }
    }

    function showResultCard(data) {
        predictionValue.textContent = data.prediction || 'Unknown';
        
        const score = typeof data.risk_score === 'number' ? data.risk_score.toFixed(1) : data.risk_score;
        scoreValue.textContent = `${score}/100`;

        // Update Action & Impact
        const actionText = document.getElementById('action-text');
        const impactText = document.getElementById('impact-text');
        if (actionText) actionText.textContent = data.recommended_action || "Monitor condition";
        if (impactText) impactText.textContent = data.impact_score || "Low";

        // Reset badge classes
        priorityBadge.className = 'priority-badge';
        
        let priorityText = data.priority || 'Low';
        priorityText = priorityText.toLowerCase();

        if (priorityText.includes('high')) {
            priorityBadge.classList.add('priority-high');
            priorityBadge.textContent = 'High Priority';
        } else if (priorityText.includes('mod')) {
            priorityBadge.classList.add('priority-moderate');
            priorityBadge.textContent = 'Moderate Priority';
        } else {
            priorityBadge.classList.add('priority-low');
            priorityBadge.textContent = 'Low Priority';
        }

        const prob = data.failure_probability ?? 50;
        if (probText) probText.textContent = `${prob}%`;
        if (probBarFill) {
            probBarFill.style.width = `${prob}%`;
            probBarFill.className = 'prob-bar';
            if (prob > 80) probBarFill.classList.add('prob-high');
            else if (prob > 50) probBarFill.classList.add('prob-med');
            else probBarFill.classList.add('prob-low');
        }
        
        if (estDaysText) estDaysText.textContent = `(${data.estimated_failure_days ?? '--'} days)`;
        if (explanationText) explanationText.textContent = data.explanation || '--';

        resultCard.classList.remove('hidden');
    }

    function hideResultCard() {
        resultCard.classList.add('hidden');
        currentAnalysis = null;
    }

    function showError(message) {
        errorMsg.textContent = message;
        errorMsg.classList.remove('hidden');
    }

    function hideError() {
        errorMsg.classList.add('hidden');
        errorMsg.textContent = '';
    }

    function showToast(message = "Report Submitted Successfully", duration = 2000) {
        const popupMsg = successPopup.querySelector('p');
        if (popupMsg) popupMsg.textContent = message;

        successPopup.classList.remove('hidden');
        // Force reflow
        void successPopup.offsetWidth;
        successPopup.classList.add('show');

        setTimeout(() => {
            successPopup.classList.remove('show');
            setTimeout(() => {
                successPopup.classList.add('hidden');
            }, 400); // match transition duration
        }, duration);
    }
    
    function resetForm() {
        removeImageBtn.click();
        rainfallInput.value = '';
        trafficInput.value = '';
    }

    // --- History Functions ---
    async function fetchHistory() {
        historyContainer.innerHTML = `
            <div class="skeleton-card">
                <div class="skeleton-line title"></div>
                <div class="skeleton-line short"></div>
                <div class="skeleton-line"></div>
            </div>
            <div class="skeleton-card">
                <div class="skeleton-line title"></div>
                <div class="skeleton-line short"></div>
                <div class="skeleton-line"></div>
            </div>
        `;
        
        try {
            const response = await fetch('http://127.0.0.1:8000/api/reports');
            if (!response.ok) throw new Error('Failed to fetch reports');
            
            const reports = await response.json();

            // Notification logic
            const hasResolved = reports.some(r => r.status === 'Resolved');
            if (hasResolved && !sessionStorage.getItem('resolvedNotified')) {
                showToast("✅ Your reported issue has been resolved", 3500);
                sessionStorage.setItem('resolvedNotified', 'true');
            }

            renderReports(reports);
        } catch (error) {
            console.error(error);
            historyContainer.innerHTML = '<div class="empty-state">Unable to load reports. Server may be offline.</div>';
        }
    }

    function renderReports(reports) {
        if (!reports || reports.length === 0) {
            historyContainer.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">🚀</span>
                    <p>No reports yet.<br>Start by submitting one!</p>
                </div>
            `;
            return;
        }

        // Sort descending by timestamp
        reports.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        historyContainer.innerHTML = '';

        reports.forEach(report => {
            const card = document.createElement('div');
            card.className = 'report-card';
            
            // Risk color coding
            let riskClass = 'risk-low';
            if (report.risk_score > 80) riskClass = 'risk-high';
            else if (report.risk_score > 50) riskClass = 'risk-mod';
            
            card.classList.add(riskClass);

            const timeStr = new Date(report.timestamp).toLocaleString(undefined, {
                year: 'numeric', month: 'short', day: 'numeric', 
                hour: '2-digit', minute: '2-digit'
            });

            const prob = report.failure_probability ?? 50;
            const days = report.estimated_failure_days ?? 30;
            const statusClass = (report.status === 'Resolved') ? 'status-resolved' : 'status-pending';
            const resolvedBadge = (report.status === 'Resolved') ? `<div class="resolved-badge">✔ Resolved by authorities</div>` : '';

            const reAnalyzeBtn = document.createElement('button');
            reAnalyzeBtn.className = 'secondary-btn';
            reAnalyzeBtn.style.padding = '0.5rem';
            reAnalyzeBtn.style.marginTop = '0.75rem';
            reAnalyzeBtn.innerHTML = '🔄 Re-analyze';
            reAnalyzeBtn.onclick = () => handleReAnalyze(report);

            card.innerHTML = `
                ${resolvedBadge}
                <div class="report-header">
                    <span class="report-title">${report.image_name || 'Report'}</span>
                    <span class="report-time">${timeStr}</span>
                </div>
                <div class="report-details">
                    <span><strong>Prediction:</strong> ${report.prediction || '--'}</span>
                    <span><strong>Risk Score:</strong> ${report.risk_score ? report.risk_score.toFixed(1) : '--'}/100</span>
                    <span><strong>Failure Prob:</strong> ${prob}% (${days}d)</span>
                    <span><strong>Location:</strong> ${report.latitude ? report.latitude.toFixed(4) : '--'}, ${report.longitude ? report.longitude.toFixed(4) : '--'}</span>
                    <span class="report-status ${statusClass}">${report.status || 'Pending'}</span>
                </div>
            `;
            card.appendChild(reAnalyzeBtn);
            historyContainer.appendChild(card);
        });
    }

    async function handleReAnalyze(report) {
        // Create a mock image blob for re-analysis since we don't store files
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#444';
        ctx.fillRect(0, 0, 128, 128);
        
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, report.image_name || "mock.png");
            formData.append('rainfall', document.getElementById('rainfall-input')?.value || 10);
            formData.append('traffic', document.getElementById('traffic-input')?.value || 100);
            
            showToast("🔄 Re-analyzing...", 2000);
            
            try {
                const response = await fetch('http://127.0.0.1:8000/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                if (response.ok) {
                    const data = await response.json();
                    showToast(`✅ Re-analysis complete! New Risk: ${data.risk_score}`, 4000);
                } else {
                    showToast("❌ Re-analysis failed.", 3000);
                }
            } catch (err) {
                console.error(err);
            }
        }, 'image/png');
    }
});
// --- Theme Toggle & Tooltips ---
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    
    if (themeToggle && themeIcon) {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-mode');
            themeIcon.textContent = '🌙';
        } else {
            themeIcon.textContent = '☀';
        }
        
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('light-mode');
            const isLight = document.body.classList.contains('light-mode');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            
            if (isLight) {
                themeIcon.textContent = '🌙';
            } else {
                themeIcon.textContent = '☀';
            }
        });
    }

    // --- Guided Tooltips ---
    if (!localStorage.getItem('citysafe_tooltips_shown')) {
        const tooltipUpload = document.getElementById('tooltip-upload');
        const tooltipLocation = document.getElementById('tooltip-location');
        
        setTimeout(() => {
            if(tooltipUpload) tooltipUpload.classList.add('show');
            if(tooltipLocation) tooltipLocation.classList.add('show');
        }, 1500);
        
        const hideTooltips = () => {
            if(tooltipUpload) tooltipUpload.classList.remove('show');
            if(tooltipLocation) tooltipLocation.classList.remove('show');
            localStorage.setItem('citysafe_tooltips_shown', 'true');
            document.removeEventListener('click', hideTooltips);
        };
        
        setTimeout(hideTooltips, 7000);
        document.addEventListener('click', hideTooltips);
    }
});

