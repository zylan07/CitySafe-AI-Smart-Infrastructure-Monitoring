document.addEventListener('DOMContentLoaded', () => {
    
    // --- Elements ---
    const totalReportsEl = document.getElementById('total-reports');
    const highRiskEl = document.getElementById('high-risk-count');
    const modRiskEl = document.getElementById('mod-risk-count');
    const lowRiskEl = document.getElementById('low-risk-count');
    
    const queueContainer = document.getElementById('queue-container');
    const notifContainer = document.getElementById('notifications-container');
    
    const lastUpdatedEl = document.getElementById('last-updated');
    const actionToast = document.getElementById('action-toast');
    const toastMsg = document.getElementById('toast-msg');

    // Filters
    const filterSearch = document.getElementById('filter-search');
    const filterRisk = document.getElementById('filter-risk');
    const filterStatus = document.getElementById('filter-status');

    // --- State ---
    let map;
    let markers = {};
    let reportsData = [];
    let geocodeCache = {};
    let trendChartInstance = null;

    // --- Map Init ---
    function initMap() {
        map = L.map('admin-map').setView([20.5937, 78.9629], 5);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
    }

    // --- Geocode Helper ---
    async function getLocationName(lat, lon) {
        const key = `${lat},${lon}`;
        if (geocodeCache[key]) return geocodeCache[key];

        try {
            const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`);
            const data = await res.json();
            if (data && data.address) {
                const name = data.address.city || data.address.town || data.address.village || data.address.county || data.address.state || "Unknown Location";
                geocodeCache[key] = name;
                return name;
            }
        } catch (e) {
            console.error("Geocode error", e);
        }
        return `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
    }

    // --- Filter Logic ---
    function getFilteredReports() {
        const query = filterSearch ? filterSearch.value.toLowerCase() : '';
        const risk = filterRisk ? filterRisk.value : 'all';
        const status = filterStatus ? filterStatus.value : 'all';

        return reportsData.filter(r => {
            // Search
            let matchSearch = true;
            if (query) {
                const name = r.image_name ? r.image_name.toLowerCase() : '';
                const locStr = `${r.latitude.toFixed(4)}, ${r.longitude.toFixed(4)}`;
                matchSearch = name.includes(query) || locStr.includes(query);
            }
            
            // Risk
            let matchRisk = true;
            if (risk === 'high') matchRisk = r.risk_score > 80;
            else if (risk === 'mod') matchRisk = r.risk_score > 50 && r.risk_score <= 80;
            else if (risk === 'low') matchRisk = r.risk_score <= 50;

            // Status
            let matchStatus = true;
            if (status === 'pending') matchStatus = r.status !== 'Resolved';
            else if (status === 'resolved') matchStatus = r.status === 'Resolved';

            return matchSearch && matchRisk && matchStatus;
        });
    }

            [filterSearch, filterRisk, filterStatus].forEach(el => {
        if(el) el.addEventListener('input', () => {
            const filtered = getFilteredReports();
            renderMap(filtered);
            renderQueue(filtered);
            renderTrendChart(filtered);
        });
    });

    // --- Core Data Fetching ---
    async function fetchData() {
        try {
            const [reportsRes, notifRes] = await Promise.all([
                fetch('http://127.0.0.1:8000/api/reports'),
                fetch('http://127.0.0.1:8000/api/notifications')
            ]);
            
            if (!reportsRes.ok || !notifRes.ok) throw new Error("API Error");
            
            const reports = await reportsRes.json();
            const notifications = await notifRes.json();
            
            reportsData = reports;
            
            updateLastUpdated();
            updateSummary(reports);
            
            const filtered = getFilteredReports();
            renderMap(filtered);
            renderQueue(filtered);
            renderTrendChart(filtered);
            
            renderNotifications(notifications);
            
        } catch (err) {
            console.error("Failed to fetch data:", err);
            // Optionally show connection error
        }
    }

    function updateLastUpdated() {
        const now = new Date();
        lastUpdatedEl.textContent = `Last updated: ${now.toLocaleTimeString()}`;
    }

    // --- Render Summary ---
    function updateSummary(reports) {
        let total = reports.length;
        let high = 0, mod = 0, low = 0;
        
        reports.forEach(r => {
            if (r.risk_score > 80) high++;
            else if (r.risk_score > 50) mod++;
            else low++;
        });

        totalReportsEl.textContent = total;
        highRiskEl.textContent = high;
        modRiskEl.textContent = mod;
        lowRiskEl.textContent = low;
    }

    // --- Render Map ---
    function renderMap(reports) {
        // Clear existing markers
        Object.values(markers).forEach(m => map.removeLayer(m));
        markers = {};

        reports.forEach(report => {
            let color = '#10b981'; // low
            if (report.risk_score > 80) color = '#ef4444'; // high
            else if (report.risk_score > 50) color = '#f59e0b'; // mod

            // Create custom SVG marker
            const markerHtml = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="${color}" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                    <circle cx="12" cy="10" r="3" fill="white"></circle>
                </svg>
            `;
            
            const customIcon = L.divIcon({
                html: markerHtml,
                className: '',
                iconSize: [24, 24],
                iconAnchor: [12, 24],
                popupAnchor: [0, -24]
            });

            const marker = L.marker([report.latitude, report.longitude], {icon: customIcon}).addTo(map);
            
            marker.on('click', () => {
                focusQueueItem(report.id);
            });
            
            const popupContent = `
                <div class="popup-content">
                    <strong>${report.image_name || 'Report'}</strong>
                    <span>Risk Score: ${report.risk_score}</span>
                    <span>Status: ${report.status}</span>
                    <button class="popup-btn" onclick="focusQueueItem('${report.id}')">Focus in queue</button>
                </div>
            `;
            
            marker.bindPopup(popupContent);
            markers[report.id] = marker;
        });
    }

    // --- Render Priority Queue ---
    function renderQueue(reports) {
        if (!reports || reports.length === 0) {
            queueContainer.innerHTML = '<div class="empty-state">No reports available</div>';
            return;
        }

        // Sort: highest risk first, but push resolved to bottom
        let sorted = [...reports].sort((a, b) => {
            if (a.status === 'Resolved' && b.status !== 'Resolved') return 1;
            if (a.status !== 'Resolved' && b.status === 'Resolved') return -1;
            return b.risk_score - a.risk_score;
        });

        queueContainer.innerHTML = '';

        sorted.forEach(report => {
            const card = document.createElement('div');
            card.className = 'queue-card';
            card.id = `queue-item-${report.id}`;
            
            let riskClass = 'risk-low';
            let isUrgent = false;
            
            if (report.risk_score > 80) {
                riskClass = 'risk-high';
                if (report.status !== 'Resolved') isUrgent = true;
            } else if (report.risk_score > 50) {
                riskClass = 'risk-mod';
            }
            
            card.classList.add(riskClass);
            if (report.status === 'Resolved') {
                card.classList.add('status-resolved');
            }
            if (isUrgent) {
                card.classList.add('urgent-card');
            }

            const timeStr = new Date(report.timestamp).toLocaleString(undefined, {
                year: 'numeric', month: 'short', day: 'numeric', 
                hour: '2-digit', minute: '2-digit'
            });

            let urgentBadge = isUrgent ? `<span class="urgent-badge">⚠️ Immediate Action Required</span>` : '';

            let actionBtn = report.status === 'Resolved' 
                ? `<span class="queue-status status-resolved">✔ Resolved</span>` 
                : `<button class="resolve-btn" onclick="markResolved('${report.id}')">Mark as Resolved</button>`;

            const prob = report.failure_probability ?? 50;
            const days = report.estimated_failure_days ?? 30;
            const priorityText = report.risk_score > 80 ? 'High' : report.risk_score > 50 ? 'Moderate' : 'Low';
            const actionText = report.recommended_action || "Monitor condition";
            const impactText = report.impact_score || "Low";

            let actionLogHtml = '<div style="margin-top: 10px; font-size: 0.85em; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;"><strong>Action Log:</strong><ul style="padding-left: 15px; margin-top: 5px; opacity: 0.8;">';
            const logs = report.action_log || [{"action": "Created", "timestamp": report.timestamp}];
            logs.forEach(log => {
                const logTime = new Date(log.timestamp).toLocaleTimeString();
                actionLogHtml += `<li>[${logTime}] ${log.action}</li>`;
            });
            actionLogHtml += '</ul></div>';

            card.innerHTML = `
                ${urgentBadge}
                <div class="queue-header">
                    <span class="queue-title">${report.image_name || 'Report'}</span>
                    <span class="queue-time">${timeStr}</span>
                </div>
                <div class="queue-details">
                    <span><strong>Risk:</strong> ${report.risk_score} (${priorityText})</span>
                    <span><strong>Failure:</strong> ${prob}% (${days} days)</span>
                    <span><strong>Action:</strong> ${actionText}</span>
                    <span><strong>Impact:</strong> ${impactText}</span>
                    <span><strong>Location:</strong> ${report.latitude.toFixed(4)}, ${report.longitude.toFixed(4)}</span>
                    <span class="queue-status status-${report.status.toLowerCase()}">${report.status}</span>
                </div>
                <div class="queue-actions">
                    ${actionBtn}
                </div>
                ${actionLogHtml}
            `;
            
            queueContainer.appendChild(card);
        });
    }

    // --- Render Notifications ---
    async function renderNotifications(notifs) {
        if (!notifs || notifs.length === 0) {
            notifContainer.innerHTML = '<div class="empty-state">No alerts</div>';
            return;
        }
        
        // Sort newest first
        notifs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        notifContainer.innerHTML = '';
        
        for (const notif of notifs) {
            // Resolve location name
            const locName = await getLocationName(notif.latitude, notif.longitude);
            
            const timeStr = new Date(notif.timestamp).toLocaleTimeString(undefined, {
                hour: '2-digit', minute: '2-digit'
            });

            const el = document.createElement('div');
            el.className = 'notification-item';
            el.innerHTML = `
                <span class="notif-title">⚠️ High Risk Detected (${notif.risk_score})</span>
                <div class="notif-details">
                    <span>📍 ${locName}</span>
                    <span>🕒 ${timeStr}</span>
                </div>
            `;
            notifContainer.appendChild(el);
        }
    }

    // --- Render Trend Chart ---
    function renderTrendChart(reports) {
        const ctx = document.getElementById('trendChart');
        if (!ctx) return;

        if (trendChartInstance) {
            trendChartInstance.destroy();
        }

        // Group by YYYY-MM-DD
        const countsByDate = {};
        reports.forEach(r => {
            const d = new Date(r.timestamp);
            const key = d.toISOString().split('T')[0];
            countsByDate[key] = (countsByDate[key] || 0) + 1;
        });

        const sortedDates = Object.keys(countsByDate).sort();
        const todayStr = new Date().toISOString().split('T')[0];
        
        const labels = sortedDates.map(d => {
            return new Date(d).toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
        });
        
        const data = sortedDates.map(d => countsByDate[d]);
        
        const pointColors = sortedDates.map(d => d === todayStr ? '#ef4444' : '#3b82f6');
        const pointRadii = sortedDates.map(d => d === todayStr ? 6 : 4);
        const pointHoverRadii = sortedDates.map(d => d === todayStr ? 8 : 6);

        // Gradient Fill
        const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 250);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        trendChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Reports',
                    data: data,
                    borderColor: '#3b82f6',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: pointRadii,
                    pointHoverRadius: pointHoverRadii,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw} reports on this day`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false, drawBorder: false },
                        ticks: { color: '#9ca3af' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                        ticks: { color: '#9ca3af', stepSize: 1, beginAtZero: true }
                    }
                }
            }
        });
    }

    // --- Actions ---
    window.downloadPDF = function() {
        if (!window.jspdf) return;
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        const margin = 15;
        let yPos = margin + 10;
        
        // Official Title
        doc.setFontSize(22);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(30, 58, 138); 
        doc.text("CitySafe", margin, yPos);
        
        doc.setFontSize(16);
        doc.setTextColor(50, 50, 50);
        doc.text("Infrastructure Report", margin + 35, yPos);
        
        yPos += 10;
        doc.setFontSize(10);
        doc.setTextColor(150, 150, 150);
        doc.setFont("helvetica", "normal");
        doc.text(`Generated on: ${new Date().toLocaleString()}`, margin, yPos);
        
        yPos += 20;
        
        // Stats
        let high = 0, mod = 0, low = 0;
        reportsData.forEach(r => {
            if (r.risk_score > 80) high++;
            else if (r.risk_score > 50) mod++;
            else low++;
        });
        
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(0, 0, 0);
        doc.text("Executive Summary", margin, yPos);
        yPos += 8;
        
        doc.setFontSize(11);
        doc.setFont("helvetica", "normal");
        doc.text(`Total Reports: ${reportsData.length}`, margin, yPos);
        yPos += 6;
        doc.text(`High Risk: ${high} | Moderate: ${mod} | Low: ${low}`, margin, yPos);
        yPos += 20;
        
        // Critical Alerts Summary
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(220, 38, 38); 
        doc.text("Critical Alerts Summary", margin, yPos);
        yPos += 8;
        
        if (high > 0) {
            doc.setFontSize(11);
            doc.setFont("helvetica", "bold");
            doc.text(`${high} CRITICAL INFRASTRUCTURE RISKS DETECTED`, margin, yPos);
            yPos += 6;
            doc.setFont("helvetica", "normal");
            doc.setTextColor(100, 100, 100);
            doc.text("Immediate attention required for the following sites.", margin, yPos);
            yPos += 15;
        } else {
            doc.setFontSize(11);
            doc.setFont("helvetica", "normal");
            doc.setTextColor(16, 185, 129); 
            doc.text("NO CRITICAL RISKS DETECTED.", margin, yPos);
            yPos += 15;
        }
        
        // Top 5 Table
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(0, 0, 0);
        doc.text("Top 5 Highest Risk Reports", margin, yPos);
        yPos += 10;
        
        const top5 = [...reportsData]
            .sort((a, b) => b.risk_score - a.risk_score)
            .slice(0, 5);
            
        doc.setFontSize(11);
        
        top5.forEach((r, idx) => {
            const riskText = r.risk_score > 80 ? 'High' : (r.risk_score > 50 ? 'Mod' : 'Low');
            
            doc.setFont("helvetica", "bold");
            doc.text(`${idx + 1}. ${r.image_name}`, margin, yPos);
            yPos += 6;
            
            doc.setFont("helvetica", "normal");
            doc.text(`Risk: ${r.risk_score} (${riskText})`, margin + 5, yPos);
            yPos += 6;
            
            doc.text(`Status: ${r.status}`, margin + 5, yPos);
            yPos += 6;
            
            doc.text(`Location: ${r.latitude.toFixed(4)}, ${r.longitude.toFixed(4)}`, margin + 5, yPos);
            yPos += 10;
        });
        
        // Footer
        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.setTextColor(150, 150, 150);
        doc.text("Generated by CitySafe AI System", margin, 285);
        
        doc.save("CitySafe_Report.pdf");
    };

    window.markResolved = async function(id) {
        try {
            const res = await fetch(`http://127.0.0.1:8000/api/report/update/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: "Resolved" })
            });
            
            if (res.ok) {
                showToast("✔ Issue marked as resolved");
                // Immediately re-fetch and re-render to update queue sorting
                fetchData();
            }
        } catch (err) {
            console.error("Failed to mark resolved", err);
        }
    };

    window.focusQueueItem = function(id) {
        const item = document.getElementById(`queue-item-${id}`);
        if (item) {
            item.scrollIntoView({ behavior: 'smooth', block: 'center' });
            item.classList.add('highlighted-card');
            setTimeout(() => {
                item.classList.remove('highlighted-card');
            }, 1500);
            
            // Log View Event
            fetch(`http://127.0.0.1:8000/api/report/log_view/${id}`, { method: 'PUT' })
                .then(r => r.json())
                .then(() => fetchData())
                .catch(e => console.error("Failed to log view", e));
        }
    };

    function showToast(msg) {
        toastMsg.textContent = msg;
        actionToast.classList.remove('hidden');
        void actionToast.offsetWidth;
        actionToast.classList.add('show');
        
        setTimeout(() => {
            actionToast.classList.remove('show');
            setTimeout(() => actionToast.classList.add('hidden'), 400);
        }, 3000);
    }

    // --- Init ---
    initMap();
    fetchData();
    setInterval(fetchData, 10000); // 10s auto-refresh
});
