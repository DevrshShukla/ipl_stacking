const API_URL = "http://127.0.0.1:8000/api";

let predictionChart = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    fetchTeamsAndVenues();
    initChart();

    document.getElementById('prediction-form').addEventListener('submit', handlePredict);
});

async function fetchTeamsAndVenues() {
    try {
        const [teamsRes, venuesRes] = await Promise.all([
            fetch(`${API_URL}/teams`),
            fetch(`${API_URL}/venues`)
        ]);

        if (teamsRes.ok && venuesRes.ok) {
            const teamsData = await teamsRes.json();
            const venuesData = await venuesRes.json();

            populateDropdown('batting-team', teamsData.teams);
            populateDropdown('bowling-team', teamsData.teams);
            populateDropdown('venue', venuesData.venues);
        }
    } catch (error) {
        console.error("Error fetching initial data:", error);
        showError("Could not connect to the backend API.");
    }
}

function populateDropdown(id, items) {
    const select = document.getElementById(id);
    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
    });
}

async function handlePredict(e) {
    e.preventDefault();
    showError("");

    const battingTeam = document.getElementById('batting-team').value;
    const bowlingTeam = document.getElementById('bowling-team').value;
    const venue = document.getElementById('venue').value;
    const currentScore = parseInt(document.getElementById('current-score').value);
    const oversCompleted = parseFloat(document.getElementById('overs-completed').value);
    const wickets = parseInt(document.getElementById('wickets').value);

    // Basic Validation
    if (battingTeam === bowlingTeam) {
        showError("Batting and Bowling teams cannot be the same.");
        return;
    }

    const payload = {
        batting_team: battingTeam,
        bowling_team: bowlingTeam,
        venue: venue,
        current_score: currentScore,
        overs_completed: oversCompleted,
        wickets: wickets
    };

    try {
        const btn = document.querySelector('.predict-btn');
        btn.textContent = "Predicting...";
        btn.disabled = true;

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            updateResult(data.predicted_score, currentScore, oversCompleted);
        } else {
            showError(data.detail || "Error predicting score");
        }
    } catch (error) {
        showError("Failed to fetch prediction. Ensure backend is running.");
    } finally {
        const btn = document.querySelector('.predict-btn');
        btn.textContent = "Predict Final Score";
        btn.disabled = false;
    }
}

function updateResult(predictedScore, currentScore, overs) {
    // Animate score
    const scoreElement = document.getElementById('predicted-score');
    animateValue(scoreElement, 0, predictedScore, 1000);

    // Update Chart
    updateChart(currentScore, predictedScore, overs);
}

function showError(msg) {
    document.getElementById('error-message').textContent = msg;
}

function initChart() {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Current Score', 'Predicted Final Score'],
            datasets: [{
                label: 'Runs',
                data: [0, 0],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(168, 85, 247, 0.7)'
                ],
                borderColor: [
                    'rgb(59, 130, 246)',
                    'rgb(168, 85, 247)'
                ],
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function updateChart(current, predicted, overs) {
    predictionChart.data.datasets[0].data = [current, predicted];
    // Calculate projected score based on run rate
    const currentRR = current / overs;
    const projectedLinear = Math.round(currentRR * 20);
    
    // Add third bar for linear projection
    predictionChart.data.labels = ['Current', 'Projected (Current RR)', 'ML Prediction'];
    predictionChart.data.datasets[0].data = [current, projectedLinear, predicted];
    predictionChart.data.datasets[0].backgroundColor = [
        'rgba(59, 130, 246, 0.7)',
        'rgba(236, 72, 153, 0.7)',
        'rgba(168, 85, 247, 0.7)'
    ];
    predictionChart.data.datasets[0].borderColor = [
        'rgb(59, 130, 246)',
        'rgb(236, 72, 153)',
        'rgb(168, 85, 247)'
    ];
    
    predictionChart.update();
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
