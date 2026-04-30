# Technical Requirements Document (TRD)
## IPL First Innings Score Prediction System

### 1. System Architecture
The application will follow a standard client-server architecture:
- **Frontend (Client)**: A lightweight, responsive web interface where users input match details and view predictions and graphs.
- **Backend (API Server)**: A RESTful API that receives input data, preprocesses it, feeds it to the trained ML model, and returns the prediction.
- **ML Model**: A pre-trained machine learning model serialized (e.g., using `pickle` or `joblib`) and loaded into the backend memory for fast inference.

### 2. Data Pipeline
The pipeline will process the raw `matches.csv` and `deliveries.csv` files.
- **Data Cleaning**:
  - Merge the deliveries and matches datasets to get venue and team information for each ball bowled.
  - Filter out matches affected by rain (D/L method) to ensure standard 20-over innings.
  - Keep only current/recent IPL teams (remove or map defunct teams like Pune Warriors).
- **Feature Selection**:
  - Relevant columns: `batting_team`, `bowling_team`, `venue`, `total_runs` (cumulative), `over`, `ball`.
- **Feature Engineering**:
  - Calculate `current_score` at every ball.
  - Convert `over` and `ball` into a single continuous variable: `overs_completed` (e.g., Over 5, Ball 2 = 5 + 2/6 = 5.33 overs).
  - Calculate `runs_in_last_5_overs` (optional but highly recommended for accuracy).
  - Encode categorical variables (`batting_team`, `bowling_team`, `venue`) using One-Hot Encoding.

### 3. Model Details
- **Algorithms to Evaluate**: 
  - *Linear Regression*: As a fast, interpretable baseline model.
  - *Decision Tree Regressor*: To capture non-linear relationships.
  - *Random Forest Regressor*: For robust predictions and reducing overfitting.
  - *XGBoost / Gradient Boosting Regressor*: Recommended for highest accuracy on this tabular dataset.
- **Model Training Process**:
  - **Train-Test Split**: Time-based splitting (e.g., train on 2008-2015 data, test on 2016-2017 data) to prevent data leakage from the future into the past.
  - Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
- **Evaluation Metrics**:
  - **MAE (Mean Absolute Error)**: Primary metric (e.g., being off by 10 runs).
  - **RMSE (Root Mean Squared Error)**: To heavily penalize large prediction errors.
  - **R² Score**: To measure how well the variance in the score is explained by the features.

### 4. Input Features (Final 5 Selected)
The model will expect an array of features derived from the following 5 user inputs:
1. **Batting Team** (Categorical -> One-Hot Encoded)
2. **Bowling Team** (Categorical -> One-Hot Encoded)
3. **Venue** (Categorical -> One-Hot Encoded)
4. **Current Runs** (Numeric, Integer)
5. **Overs Completed** (Numeric, Float - e.g., 10.5)

*(Note: The model pipeline must include a `ColumnTransformer` or save the feature names to ensure the exact same dummy columns are passed during inference as were used during training).*

### 5. Model Deployment Strategy
- **API Creation**: Use **FastAPI** (preferred for speed and automatic Swagger documentation) or **Flask**.
- **Model Serialization**: Save the trained model and the One-Hot Encoder using `joblib` or `pickle`. Load these into memory when the API server starts to avoid disk reads during requests.
- **Integration**: The frontend will make an asynchronous AJAX call (`fetch` API or `axios`) to the `/predict` endpoint with a JSON payload.

### 6. Graph/Visualization Implementation
- **Libraries**: **Chart.js** or **Recharts** for rendering interactive, client-side graphs.
- **What to Show**:
  - **Bar Chart**: A visual comparison showing the "Predicted Score" alongside the "Average First Innings Score at this Venue".
  - **Distribution Curve (Optional)**: A chart displaying the range of possible scores (e.g., plotting the confidence interval or variance).

### 7. Tech Stack
- **Frontend**: HTML5, CSS3 (Vanilla or Tailwind CSS), JavaScript (Vanilla JS or React).
- **Backend**: Python 3.9+, FastAPI (or Flask), Uvicorn (ASGI server).
- **ML Libraries**: Scikit-learn, Pandas, NumPy, XGBoost.

### 8. API Design
**Endpoint**: `POST /predict`

**Request Format (JSON)**:
```json
{
  "batting_team": "Chennai Super Kings",
  "bowling_team": "Mumbai Indians",
  "venue": "Wankhede Stadium",
  "current_runs": 85,
  "overs_completed": 10.2
}
```

**Response Format (JSON)**:
```json
{
  "predicted_score": 178,
  "venue_average": 165,
  "status": "success"
}
```

### 9. Folder Structure
```text
ipl_score_predictor/
│
├── backend/
│   ├── app.py                 # FastAPI/Flask application
│   ├── model/
│   │   ├── model.pkl          # Serialized ML model
│   │   └── encoder.pkl        # Serialized categorical encoder
│   ├── requirements.txt       # Python dependencies
│   └── notebooks/
│       └── model_training.ipynb # Jupyter notebook for EDA and training
│
└── frontend/
    ├── index.html             # Main UI
    ├── style.css              # Styling
    ├── script.js              # API calls and Chart rendering
    └── assets/                # Images, icons
```

### 10. Scalability Considerations
- **Stateless API**: The backend API should be completely stateless, allowing it to be easily containerized (using Docker) and scaled horizontally using a load balancer during high-traffic live matches.
- **Caching**: Historical data computations (like venue averages) should be pre-computed and cached in memory.

### 11. Error Handling
- **Frontend Validation**: Prevent form submission if inputs are missing or illogical (e.g., overs > 20, or Batting Team == Bowling Team).
- **Backend Validation**: The API must validate the incoming JSON schema using Pydantic (in FastAPI) to ensure data types are correct before inference.
- **Graceful Degradation**: If the ML model fails, the API should return a meaningful HTTP 500 error, and the UI should display a user-friendly alert.

### 12. Security Considerations
- **CORS Configuration**: The backend must configure Cross-Origin Resource Sharing (CORS) to only accept API requests from the designated frontend domain.
- **Input Sanitization**: Strictly type-check inputs to prevent injection attacks and ensure the model pipeline doesn't crash on unexpected string inputs.
- **Rate Limiting**: Implement basic IP-based rate limiting on the `/predict` endpoint to prevent abuse or denial-of-service (DoS) attacks.
