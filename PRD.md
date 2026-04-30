# Product Requirements Document (PRD)
## IPL First Innings Score Prediction System

### 1. Product Overview
The IPL First Innings Score Prediction System is a web-based machine learning application designed to predict the final total score of the batting team in the first innings of an Indian Premier League (IPL) match. By capturing live match details—such as the current score, overs completed, playing teams, and venue—the system will provide users with a highly accurate score projection, enhancing fan engagement and assisting fantasy league players.

### 2. Problem Statement
Cricket, especially the T20 format like IPL, is highly dynamic. Fans, analysts, and fantasy sports players often try to estimate what a competitive total would be based on the current match situation. However, human estimations are often biased and lack statistical backing. There is a need for a data-driven tool that uses historical match data to accurately forecast the first innings score based on the current match state.

### 3. Objectives
- **Accurate Prediction**: Provide a reliable first innings total score prediction using historical IPL data.
- **Simplicity**: Offer a minimal, intuitive user interface that requires no technical expertise.
- **Fast Response**: Return predictions in near real-time.
- **Visual Insights**: Present data visually to help users understand the context of the prediction (e.g., predicted score vs. historical average).

### 4. Target Users
- **Cricket Fans & Enthusiasts**: Looking for real-time predictions while watching live matches.
- **Fantasy Cricket Players**: Seeking data-driven insights to make informed decisions about team selections and contest strategies.
- **Sports Analysts/Bloggers**: Needing quick statistical projections for content creation.

### 5. Features List
#### Core Features
- **Team Selection**: Dropdown menus to select the Batting Team and Bowling Team.
- **Venue Selection**: Dropdown menu to select the stadium/venue.
- **Match State Inputs**: Numeric input fields for "Current Runs" and "Overs Completed".
- **Prediction Engine**: A "Predict" button that processes the inputs and displays the projected first innings total.

#### Optional Features
- **Additional Match Variables**: Numeric inputs for "Wickets Fallen" and "Current Run Rate" to improve prediction accuracy.
- **Historical Comparison**: A visual graph comparing the predicted score against the historical average score at the selected venue.

### 6. User Flow (Step-by-step)
1. **Landing**: The user opens the web application and sees a clean, simple dashboard.
2. **Input Selection**: 
   - User selects the Batting Team from a dropdown.
   - User selects the Bowling Team from a dropdown.
   - User selects the Venue from a dropdown.
3. **Data Entry**: 
   - User enters the "Current Runs" (e.g., 85).
   - User enters the "Overs Completed" (e.g., 10.2).
4. **Action**: User clicks the "Predict Score" button.
5. **Result Display**: The system displays the predicted final score (e.g., "Predicted Score: 175 - 185").
6. **Visualization**: Below the prediction, a graph appears showing the predicted score against the venue's historical average.

### 7. UI/UX Requirements
- **Theme**: Clean, responsive, and mobile-friendly design (preferably a cricket-themed color palette like blue, green, and white).
- **Form Layout**: A single-column or neatly aligned two-column form for easy data entry.
- **Validation**: 
  - Dropdowns must not allow identical selections for Batting and Bowling teams.
  - "Overs Completed" must be between 5.0 and 19.5 (predictions are unreliable in the first few overs).
  - "Current Runs" must be a positive integer.
- **Visuals**: Use clear typography for the final predicted score to make it stand out. 

### 8. Functional Requirements
- The system must accept exactly 5 mandatory inputs.
- The system must communicate with the ML backend API via HTTP POST request.
- The system must handle decimal inputs for overs appropriately (e.g., 10.1 means 10 overs and 1 ball).
- The system must render a dynamic chart (bar chart or gauge) upon receiving the prediction.

### 9. Non-Functional Requirements
- **Performance**: The prediction API should respond in under 500ms.
- **Availability**: The web application should be highly available, especially during the IPL season.
- **Responsiveness**: The UI must adapt to mobile, tablet, and desktop screens seamlessly.

### 10. Success Metrics (KPIs)
- **User Engagement**: Number of predictions made per session.
- **Model Accuracy**: Mean Absolute Error (MAE) of predictions vs. actual match results (tracked offline).
- **System Performance**: API latency and uptime.

### 11. Assumptions
- The dataset provided (`matches.csv` and `deliveries.csv`) contains sufficient historical data to train a reliable model.
- Weather conditions and pitch changes are not explicitly factored into this v1.0 model.
- The user will primarily use the app during live matches (typically after the powerplay).

### 12. Constraints
- The prediction accuracy is limited by the unpredictability of the sport (e.g., sudden batting collapses).
- Only teams and venues present in the historical dataset can be supported.

### 13. Future Scope
- **Second Innings Prediction**: Predict the probability of the chasing team winning based on the target.
- **Player-Specific Data**: Incorporate striker/non-striker data for more granular predictions.
- **Live API Integration**: Automatically fetch live match data so the user doesn't have to input current runs and overs manually.
