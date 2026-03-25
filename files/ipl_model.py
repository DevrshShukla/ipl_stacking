import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
matches    = pd.read_csv('/mnt/user-data/uploads/matches.csv')
deliveries = pd.read_csv('/mnt/user-data/uploads/deliveries.csv')

print("="*60)
print("STEP 1: RAW DATA SHAPES")
print(f"matches.csv    : {matches.shape}")
print(f"deliveries.csv : {deliveries.shape}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: PREPROCESSING")

# --- MATCHES PREPROCESSING ---
print("\n[matches.csv]")

# 2a. Drop rows where winner is NaN (abandoned matches – no result to learn from)
before = len(matches)
matches = matches.dropna(subset=['winner'])
print(f"  2a. Dropped {before - len(matches)} abandoned matches (winner=NaN)")

# 2b. Fill city NaN with 'Unknown' – city is used as a contextual feature
matches['city'] = matches['city'].fillna('Unknown')
print(f"  2b. Filled {51} city nulls → 'Unknown'")

# 2c. Fill method NaN (DLS etc.) with 'Normal' – most matches are normal
matches['method'] = matches['method'].fillna('Normal')
print(f"  2c. Filled method nulls → 'Normal'")

# 2d. Fill player_of_match NaN with 'Unknown'
matches['player_of_match'] = matches['player_of_match'].fillna('Unknown')
print(f"  2d. Filled player_of_match nulls → 'Unknown'")

# 2e. Fill target_runs / target_overs nulls (super-over matches)
matches['target_runs']  = matches['target_runs'].fillna(0)
matches['target_overs'] = matches['target_overs'].fillna(0)
print(f"  2e. Filled target_runs/target_overs nulls → 0 (super-over / edge cases)")

# 2f. Parse date → datetime; extract year/month for temporal features
matches['date'] = pd.to_datetime(matches['date'])
matches['match_year']  = matches['date'].dt.year
matches['match_month'] = matches['date'].dt.month
print(f"  2f. Parsed 'date' → match_year, match_month")

# 2g. Normalize team names (franchises renamed over seasons)
name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab':  'Punjab Kings',
    'Rising Pune Supergiant':  'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}
for col in ['team1','team2','toss_winner','winner','batting_team','bowling_team']:
    if col in matches.columns:
        matches[col] = matches[col].replace(name_map)
for col in ['batting_team','bowling_team']:
    deliveries[col] = deliveries[col].replace(name_map)
print(f"  2g. Normalized {len(name_map)} franchise name variants")

# --- DELIVERIES PREPROCESSING ---
print("\n[deliveries.csv]")

# 2h. Keep only innings 1 & 2 (innings 3+ are super-over deliveries – tiny dataset, different rules)
before = len(deliveries)
deliveries = deliveries[deliveries['inning'].isin([1, 2])]
print(f"  2h. Dropped {before - len(deliveries)} super-over delivery rows (inning > 2)")

# 2i. Fill extras_type NaN with 'normal' – most balls are regular deliveries
deliveries['extras_type'] = deliveries['extras_type'].fillna('normal')
print(f"  2i. Filled extras_type NaN → 'normal'")

# 2j. Fill player_dismissed / dismissal_kind / fielder NaN with 'none'
for col in ['player_dismissed','dismissal_kind','fielder']:
    deliveries[col] = deliveries[col].fillna('none')
print(f"  2j. Filled player_dismissed / dismissal_kind / fielder NaN → 'none'")

# 2k. Remove rows where total_runs > 36 (data-entry errors; max legitimate = 7+extras per ball)
before = len(deliveries)
deliveries = deliveries[deliveries['total_runs'] <= 36]
print(f"  2k. Dropped {before - len(deliveries)} rows with total_runs > 36 (likely data errors)")

print(f"\n  ✔ Clean deliveries shape: {deliveries.shape}")
print(f"  ✔ Clean matches shape   : {matches.shape}")

# ─────────────────────────────────────────────
# 3. AGGREGATE INNINGS SCORE (TARGET VARIABLE)
# ─────────────────────────────────────────────
# Sum total_runs per match per inning → this is the innings score we want to predict
innings_score = (
    deliveries.groupby(['match_id','inning','batting_team','bowling_team'])
    ['total_runs'].sum().reset_index()
)
innings_score.rename(columns={'total_runs': 'innings_score'}, inplace=True)

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: FEATURE ENGINEERING")

# --- Ball-level aggregations per innings ---
# 4a. total wickets per innings
wickets = deliveries.groupby(['match_id','inning'])['is_wicket'].sum().reset_index()
wickets.rename(columns={'is_wicket':'total_wickets'}, inplace=True)

# 4b. total extras per innings
extras = deliveries.groupby(['match_id','inning'])['extra_runs'].sum().reset_index()
extras.rename(columns={'extra_runs':'total_extras'}, inplace=True)

# 4c. total balls bowled per innings (legal deliveries only – no wides/no-balls)
legal = deliveries[~deliveries['extras_type'].isin(['wides','noballs'])]
balls_bowled = legal.groupby(['match_id','inning']).size().reset_index(name='balls_bowled')

# 4d. boundary count: 4s and 6s
boundaries = deliveries[deliveries['batsman_runs'].isin([4,6])]
boundary_count = boundaries.groupby(['match_id','inning']).size().reset_index(name='boundary_count')

# 4e. dot-ball count (batsman scored 0, no extras)
dots = deliveries[(deliveries['batsman_runs'] == 0) & (deliveries['extra_runs'] == 0)]
dot_count = dots.groupby(['match_id','inning']).size().reset_index(name='dot_ball_count')

# 4f. unique batters per innings
unique_batters = deliveries.groupby(['match_id','inning'])['batter'].nunique().reset_index()
unique_batters.rename(columns={'batter':'unique_batters'}, inplace=True)

# 4g. first-6-overs run rate (powerplay)
powerplay = deliveries[deliveries['over'] < 6]
pp_runs = powerplay.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
pp_runs.rename(columns={'total_runs':'powerplay_runs'}, inplace=True)

# 4h. middle-overs (6–15) run rate
middle = deliveries[(deliveries['over'] >= 6) & (deliveries['over'] < 16)]
mid_runs = middle.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
mid_runs.rename(columns={'total_runs':'middle_overs_runs'}, inplace=True)

# 4i. death-overs (16–20) runs
death = deliveries[deliveries['over'] >= 16]
death_runs = death.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
death_runs.rename(columns={'total_runs':'death_overs_runs'}, inplace=True)

# 4j. powerplay wickets
pp_wkts = powerplay.groupby(['match_id','inning'])['is_wicket'].sum().reset_index()
pp_wkts.rename(columns={'is_wicket':'powerplay_wickets'}, inplace=True)

# 4k. run rate after 10 overs
first10 = deliveries[deliveries['over'] < 10]
f10_runs = first10.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
f10_runs.rename(columns={'total_runs':'first10_runs'}, inplace=True)

# Merge all ball-level features
feat = innings_score.copy()
for df in [wickets, extras, balls_bowled, boundary_count, dot_count,
           unique_batters, pp_runs, mid_runs, death_runs, pp_wkts, f10_runs]:
    feat = feat.merge(df, on=['match_id','inning'], how='left')

feat.fillna(0, inplace=True)

# Merge match-level features
feat = feat.merge(
    matches[['id','season','city','venue','toss_winner','toss_decision',
             'match_type','match_year','match_month']],
    left_on='match_id', right_on='id', how='left'
)

# 4l. toss_advantage: 1 if batting team won the toss, else 0
feat['toss_advantage'] = (feat['batting_team'] == feat['toss_winner']).astype(int)

# 4m. is_first_innings
feat['is_first_innings'] = (feat['inning'] == 1).astype(int)

# 4n. run_rate (actual RPO)
feat['run_rate'] = feat['innings_score'] / (feat['balls_bowled'] / 6 + 1e-9)

# 4o. boundary_rate: boundaries per over
feat['boundary_rate'] = feat['boundary_count'] / (feat['balls_bowled'] / 6 + 1e-9)

# 4p. wicket_rate: wickets per over
feat['wicket_rate'] = feat['total_wickets'] / (feat['balls_bowled'] / 6 + 1e-9)

# 4q. economy: runs per ball
feat['economy_per_ball'] = feat['innings_score'] / (feat['balls_bowled'] + 1e-9)

# 4r. extras_percentage
feat['extras_pct'] = feat['total_extras'] / (feat['innings_score'] + 1e-9) * 100

# 4s. powerplay contribution %
feat['pp_contribution_pct'] = feat['powerplay_runs'] / (feat['innings_score'] + 1e-9) * 100

# 4t. season number (ordinal – captures IPL maturity/evolution)
season_order = {s: i for i, s in enumerate(sorted(matches['season'].unique()))}
feat['season_num'] = feat['season'].map(season_order)

print("  Features created:")
engineered = ['total_wickets','total_extras','balls_bowled','boundary_count','dot_ball_count',
              'unique_batters','powerplay_runs','middle_overs_runs','death_overs_runs',
              'powerplay_wickets','first10_runs','toss_advantage','is_first_innings',
              'run_rate','boundary_rate','wicket_rate','economy_per_ball',
              'extras_pct','pp_contribution_pct','season_num']
for f in engineered:
    print(f"    ✔ {f}")

# ─────────────────────────────────────────────
# 5. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
cat_cols = ['batting_team','bowling_team','city','venue','toss_decision','match_type']
le = {}
for c in cat_cols:
    feat[c] = feat[c].astype(str)
    le[c] = LabelEncoder()
    feat[c+'_enc'] = le[c].fit_transform(feat[c])

# ─────────────────────────────────────────────
# 6. MODEL DATASET
# ─────────────────────────────────────────────
feature_cols = (
    ['batting_team_enc','bowling_team_enc','city_enc','venue_enc',
     'toss_decision_enc','match_type_enc','toss_advantage','is_first_innings',
     'season_num','match_year','match_month','inning']
    + engineered
)
# Remove leaky features (they implicitly contain innings_score)
leaky = ['run_rate','economy_per_ball','innings_score']
feature_cols = [f for f in feature_cols if f not in leaky]

X = feat[feature_cols].copy()
y = feat['innings_score'].copy()

# Drop any remaining NaN rows
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]

print(f"\n  Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 7. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: MODEL TRAINING & EVALUATION")

results = {}

def evaluate(name, model, X_tr, X_te, y_tr, y_te, scaled=False):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mse  = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, preds)
    print(f"\n  [{name}]")
    print(f"    MSE  : {mse:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")
    results[name] = {'MSE': round(mse,4), 'RMSE': round(rmse,4), 'R2': round(r2,4)}
    return model, rmse

lr_model,  lr_rmse  = evaluate("Linear Regression",        LinearRegression(),              X_train_sc, X_test_sc, y_train, y_test)
dt_model,  dt_rmse  = evaluate("Decision Tree Regressor",  DecisionTreeRegressor(max_depth=10, random_state=42), X_train, X_test, y_train, y_test)
rf_model,  rf_rmse  = evaluate("Random Forest Regressor",  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), X_train, X_test, y_train, y_test)
svm_model, svm_rmse = evaluate("SVM (SVR)",                SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1), X_train_sc, X_test_sc, y_train, y_test)

# ─────────────────────────────────────────────
# 8. STACKING (best 2 models)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: STACKING ENSEMBLE")

# Rank by RMSE (lower is better)
ranked = sorted(results.items(), key=lambda x: x[1]['RMSE'])
best2 = [ranked[0][0], ranked[1][0]]
print(f"\n  Best 2 models: {best2[0]} (RMSE={ranked[0][1]['RMSE']}) | {best2[1]} (RMSE={ranked[1][1]['RMSE']})")

# Map model names to objects + whether they need scaling
model_map = {
    "Linear Regression":       (LinearRegression(), True),
    "Decision Tree Regressor": (DecisionTreeRegressor(max_depth=10, random_state=42), False),
    "Random Forest Regressor": (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
    "SVM (SVR)":               (SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1), True),
}

# For stacking, use unscaled X (sklearn pipeline handles internally via base estimators)
from sklearn.pipeline import make_pipeline
estimators = []
for name in best2:
    mdl, needs_scale = model_map[name]
    if needs_scale:
        pipe = make_pipeline(StandardScaler(), mdl)
    else:
        pipe = mdl
    estimators.append((name.replace(' ','_').lower(), pipe))

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5, passthrough=False
)
stack.fit(X_train, y_train)
stack_preds = stack.predict(X_test)
s_mse  = mean_squared_error(y_test, stack_preds)
s_rmse = np.sqrt(s_mse)
s_r2   = r2_score(y_test, stack_preds)

print(f"\n  [Stacking Ensemble ({best2[0]} + {best2[1]})]")
print(f"    MSE  : {s_mse:.4f}")
print(f"    RMSE : {s_rmse:.4f}")
print(f"    R²   : {s_r2:.4f}")
results['Stacking Ensemble'] = {'MSE': round(s_mse,4), 'RMSE': round(s_rmse,4), 'R2': round(s_r2,4), 'components': best2}

# ─────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: FINAL SUMMARY")
print(f"\n{'Model':<35} {'MSE':>10} {'RMSE':>10} {'R²':>8}")
print("-"*65)
for name, m in results.items():
    print(f"{name:<35} {m['MSE']:>10.4f} {m['RMSE']:>10.4f} {m['R2']:>8.4f}")

# Save results to JSON for docx generation
with open('/home/claude/model_results.json','w') as f:
    json.dump({'results': results, 'best2': best2,
               'feature_cols': feature_cols,
               'n_train': int(len(X_train)),
               'n_test':  int(len(X_test)),
               'n_total': int(len(X))}, f, indent=2)

print("\n✅ Results saved to /home/claude/model_results.json")
