# Hotel Booking Cancellation Prediction System

## Project Overview

This project develops a machine learning system to predict hotel booking cancellations for Alpha Hotel. The system aims to help hotel management make data-driven decisions to optimize revenue, reduce operational inefficiencies, and improve customer relationship management.

## Table of Contents

- [Business Understanding](#business-understanding)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Contributors](#contributors)

---

## Business Understanding

### Background

Between 2015 and 2017, Portugal's hospitality industry experienced significant growth, with over 21 million visitors in 2017 alone. However, this growth brought operational challenges, particularly with increasing cancellation rates driven by Online Travel Agencies (OTA) and flexible cancellation policies.

### Business Problem

Hotels face substantial challenges from unpredictable cancellations:

- **Financial Loss**: Empty rooms that cannot be resold lead to direct revenue loss, especially during high season
- **Operational Inefficiency**: Resources allocated based on reservations (staff, housekeeping, logistics) are wasted when cancellations occur
- **Planning Uncertainty**: Unpredictable cancellations hamper revenue forecasting and strategic marketing decisions

### Stakeholders

| No  | Stakeholder                     | Role                                                     |
| --- | ------------------------------- | -------------------------------------------------------- |
| 1   | Hotel Management                | Strategic decision-making                                |
| 2   | Reservation & Front Office Team | Manages bookings and guest interactions                  |
| 3   | Marketing & CRM Team            | Executes promotions and maintains customer relationships |
| 4   | Customers / Hotel Guests        | Service users                                            |
| 5   | Hotel Owners / Investors        | Capital owners and long-term decision makers             |

### Project Objectives

**Primary Goal**: Build a supervised learning model to classify whether a reservation will be cancelled or not, enabling management to:

- Take preventive actions (re-confirmation, promotions, upgrades)
- Adjust pricing and overbooking strategies
- Improve operational efficiency and revenue
- Reduce customer complaints and maintain hotel reputation
- Understand root causes of cancellations

**Specific Objectives**:

1. Identify customer behavior patterns leading to cancellations
2. Provide cancellation probability scores for each reservation
3. Support business decisions regarding overbooking, discounts, and proactive communication

### Business Impact Analysis

Using confusion matrix to understand business implications:

|                            | Predicted: Not Cancel (0) | Predicted: Cancel (1) |
| -------------------------- | ------------------------- | --------------------- |
| **Actual: Not Cancel (0)** | True Negative (TN)        | False Positive (FP)   |
| **Actual: Cancel (1)**     | False Negative (FN)       | True Positive (TP)    |

**False Negative (FN)** - Most costly: Model predicts no cancellation but booking is cancelled

- Hotel loses potential revenue as empty room cannot be resold
- Additional operational costs already incurred (staff, logistics)
- Cost: $500 per FN

**False Positive (FP)** - Less costly: Model predicts cancellation but booking proceeds

- May lead to overbooking issues
- However, revenue is still generated from the booking
- Cost: $100 per FP

**True Positive (TP)** - Value generation: Correctly predicts cancellation

- Hotel can take preventive measures
- Opportunity for persuasive strategies and promotions
- Revenue: $200 per TP

### Business Metrics

```
Cost Assumptions:
- FP Cost = $100
- FN Cost = $500
- TP Revenue = $200

Formula:
- total_cost = (fp × $100) + (fn × $500)
- total_revenue = tp × $200
- net_benefit = total_revenue - total_cost
- roi = (net_benefit / total_cost) × 100
```

### Machine Learning Strategy

Build a cancellation prediction model using Supervised Learning that:

- Predicts `is_canceled` based on historical customer and booking data
- Optimizes for **F2 Score** (beta=2) - emphasizes recall over precision
- Minimizes False Negatives (FN) which are more costly ($500) than False Positives ($100)

**Why F2 Score?**

The F2 Score weighs recall 2:1 over precision, making it ideal for scenarios where missing a cancellation (FN) is more costly than a false alarm (FP).

Formula:

```
F2 = (5 × Precision × Recall) / (4 × Precision + Recall)
```

**Expected Business Benefits**:

1. **Reduce Losses from Unexpected Cancellations**

   - More sensitive to potential cancellations
   - Enables preventive actions (discounts, confirmations)
   - Safer overbooking strategies

2. **Improve Operational Efficiency**

   - Avoid resource waste (staff, food, housekeeping)
   - Optimize staff scheduling and room preparation
   - Reduce fixed costs without revenue

3. **Maximize Model ROI**
   - Reducing FN decreases $500 losses per case
   - Increases net benefit and ROI

---

## Dataset Information

**Source**: Hotel booking data from two hotels in Portugal (2015-2017)

**Dataset Characteristics**:

- Total records: 119,390 bookings
- Features: 32 variables (after data cleaning)
- Target variable: `is_canceled` (0 = Not Cancelled, 1 = Cancelled)
- Cancellation rate: ~27%
- Class imbalance: Present (handled with oversampling techniques)

**Key Features**:

| Category                | Features                                                                                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| **Booking Information** | lead_time, arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month |
| **Stay Details**        | stays_in_weekend_nights, stays_in_week_nights, adults, children, babies                               |
| **Services**            | meal, reserved_room_type, assigned_room_type, required_car_parking_spaces, total_of_special_requests  |
| **Customer Profile**    | is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, customer_type              |
| **Booking Channel**     | market_segment, distribution_channel, agent, company, country                                         |
| **Financial**           | adr (Average Daily Rate), deposit_type, booking_changes, days_in_waiting_list                         |

---

## Project Structure

```
Final Project New/
│
├── README.md                                    # This file
├── alphahotel.jpeg                             # Hotel logo
│
├── 01_Data_Cleansing.ipynb                     # Data cleaning and preparation
├── 02_exploration_data_analysis_revisi.ipynb   # EDA and feature analysis
├── 03_preprocessing_modeling_revisi2.ipynb     # Model training and evaluation
│
├── hotel_bookings.csv                          # Raw dataset
├── 1_hotel_bookings_prepared.csv               # Cleaned dataset
├── 2_hotel_bookings_test.csv                   # Test dataset
│
├── models/                                      # Saved models and applications
│   ├── best_hotel_cancellation_model_20251015_122216.sav
│   ├── preprocessor_pipeline_20251015_122216.sav
│   ├── model_metadata_20251015_122216.json
│   ├── hotel_cancellation_app.py               # Streamlit app - Detailed analysis
│   ├── customer_reservation_app.py             # Streamlit app - Customer interface
│   └── alphahotel.jpeg                         # Hotel logo for apps
│
└── .venv_312/                                   # Python 3.12 virtual environment
```

---

## Installation & Requirements

### System Requirements

- Python 3.12.x
- Minimum 4GB RAM
- 2GB free disk space

### Dependencies

```bash
pandas>=2.3.3
numpy>=2.3.3
scikit-learn==1.6.1
xgboost>=3.0.5
lightgbm>=4.5.0
imbalanced-learn>=0.14.0
category_encoders>=2.8.1
matplotlib>=3.9.0
seaborn>=0.13.0
joblib>=1.5.2
streamlit>=1.50.0
```

### Installation Steps

1. **Clone or download the project**

   ```bash
   cd "/path/to/Final Project New"
   ```

2. **Create virtual environment with Python 3.12**

   ```bash
   python3.12 -m venv .venv_312
   ```

3. **Activate virtual environment**

   ```bash
   # macOS/Linux
   source .venv_312/bin/activate

   # Windows
   .venv_312\Scripts\activate
   ```

4. **Install required packages**

   ```bash
   pip install --upgrade pip
   pip install pandas numpy scikit-learn==1.6.1 xgboost lightgbm \
               imbalanced-learn category_encoders matplotlib seaborn \
               joblib streamlit
   ```

5. **Verify installation**
   ```bash
   python -c "import sklearn; print(sklearn.__version__)"  # Should print 1.6.1
   python -c "import category_encoders; print('OK')"       # Should print OK
   ```

---

## Usage Guide

### Step 1: Data Preparation

**Notebook**: `01_Data_Cleansing.ipynb`

**Purpose**: Clean and prepare raw hotel booking data

**Process**:

1. Load raw dataset (`hotel_bookings.csv`)
2. Handle missing values and data types
3. Remove duplicates and outliers
4. Fix data inconsistencies
5. Save cleaned dataset (`1_hotel_bookings_prepared.csv`)

**How to run**:

```bash
# Open Jupyter Notebook
jupyter notebook 01_Data_Cleansing.ipynb

# Or use VS Code with Jupyter extension
# Execute all cells sequentially
```

**Key Outputs**:

- `1_hotel_bookings_prepared.csv` - Cleaned dataset ready for analysis
- Data quality report
- Basic statistics

---

### Step 2: Exploratory Data Analysis

**Notebook**: `02_exploration_data_analysis_revisi.ipynb`

**Purpose**: Understand data patterns and feature relationships

**Analysis Performed**:

1. **Target Variable Analysis**

   - Cancellation rate: ~27%
   - Class distribution visualization

2. **Hotel Type Analysis**

   - City Hotel vs Resort Hotel comparison
   - Cancellation patterns by hotel type

3. **Temporal Analysis**

   - Lead time impact on cancellations
   - Seasonal patterns (arrival month, week)
   - Booking trends over time

4. **Customer Behavior Analysis**

   - Repeated guests vs new customers
   - Previous cancellation history impact
   - Customer type segmentation

5. **Booking Characteristics**

   - Stay duration analysis
   - Guest composition (adults, children, babies)
   - Room type preferences
   - Meal plan selection

6. **Market Segment Analysis**

   - Online TA, Offline TA/TO, Direct, Corporate, Groups
   - Distribution channel effectiveness

7. **Financial Analysis**
   - ADR (Average Daily Rate) distribution
   - Deposit type impact
   - Special requests correlation

**Key Insights**:

- Longer lead times correlate with higher cancellation rates
- Online TA bookings have higher cancellation probability
- City Hotels experience more cancellations than Resort Hotels
- Transient customers are more likely to cancel
- No deposit bookings have higher cancellation rates

**How to run**:

```bash
jupyter notebook 02_exploration_data_analysis_revisi.ipynb
```

**Key Outputs**:

- Comprehensive visualizations
- Statistical summaries
- Feature importance insights
- Business intelligence for stakeholders

---

### Step 3: Model Development and Training

**Notebook**: `03_preprocessing_modeling_revisi2.ipynb`

**Purpose**: Build and optimize machine learning models

**Process Workflow**:

1. **Data Preprocessing**

   - Feature engineering (total_nights, weekend_ratio, booking_complexity, etc.)
   - Categorical encoding (OneHot, Target encoding)
   - Numerical scaling (RobustScaler)
   - Pipeline creation

2. **Train-Test Split**

   - 80% training, 20% testing
   - Stratified split to maintain class distribution

3. **Benchmark Models** (Cell 13-24)

   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
   - Gradient Boosting
   - Evaluation using F2 Score as primary metric

4. **Handling Class Imbalance** (Cell 25-35)

   - K-Fold Cross Validation (5 folds)
   - Testing multiple oversampling techniques:
     - RandomOverSampler
     - SMOTE
     - ADASYN
   - Combination testing with different models

5. **Model Selection** (Cell 36-43)

   - Top 3 combinations based on F2 Score:
     1. RandomOverSampler + LightGBM (F2: 0.7912)
     2. RandomOverSampler + XGBoost (F2: 0.7869)
     3. SMOTE + XGBoost (comparison baseline)

6. **Hyperparameter Tuning** (Cell 44-49)

   - RandomizedSearchCV for RandomOverSampler + XGBoost
   - 100-150 iterations
   - 5-fold stratified cross-validation
   - Optimizing for F2 Score

7. **Final Model Evaluation** (Cell 50-55)

   - Performance metrics on test set
   - Confusion matrix analysis
   - Business impact calculation
   - ROI assessment

8. **Model Interpretation** (Cell 56-58)

   - SHAP analysis for feature importance
   - Individual prediction explanations
   - Feature contribution visualization

9. **Model Persistence** (Cell 59-60)
   - Save best model as `.sav` file
   - Save preprocessing pipeline
   - Export model metadata (JSON)

**Final Model Performance**:

```
Model: RandomOverSampler + XGBoost (F2 Score Optimized)

Performance Metrics:
- F2 Score:    0.8155 (primary metric)
- Recall:      0.9292 (92.92% - catches most cancellations)
- Precision:   0.5474 (54.74%)
- ROC-AUC:     0.9159 (excellent discrimination)
- Accuracy:    0.7693 (76.93%)

Business Impact:
- Net Benefit:     $353,800
- ROI:             65.62%
- Total Revenue:   $893,000
- Total Cost:      $539,200
```

**Why This Model?**

The F2 Score optimization prioritizes recall (92.92%), meaning:

- Catches 93 out of 100 actual cancellations
- Reduces costly False Negatives (FN @ $500 each)
- Accepts more False Positives (FP @ $100 each)
- Maximizes business value and ROI

**How to run**:

```bash
jupyter notebook 03_preprocessing_modeling_revisi2.ipynb
```

**Important Notes**:

- Cells must be executed sequentially
- Training time: ~30-45 minutes for full hyperparameter tuning
- Requires sufficient RAM for large dataset operations
- Model files will be saved in current directory

**Key Outputs**:

- `best_hotel_cancellation_model_20251015_122216.sav` - Trained model
- `preprocessor_pipeline_20251015_122216.sav` - Preprocessing pipeline
- `model_metadata_20251015_122216.json` - Model information and metrics
- Performance visualizations and reports

---

## Deployment

### Streamlit Applications

Two Streamlit applications are provided for different use cases:

#### 1. Hotel Cancellation Predictor (Detailed Analysis)

**File**: `models/hotel_cancellation_app.py`

**Purpose**: Comprehensive booking analysis tool for hotel staff

**Features**:

- Detailed input form for all booking parameters
- Real-time cancellation probability prediction
- Risk level assessment (Low/Medium/High)
- Key risk factors identification
- Revenue impact analysis
- Business intelligence dashboard
- Risk mitigation strategies

**How to run**:

```bash
# Navigate to models directory
cd models

# Activate virtual environment
source ../.venv_312/bin/activate  # macOS/Linux
# or
..\.venv_312\Scripts\activate     # Windows

# Run Streamlit app
streamlit run hotel_cancellation_app.py

# App will open at http://localhost:8501
```

**Usage**:

1. Enter complete booking information:
   - Hotel type, arrival dates
   - Stay duration, guest details
   - Room type, meal plan
   - Customer profile, booking history
   - Financial details, country
2. Click "Predict Cancellation Risk"
3. Review prediction results and recommendations

**Model Information Display**:

- Algorithm: XGBoost with RandomOverSampler (F2 Score Optimized)
- F2 Score: 0.8155
- Recall: 92.92%
- ROC-AUC: 0.9159
- Net Benefit: $353,800
- ROI: 65.62%

---

#### 2. Customer Reservation System (Simplified Interface)

**File**: `models/customer_reservation_app.py`

**Purpose**: User-friendly reservation interface for customers

**Features**:

- Simplified booking form
- Room type selection with pricing
- Automatic pricing calculation (including peak season adjustments)
- Booking confidence score
- Risk assessment
- Overbooking recommendations
- Revenue management strategies

**Room Types & Pricing**:

```
A - Standard Room:        $85/night
B - Superior Room:        $95/night
C - Deluxe Room:          $110/night
D - Junior Suite:         $125/night
E - Executive Room:       $150/night
F - Suite:                $175/night
G - Premium Suite:        $200/night
H - Presidential Suite:   $250/night
I - Family Room:          $120/night
J - Twin Room:            $90/night
K - Ocean View:           $160/night
L - Garden View:          $180/night
```

**How to run**:

```bash
# Navigate to models directory
cd models

# Activate virtual environment
source ../.venv_312/bin/activate  # macOS/Linux
# or
..\.venv_312\Scripts\activate     # Windows

# Run Streamlit app on different port
streamlit run customer_reservation_app.py --server.port 8502

# App will open at http://localhost:8502
```

**Usage**:

1. Select hotel type (City Hotel / Resort Hotel)
2. Choose check-in date
3. Specify stay duration
4. Enter guest details (adults, children, babies)
5. Select room type and meal plan
6. Optional: parking and special requests
7. Click "Check Booking Confidence"
8. Review booking analysis and confidence score

**Confidence Levels**:

- High Confidence (>70%): Booking likely to proceed
- Medium Confidence (50-70%): Monitor booking status
- Low Confidence (<50%): High cancellation risk

---

### Running Both Apps Simultaneously

You can run both applications at the same time on different ports:

```bash
# Terminal 1 - Hotel Cancellation Predictor
cd models
source ../.venv_312/bin/activate
streamlit run hotel_cancellation_app.py

# Terminal 2 - Customer Reservation System
cd models
source ../.venv_312/bin/activate
streamlit run customer_reservation_app.py --server.port 8502
```

Access:

- Hotel Cancellation Predictor: http://localhost:8501
- Customer Reservation System: http://localhost:8502

---

### Troubleshooting

**Issue**: Error loading model - `_RemainderColsList` attribute error

**Solution**: Ensure Python 3.12 and scikit-learn 1.6.1 are being used

```bash
python --version  # Should show 3.12.x
python -c "import sklearn; print(sklearn.__version__)"  # Should show 1.6.1
```

**Issue**: Module `category_encoders` not found

**Solution**: Install the package

```bash
pip install category_encoders
```

**Issue**: XGBoost serialization warning

**Solution**: This is informational only and does not affect functionality. The model works correctly.

**Issue**: Streamlit port already in use

**Solution**: Use a different port

```bash
streamlit run hotel_cancellation_app.py --server.port 8503
```

---

## Model Performance

### Confusion Matrix Analysis

Based on test set (23,878 predictions):

```
Confusion Matrix:
                    Predicted: No Cancel  |  Predicted: Cancel
Actual: No Cancel              14,158     |       3,234
Actual: Cancel                    421     |       6,065
```

**Metrics Breakdown**:

- True Negatives (TN): 14,158 - Correctly predicted no cancellation
- False Positives (FP): 3,234 - Predicted cancel but didn't cancel
- False Negatives (FN): 421 - Predicted no cancel but cancelled (most costly)
- True Positives (TP): 6,065 - Correctly predicted cancellation

**Business Value**:

```
Costs:
- FP Cost: 3,234 × $100 = $323,400
- FN Cost: 421 × $500 = $210,500
- Total Cost: $533,900

Revenue:
- TP Revenue: 6,065 × $200 = $1,213,000

Net Benefit: $1,213,000 - $533,900 = $679,100
ROI: 127.2%
```

### Comparison with Baseline

**Baseline Strategy** (always predict majority class - no cancellation):

- Would miss all 6,486 cancellations
- Loss: 6,486 × $500 = $3,243,000

**Our Model**:

- Catches 6,065 out of 6,486 cancellations (93.5%)
- Reduces loss to $210,500 (93% reduction)
- **Savings**: $3,032,500 compared to baseline

### Model Advantages

1. **High Recall (92.92%)**: Catches vast majority of cancellations
2. **Strong ROI (65.62%)**: Positive business impact
3. **Excellent Discrimination (ROC-AUC: 0.9159)**: Clear separation between classes
4. **Actionable Predictions**: Provides probability scores for risk-based interventions
5. **Interpretable**: SHAP analysis explains individual predictions

### Model Limitations

1. **Lower Precision (54.74%)**: About 45% false alarms
2. **False Positives**: May lead to unnecessary interventions
3. **Data Dependency**: Performance based on 2015-2017 Portugal hotel data
4. **External Factors**: Cannot account for unforeseen events (pandemics, economic crises)
5. **Requires Updates**: Model should be retrained periodically with new data

---

## Business Recommendations

### 1. Operational Strategies

**For High-Risk Bookings (Probability > 70%)**:

- Require deposit or pre-payment
- Send confirmation within 24 hours
- Follow up 48 hours before arrival
- Offer flexible rebooking options
- Consider 115-120% overbooking

**For Medium-Risk Bookings (Probability 40-70%)**:

- Standard confirmation process
- Monitor for booking changes
- Consider upgrade offers
- Prepare backup booking options
- Allow 108-115% overbooking

**For Low-Risk Bookings (Probability < 40%)**:

- Standard process
- Focus on service excellence
- Regular confirmation
- Allow 102-108% overbooking

### 2. Revenue Management

- Implement dynamic pricing based on cancellation risk
- Adjust overbooking policies per booking segment
- Prioritize high-confidence bookings during peak seasons
- Offer last-minute promotions for predicted cancellations

### 3. Customer Retention

- Identify high-risk customer segments early
- Provide personalized incentives (loyalty points, upgrades)
- Enhance communication with at-risk bookings
- Improve customer experience for repeat guests

### 4. Continuous Improvement

- Monitor model performance monthly
- Retrain model quarterly with new data
- Track business KPIs (revenue, cancellation rate, customer satisfaction)
- Gather feedback from hotel staff and customers
- Adjust cost assumptions based on actual business impact

---

## Technical Details

### Feature Engineering

The model uses 38 engineered features including:

**Created Features**:

- `total_nights`: Sum of weekend and weekday nights
- `total_people`: Total guests (adults + children + babies)
- `weekend_ratio`: Proportion of weekend nights
- `booking_complexity`: Score based on changes, requests, parking needs
- `customer_reliability`: Ratio of successful bookings to total history
- `lead_time_category`: Categorized booking advance time
- `season`: Season derived from arrival month
- `is_peak_season`: Binary indicator for peak months

### Model Architecture

```
Pipeline:
1. Preprocessing
   - Numerical: SimpleImputer + RobustScaler
   - Categorical (Low cardinality): OneHotEncoder
   - Categorical (High cardinality): TargetEncoder

2. Oversampling
   - RandomOverSampler
   - Handles class imbalance (73% vs 27%)

3. Classification
   - XGBoost Classifier
   - Hyperparameters optimized via RandomizedSearchCV
   - Objective: F2 Score maximization
```

### Hyperparameters

```python
Best Parameters (RandomOverSampler + XGBoost):
- n_estimators: 762
- max_depth: 9
- learning_rate: 0.0382
- colsample_bytree: 0.8784
- subsample: 0.8963
- gamma: 0.2850
- min_child_weight: 6
- reg_alpha: 0.7004
- reg_lambda: 3.5916
- scale_pos_weight: 3.6321
```

---

## Future Enhancements

### Short-term (1-3 months)

- [ ] Add real-time data pipeline for continuous learning
- [ ] Implement A/B testing framework for model versions
- [ ] Create automated reporting dashboard
- [ ] Develop mobile-friendly interface

### Medium-term (3-6 months)

- [ ] Integrate with hotel booking systems (API)
- [ ] Add multi-language support
- [ ] Implement customer segmentation models
- [ ] Develop pricing optimization module

### Long-term (6-12 months)

- [ ] Expand to multiple hotel chains
- [ ] Incorporate external data (weather, events, economic indicators)
- [ ] Build recommendation engine for upselling
- [ ] Develop forecasting models for demand planning

---

## Contributors

**Project Team**:

- Data Science Team: Model development and analysis
- Business Intelligence: Requirements and validation
- IT Department: Deployment and infrastructure
- Hotel Management: Domain expertise and feedback

**Academic Institution**: Purwadhika Digital Technology School

**Project Duration**: October 2024 - October 2025

---

## License

This project is developed for educational and business purposes. All rights reserved.

---

## Contact & Support

For questions, issues, or suggestions regarding this project:

**Technical Support**:

- Check troubleshooting section in this README
- Review notebook comments and documentation
- Verify environment setup and dependencies

**Business Inquiries**:

- Contact hotel management team
- Request access to stakeholder reports
- Schedule demo or presentation

---

## Acknowledgments

- Dataset source: Hotel booking demand datasets
- Portugal Tourism Board for context and insights
- Open-source community for libraries and tools
- Academic advisors for guidance and feedback

---

**Last Updated**: October 19, 2025

**Version**: 1.0

**Model Version**: best_hotel_cancellation_model_20251015_122216

---

## Quick Start Commands

```bash
# Clone project
cd "/path/to/Final Project New"

# Setup environment
python3.12 -m venv .venv_312
source .venv_312/bin/activate
pip install -r requirements.txt  # If requirements.txt exists

# Run notebooks (in order)
jupyter notebook 01_Data_Cleansing.ipynb
jupyter notebook 02_exploration_data_analysis_revisi.ipynb
jupyter notebook 03_preprocessing_modeling_revisi2.ipynb

# Run Streamlit apps
cd models
streamlit run hotel_cancellation_app.py
streamlit run customer_reservation_app.py --server.port 8502
```

---

**End of Documentation**
