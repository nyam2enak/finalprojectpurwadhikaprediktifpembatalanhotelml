# Hotel Booking Cancellation Predictor
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# Feature engineering functions
def calculate_engineered_features(data):
    """Calculate additional features for prediction"""
    # Total nights
    data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
    
    # Total people
    data['total_people'] = data['adults'] + data['children'].fillna(0) + data['babies']
    
    # Weekend ratio
    data['weekend_ratio'] = data['stays_in_weekend_nights'] / (data['total_nights'] + 1e-6)
    
    # Booking complexity
    complexity_score = (
        data['booking_changes'] * 2 +
        data['total_of_special_requests'] +
        data['required_car_parking_spaces'] +
        (data['children'].fillna(0) > 0).astype(int) +
        (data['babies'] > 0).astype(int)
    )
    data['booking_complexity'] = complexity_score
    
    # Customer reliability
    total_history = data['previous_cancellations'] + data['previous_bookings_not_canceled']
    data['customer_reliability'] = np.where(
        total_history > 0,
        data['previous_bookings_not_canceled'] / total_history,
        0.5  # Neutral for new customers
    )
    
    # Lead time category
    data['lead_time_category'] = pd.cut(
        data['lead_time'],
        bins=[-1, 7, 30, 90, 365, float('inf')],
        labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
    )
    
    # Season based on arrival month
    month_to_season = {
        'January': 'Winter', 'February': 'Winter', 'March': 'Spring',
        'April': 'Spring', 'May': 'Spring', 'June': 'Summer',
        'July': 'Summer', 'August': 'Summer', 'September': 'Fall',
        'October': 'Fall', 'November': 'Fall', 'December': 'Winter'
    }
    data['season'] = data['arrival_date_month'].map(month_to_season)
    
    # Peak season (summer and winter holidays)
    peak_months = ['June', 'July', 'August', 'December']
    data['is_peak_season'] = data['arrival_date_month'].isin(peak_months).astype(int)
    
    return data

def create_reservation_date(arrival_year, arrival_month, arrival_day, lead_time):
    """Create reservation date from arrival date and lead time"""
    try:
        arrival_date = datetime(arrival_year, 
                              list(range(1, 13))[['January', 'February', 'March', 'April', 'May', 'June',
                                                  'July', 'August', 'September', 'October', 'November', 'December'].index(arrival_month)], 
                              arrival_day)
        reservation_date = arrival_date - pd.Timedelta(days=lead_time)
        return reservation_date.date()
    except:
        return None

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_hotel_cancellation_model_20251015_122216.sav")
        with open("model_metadata_20251015_122216.json", "r") as f:
            metadata = json.load(f)
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)

def main():
 
    st.image("alphahotel.jpeg", width=200)
    st.title("Booking Cancellation Predictor")

    
    # Load model
    model, metadata, error = load_model()
    
    if error:
        st.error(f"Error loading model: {error}")
        st.info("Make sure the model files are in the same directory as this script")
        return
    
    # Display model info in sidebar
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Algorithm**: {metadata['algorithm']}")
        st.write(f"**Optimization**: {metadata['optimization_metric']}")
        st.write(f"**F2 Score**: {metadata['performance_metrics']['f2_score']:.4f}")
        st.write(f"**ROC-AUC**: {metadata['performance_metrics']['roc_auc']:.4f}")
        st.write(f"**Precision**: {metadata['performance_metrics']['precision']:.4f}")
        st.write(f"**Recall**: {metadata['performance_metrics']['recall']:.4f}")
        st.write(f"**Accuracy**: {metadata['performance_metrics']['accuracy']:.4f}")
        
        st.subheader("Business Impact")
        st.write(f"**Net Benefit**: ${metadata['business_metrics']['net_benefit']:,.0f}")
        st.write(f"**ROI**: {metadata['business_metrics']['roi_percentage']:.2f}%")
        
        st.info("Model optimized for F2 Score (beta=2.0) - prioritizes catching cancellations over precision")
    
    # Hotel Booking Input Form
    st.write("Enter complete booking information for accurate cancellation prediction")
    
    # Main booking information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Hotel & Dates")
        hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
        
        # Arrival date
        arrival_year = st.selectbox("Arrival Year", [2024, 2025, 2026, 2027, 2028], index=1)
        arrival_month = st.selectbox("Arrival Month", 
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
            index=9)  # October default
        arrival_day = st.number_input("Arrival Day", min_value=1, max_value=31, value=15)
        arrival_week = st.number_input("Week Number", min_value=1, max_value=53, value=42)
        
    with col2:
        st.subheader("Stay Duration")
        weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=30, value=1)
        week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=2)
        days_waiting = st.number_input("Days in Waiting List", min_value=0, max_value=100, value=0)
        
        # Calculate lead time from today
        today = date.today()
        try:
            month_num = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"].index(arrival_month) + 1
            arrival_date_obj = date(arrival_year, month_num, arrival_day)
            lead_time = max((arrival_date_obj - today).days, 0)
        except:
            lead_time = 30
        
        st.info(f"Lead Time: {lead_time} days from today")
        
    with col3:
        st.subheader("Guests")
        adults = st.number_input("Adults", min_value=1, max_value=20, value=2)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        babies = st.number_input("Babies", min_value=0, max_value=10, value=0)
        
        # Meal and room
        meal = st.selectbox("Meal Plan", ["BB", "FB", "HB", "SC", "Undefined"])
        reserved_room = st.selectbox("Reserved Room Type", 
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"])
        assigned_room = st.selectbox("Assigned Room Type",
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"])
    
    with col4:
        st.subheader("Booking Details")
        adr = st.number_input("Average Daily Rate", min_value=0.0, max_value=1000.0, value=100.0)
        parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=10, value=0)
        special_requests = st.number_input("Special Requests", min_value=0, max_value=10, value=0)
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
    
    # Customer information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Customer Profile")
        market_segment = st.selectbox("Market Segment", 
            ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"])
        customer_type = st.selectbox("Customer Type", 
            ["Transient", "Contract", "Transient-Party", "Group"])
        is_repeated = st.selectbox("Repeated Guest", ["No", "Yes"])
        
    with col2:
        st.subheader("History")
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=50, value=0)
        previous_bookings = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=50, value=0)
        
    with col3:
        st.subheader("Financial")
        deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        agent = st.number_input("Agent ID", min_value=0.0, max_value=500.0, value=0.0, 
                               help="0 if no agent")
        company = st.number_input("Company ID", min_value=0.0, max_value=500.0, value=0.0,
                                 help="0 if no company")
        
        # Common countries in hotel datasets
        countries = [
            "PRT", "GBR", "USA", "ESP", "ITA", "FRA", "DEU", "NLD", 
            "BEL", "CHE", "AUT", "IRL", "POL", "SWE", "NOR", "DNK",
            "FIN", "RUS", "BRA", "CAN", "AUS", "JPN", "CHN", "IND",
            "ARE", "SAU", "TUR", "GRC", "CZE", "HUN", "BGR", "ROU",
            "HRV", "SVN", "SVK", "EST", "LVA", "LTU", "LUX", "MLT",
            "CYP", "ISL", "AGO", "DZA", "ARG", "ARM", "AZE", "BHR"
        ]
        country = st.selectbox("Country", countries, index=0)
    
    # Prediction button
    if st.button("Predict Cancellation Risk", type="primary"):
        with st.spinner("Processing booking data and making prediction..."):
            try:
                # Create complete input dataframe
                input_data = pd.DataFrame({
                    'hotel': [hotel],
                    'lead_time': [lead_time],
                    'arrival_date_year': [arrival_year],
                    'arrival_date_month': [arrival_month],
                    'arrival_date_week_number': [arrival_week],
                    'arrival_date_day_of_month': [arrival_day],
                    'stays_in_weekend_nights': [weekend_nights],
                    'stays_in_week_nights': [week_nights],
                    'adults': [adults],
                    'children': [children if children > 0 else np.nan],
                    'babies': [babies],
                    'meal': [meal],
                    'country': [country],
                    'market_segment': [market_segment],
                    'is_repeated_guest': [1 if is_repeated == "Yes" else 0],
                    'previous_cancellations': [previous_cancellations],
                    'previous_bookings_not_canceled': [previous_bookings],
                    'reserved_room_type': [reserved_room],
                    'assigned_room_type': [assigned_room],
                    'booking_changes': [booking_changes],
                    'deposit_type': [deposit_type],
                    'agent': [agent if agent > 0 else np.nan],
                    'company': [company if company > 0 else np.nan],
                    'days_in_waiting_list': [days_waiting],
                    'customer_type': [customer_type],
                    'adr': [adr],
                    'required_car_parking_spaces': [parking_spaces],
                    'total_of_special_requests': [special_requests]
                })
                
                # Calculate engineered features
                input_data = calculate_engineered_features(input_data)
                
                # Display input data summary
                st.subheader("Booking Information Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Dates & Duration**")
                    st.write(f"Arrival: {arrival_day} {arrival_month} {arrival_year}")
                    st.write(f"Lead Time: {lead_time} days")
                    st.write(f"Total Nights: {input_data['total_nights'].iloc[0]}")
                    st.write(f"Weekend Ratio: {input_data['weekend_ratio'].iloc[0]:.2f}")
                
                with col2:
                    st.write("**Guests & Pricing**")
                    st.write(f"Total People: {int(input_data['total_people'].iloc[0])}")
                    st.write(f"Daily Rate: ${adr:.2f}")
                    st.write(f"Total Value: ${adr * input_data['total_nights'].iloc[0]:.2f}")
                    st.write(f"Season: {input_data['season'].iloc[0]}")
                
                with col3:
                    st.write("**Customer Profile**")
                    st.write(f"Lead Time Category: {input_data['lead_time_category'].iloc[0]}")
                    st.write(f"Booking Complexity: {int(input_data['booking_complexity'].iloc[0])}")
                    st.write(f"Customer Reliability: {input_data['customer_reliability'].iloc[0]:.2f}")
                    st.write(f"Peak Season: {'Yes' if input_data['is_peak_season'].iloc[0] else 'No'}")
                
                # Make actual prediction
                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0, 1]
                except Exception as pred_error:
                    st.warning(f"Prediction error: {pred_error}. Using fallback model.")
                    # Enhanced fallback prediction with realistic cancellation rates
                    probability = 0.30  # Base industry cancellation rate
                    
                    # Lead time impact (major factor)
                    if lead_time > 180:
                        probability += 0.35  # Very long advance booking
                    elif lead_time > 120:
                        probability += 0.25  # Long advance booking
                    elif lead_time > 60:
                        probability += 0.15  # Medium advance booking
                    elif lead_time > 30:
                        probability += 0.05  # Short advance booking
                    elif lead_time < 3:
                        probability += 0.20  # Last minute booking
                    
                    # ADR impact
                    if adr < 60:
                        probability += 0.15  # Very budget booking
                    elif adr < 100:
                        probability += 0.08  # Budget booking
                    elif adr > 250:
                        probability += 0.12  # Luxury booking uncertainty
                    
                    # Guest composition impact
                    total_guests = adults + children + babies
                    if total_guests > 6:
                        probability += 0.15  # Large groups
                    elif total_guests == 1:
                        probability += 0.08  # Solo travelers
                    
                    # Stay duration impact
                    total_nights = weekend_nights + week_nights
                    if total_nights > 10:
                        probability += 0.12  # Extended stays
                    elif total_nights == 1:
                        probability += 0.05  # One night stays
                    
                    # Customer history impact
                    if previous_cancellations > 0:
                        probability += min(0.25, previous_cancellations * 0.15)
                    if previous_bookings > 0:
                        probability -= min(0.10, previous_bookings * 0.02)
                    
                    # Market segment impact
                    if market_segment == "Online TA":
                        probability += 0.12
                    elif market_segment == "Groups":
                        probability += 0.08
                    elif market_segment == "Corporate":
                        probability -= 0.05
                    
                    # Deposit type impact
                    if deposit_type == "Non Refund":
                        probability -= 0.15
                    elif deposit_type == "Refundable":
                        probability -= 0.05
                    
                    # Booking changes impact
                    if booking_changes > 2:
                        probability += 0.10
                    
                    # Special requests impact (engagement)
                    if special_requests > 3:
                        probability -= 0.08
                    
                    # Room type mismatch
                    if reserved_room != assigned_room:
                        probability += 0.08
                    
                    # Peak season adjustment
                    if input_data['is_peak_season'].iloc[0]:
                        probability += 0.05
                    
                    # Cap probability between 5% and 85%
                    probability = max(0.05, min(0.85, probability))
                    prediction = 1 if probability > 0.5 else 0
                
                st.subheader("Prediction Results")
                
                # Display prediction with color coding
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if probability > 0.7:
                        st.error(f"🔴 HIGH RISK\n{probability:.1%} probability")
                    elif probability > 0.4:
                        st.warning(f"🟡 MEDIUM RISK\n{probability:.1%} probability")
                    else:
                        st.success(f"🟢 LOW RISK\n{probability:.1%} probability")
                
                with col2:
                    st.metric("Cancellation Probability", f"{probability:.1%}")
                    st.metric("Prediction", "Cancel" if prediction == 1 else "No Cancel")
                
                with col3:
                    confidence = max(probability, 1-probability)
                    st.metric("Model Confidence", f"{confidence:.1%}")
                    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
                    st.metric("Risk Level", risk_level)
                    st.info(f"F2 Score Model\n(Recall-optimized)")
                
                # Feature importance for this prediction (simplified)
                st.subheader("Key Risk Factors")
                risk_factors = []
                
                # Lead time analysis
                if lead_time > 180:
                    risk_factors.append(f"Very long lead time ({lead_time} days) - high uncertainty")
                elif lead_time > 120:
                    risk_factors.append(f"Long lead time ({lead_time} days) increases uncertainty")
                elif lead_time > 60:
                    risk_factors.append(f"Medium lead time ({lead_time} days) - moderate risk")
                elif lead_time < 3:
                    risk_factors.append(f"Very short lead time ({lead_time} days) - last minute booking")
                elif lead_time < 7:
                    risk_factors.append(f"Short lead time ({lead_time} days) - limited planning")
                
                # Customer history
                if previous_cancellations > 0:
                    risk_factors.append(f"Customer has {previous_cancellations} previous cancellations")
                
                if previous_bookings > 5:
                    risk_factors.append(f"Loyal customer with {previous_bookings} successful bookings")
                elif previous_bookings == 0 and previous_cancellations == 0:
                    risk_factors.append("New customer - no booking history")
                
                # Pricing analysis
                if adr < 60:
                    risk_factors.append(f"Very low daily rate (${adr:.2f}) - budget conscious")
                elif adr < 100:
                    risk_factors.append(f"Low daily rate (${adr:.2f}) - price sensitive segment")
                elif adr > 250:
                    risk_factors.append(f"Premium daily rate (${adr:.2f}) - luxury segment")
                elif adr > 200:
                    risk_factors.append(f"High daily rate (${adr:.2f}) - upscale booking")
                
                # Group size and composition
                total_guests = adults + children + babies
                if total_guests > 6:
                    risk_factors.append(f"Large group booking ({total_guests} guests)")
                elif total_guests == 1:
                    risk_factors.append("Solo traveler booking")
                elif children > 0:
                    risk_factors.append(f"Family booking with {children} children")
                
                # Stay duration
                total_nights = weekend_nights + week_nights
                if total_nights > 10:
                    risk_factors.append(f"Extended stay ({total_nights} nights)")
                elif total_nights == 1:
                    risk_factors.append("One night stay - short duration")
                
                # Booking complexity
                complexity = input_data['booking_complexity'].iloc[0]
                if complexity > 5:
                    risk_factors.append(f"High booking complexity score: {int(complexity)}")
                elif complexity > 3:
                    risk_factors.append(f"Moderate booking complexity score: {int(complexity)}")
                
                # Market segment analysis
                if market_segment == "Online TA":
                    risk_factors.append("Online travel agent booking - higher volatility")
                elif market_segment == "Groups":
                    risk_factors.append("Group booking - coordination challenges")
                elif market_segment == "Corporate":
                    risk_factors.append("Corporate booking - typically more stable")
                elif market_segment == "Direct":
                    risk_factors.append("Direct booking - customer engagement")
                
                # Room and service factors
                if reserved_room != assigned_room:
                    risk_factors.append(f"Room type change: {reserved_room} to {assigned_room}")
                
                if booking_changes > 2:
                    risk_factors.append(f"Multiple booking changes ({booking_changes})")
                
                if special_requests > 3:
                    risk_factors.append(f"High engagement - {special_requests} special requests")
                elif special_requests == 0:
                    risk_factors.append("No special requests - standard booking")
                
                # Seasonal factors
                if input_data['is_peak_season'].iloc[0]:
                    risk_factors.append("Peak season booking - higher demand period")
                
                # Deposit impact
                if deposit_type == "Non Refund":
                    risk_factors.append("Non-refundable deposit - higher commitment")
                elif deposit_type == "No Deposit":
                    risk_factors.append("No deposit required - lower commitment barrier")
                
                # Weekend ratio
                weekend_ratio = input_data['weekend_ratio'].iloc[0]
                if weekend_ratio > 0.6:
                    risk_factors.append("Weekend-heavy stay - leisure travel pattern")
                elif weekend_ratio == 0:
                    risk_factors.append("Weekday-only stay - business travel pattern")
                
                # Display risk factors
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("No significant risk factors identified")
                    
                    # Business Intelligence Dashboard
                    st.subheader("Business Intelligence Dashboard")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Risk Mitigation Strategies**")
                        if probability > 0.7:
                            st.error("**HIGH RISK - Immediate Actions Required:**")
                            st.write("• Send confirmation email within 24 hours")
                            st.write("• Offer flexible cancellation terms")
                            st.write("• Consider overbooking protection")
                            st.write("• Implement retention campaign")
                            st.write("• Personal follow-up recommended")
                        elif probability > 0.4:
                            st.warning("**MEDIUM RISK - Monitor Closely:**")
                            st.write("• Standard confirmation process")
                            st.write("• Monitor for booking changes")
                            st.write("• Prepare backup booking options")
                            st.write("• Consider incentive offers")
                        else:
                            st.success("**LOW RISK - Standard Process:**")
                            st.write("• Standard confirmation and follow-up")
                            st.write("• Regular service delivery")
                            st.write("• Focus on service excellence")
                    
                    with col2:
                        st.write("**Revenue Impact Analysis**")
                        total_nights = input_data['total_nights'].iloc[0]
                        total_value = adr * total_nights
                        potential_loss = total_value * probability
                        
                        st.metric("Total Booking Value", f"${total_value:.2f}")
                        st.metric("Potential Revenue at Risk", f"${potential_loss:.2f}")
                        
                        if potential_loss > 1000:
                            st.error("⚠️ High-value booking at risk!")
                        elif potential_loss > 500:
                            st.warning("💡 Monitor this booking closely")
                        else:
                            st.success("✅ Low financial risk")
                        
                        st.caption(f"Model: RandomOverSampler + XGBoost (F2={metadata['performance_metrics']['f2_score']:.4f})")
                
            except Exception as e:
                st.error(f"Error processing booking data: {str(e)}")
                st.write("Please check all input fields and try again.")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
# Model Information Sidebar
# with st.sidebar:
    # st.subheader("🤖 Model Information")
    # st.write("**Model**: ADASYN + XGBoost Ensemble")
    # st.write("**Performance Metrics**:")
    # st.write("• F0.5 Score: 73.12%")
    # st.write("• ROC-AUC: 91.54%")
    # st.write("• Precision: 77.48%")
   #  st.write("• Recall: 69.54%")
    
    # st.write("**Features**: 38 engineered features")
    # st.write("**Training Data**: 119,390 bookings")
    
    # st.subheader("Test High Risk Scenarios")
    # st.write("**Try these for HIGH cancellation risk:**")
    # st.write("• Lead time > 180 days + Low ADR (<60)")
    # st.write("• Lead time > 120 days + Large group (6+ people)")
   #  st.write("• Lead time < 3 days + Premium room (>250)")
    # st.write("• Previous cancellations > 2")
    # st.write("• Online TA + Long stay (10+ nights)")
    # st.write("• Room type mismatch + High ADR")
    
    # st.write("**Expected Risk Levels:**")
    # st.write("• Low Risk: 5-35% cancellation")
    # st.write("• Medium Risk: 35-60% cancellation")
    # st.write("• High Risk: 60-85% cancellation")
    
    # with st.expander("Model Limitations"):
    #     st.write("• Probabilistic estimates only")
    #     st.write("• External factors not included")
    #     st.write("• Performance may vary for new segments")
    
    # st.write("---")
    # st.write("**Last Updated**: October 2024")
   #  st.write("**Version**: 1.0")



if __name__ == "__main__":
    main()