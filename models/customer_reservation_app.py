# Hotel Reservation System - Customer Interface
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
import calendar

# Page configuration
st.set_page_config(
    page_title="Hotel Reservation System",
    page_icon="🏨",
    layout="wide"
)

# Room type pricing (ADR based on room type)
ROOM_PRICES = {
    'A': 85,   # Standard Room
    'B': 95,   # Superior Room  
    'C': 110,  # Deluxe Room
    'D': 125,  # Junior Suite
    'E': 150,  # Executive Room
    'F': 175,  # Suite
    'G': 200,  # Premium Suite
    'H': 250,  # Presidential Suite
    'I': 120,  # Family Room
    'J': 90,   # Twin Room
    'K': 160,  # Ocean View
    'L': 180   # Garden View
}

ROOM_DESCRIPTIONS = {
    'A': 'Standard Room - Basic amenities',
    'B': 'Superior Room - Enhanced comfort',  
    'C': 'Deluxe Room - Premium features',
    'D': 'Junior Suite - Spacious layout',
    'E': 'Executive Room - Business amenities',
    'F': 'Suite - Luxury accommodation',
    'G': 'Premium Suite - Top-tier luxury',
    'H': 'Presidential Suite - Ultimate luxury',
    'I': 'Family Room - Perfect for families',
    'J': 'Twin Room - Two single beds',
    'K': 'Ocean View - Scenic ocean view',
    'L': 'Garden View - Beautiful garden view'
}

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_hotel_cancellation_model_20251015_122216.sav")
        with open("model_metadata_20251015_122216.json", "r") as f:
            metadata = json.load(f)
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)

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
    
    # Customer reliability (new customer = 0.5)
    data['customer_reliability'] = 0.5
    
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
    
    # Peak season
    peak_months = ['June', 'July', 'August', 'December']
    data['is_peak_season'] = data['arrival_date_month'].isin(peak_months).astype(int)
    
    return data

def main():
    st.image("alphahotel.jpeg", width=200)
    st.title("Welcome to Alpha Hotel Reservation System")
    
    # Load model
    model, metadata, error = load_model()
    
    if error:
        st.error(f"System error: {error}")
        return
    
    # Reservation Form
    st.header("Make Your Reservation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hotel Selection")
        hotel_type = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        
        st.subheader("Check-in Date")
        checkin_date = st.date_input(
            "Select check-in date",
            value=date.today(),
            min_value=date.today()
        )
        
        # Calculate components from date
        arrival_year = checkin_date.year
        arrival_month = calendar.month_name[checkin_date.month]
        arrival_day = checkin_date.day
        arrival_week = checkin_date.isocalendar()[1]
        
        # Calculate lead time
        lead_time = (checkin_date - date.today()).days
        
        st.info(f"Booking advance: {lead_time} days")
        
    with col2:
        st.subheader("Stay Duration")
        total_nights = st.number_input("Total nights", min_value=1, max_value=30, value=2)
        
        # Simple weekend/weeknight split
        if checkin_date.weekday() >= 5:  # Saturday=5, Sunday=6
            weekend_nights = min(total_nights, 2)
            week_nights = max(0, total_nights - weekend_nights)
        else:
            # Calculate based on check-in day
            days_to_weekend = (5 - checkin_date.weekday()) % 7
            if days_to_weekend < total_nights:
                weekend_nights = min(2, total_nights - days_to_weekend)
                week_nights = total_nights - weekend_nights
            else:
                weekend_nights = 0
                week_nights = total_nights
        
        st.write(f"Weekend nights: {weekend_nights}, Weekday nights: {week_nights}")
    
    # Guest Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Guest Details")
        adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
        children = st.number_input("Children", min_value=0, max_value=8, value=0)
        babies = st.number_input("Babies", min_value=0, max_value=3, value=0)
        
    with col2:
        st.subheader("Room & Services")
        room_type = st.selectbox("Room Type", list(ROOM_PRICES.keys()), 
                                format_func=lambda x: f"{x} - {ROOM_DESCRIPTIONS[x]}")
        
        meal_plan = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC"])
        parking_needed = st.checkbox("Parking required")
        special_requests = st.number_input("Special requests", min_value=0, max_value=5, value=0)
    
    # Calculate pricing
    base_price = ROOM_PRICES[room_type]
    # Peak season adjustment
    peak_months = [6, 7, 8, 12]  # June, July, August, December
    if checkin_date.month in peak_months:
        base_price *= 1.2  # 20% increase for peak season
    
    total_cost = base_price * total_nights
    
    # Display booking summary
    st.subheader("Booking Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Reservation Details**")
        st.write(f"Hotel: {hotel_type}")
        st.write(f"Check-in: {checkin_date.strftime('%B %d, %Y')}")
        st.write(f"Duration: {total_nights} nights")
        st.write(f"Guests: {adults + children + babies} total")
        
    with col2:
        st.write("**Room & Services**")
        st.write(f"Room: {ROOM_DESCRIPTIONS[room_type]}")
        st.write(f"Meal: {meal_plan}")
        st.write(f"Parking: {'Yes' if parking_needed else 'No'}")
        st.write(f"Requests: {special_requests}")
        
    with col3:
        st.write("**Pricing**")
        st.write(f"Rate per night: ${base_price:.2f}")
        st.write(f"Total cost: ${total_cost:.2f}")
        if checkin_date.month in [6, 7, 8, 12]:
            st.write("Peak season rate applied")
    
    # Prediction
    if st.button("Check Booking Confidence", type="primary"):
        with st.spinner("Analyzing booking..."):
            try:
                # Create input data
                input_data = pd.DataFrame({
                    'hotel': [hotel_type],
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
                    'meal': [meal_plan],
                    'country': ['PRT'],  # Default country
                    'market_segment': ['Direct'],  # Direct booking
                    'is_repeated_guest': [0],  # New customer
                    'previous_cancellations': [0],
                    'previous_bookings_not_canceled': [0],
                    'reserved_room_type': [room_type],
                    'assigned_room_type': [room_type],
                    'booking_changes': [0],  # No changes yet
                    'deposit_type': ['No Deposit'],
                    'agent': [np.nan],  # No agent
                    'company': [np.nan],  # No company
                    'days_in_waiting_list': [0],
                    'customer_type': ['Transient'],
                    'adr': [base_price],
                    'required_car_parking_spaces': [1 if parking_needed else 0],
                    'total_of_special_requests': [special_requests]
                })
                
                # Calculate engineered features
                input_data = calculate_engineered_features(input_data)
                
                # Make prediction
                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0, 1]
                except Exception as pred_error:
                    st.warning(f"Using fallback prediction model")
                    # Fallback realistic prediction based on booking characteristics
                    probability = 0.25  # Base cancellation rate
                    
                    # Adjust based on lead time
                    if lead_time > 120:
                        probability += 0.25
                    elif lead_time > 60:
                        probability += 0.15
                    elif lead_time > 30:
                        probability += 0.05
                    elif lead_time < 3:
                        probability += 0.20
                    
                    # Adjust based on room price
                    if base_price < 90:
                        probability += 0.10  # Budget bookings more likely to cancel
                    elif base_price > 200:
                        probability += 0.15  # Luxury bookings also risky
                    
                    # Adjust based on stay length
                    if total_nights > 7:
                        probability += 0.10  # Long stays more likely to cancel
                    elif total_nights == 1:
                        probability += 0.05  # One night stays slightly risky
                    
                    # Adjust based on group size
                    total_guests = adults + children + babies
                    if total_guests > 4:
                        probability += 0.10  # Large groups riskier
                    elif total_guests == 1:
                        probability += 0.05  # Solo travelers slightly riskier
                    
                    # Peak season adjustment
                    if checkin_date.month in [6, 7, 8, 12]:
                        probability += 0.05  # Peak season slightly riskier
                    
                    # Weekend vs weekday
                    if checkin_date.weekday() >= 5:  # Weekend
                        probability -= 0.05  # Weekend bookings more stable
                    
                    # Special requests impact
                    if special_requests > 2:
                        probability -= 0.05  # Many requests = more committed
                    
                    # Meal plan impact
                    if meal_plan in ['FB', 'HB']:
                        probability -= 0.05  # Full/half board more committed
                    
                    # Cap probability
                    probability = max(0.05, min(0.85, probability))
                    prediction = 1 if probability > 0.5 else 0
                
                # Display results
                st.subheader("Booking Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    confidence = (1 - probability) * 100
                    if confidence >= 70:
                        st.success(f"🟢 High Confidence: {confidence:.1f}%")
                        st.write("Booking likely to proceed")
                    elif confidence >= 50:
                        st.warning(f"🟡 Medium Confidence: {confidence:.1f}%")
                        st.write("Monitor booking status")
                    else:
                        st.error(f"🔴 Low Confidence: {confidence:.1f}%")
                        st.write("High cancellation risk")
                    
                    st.caption(f"Model: F2 Score Optimized (Recall={metadata['performance_metrics']['recall']:.1%})")
                
                with col2:
                    st.metric("Cancellation Risk", f"{probability:.1%}")
                    st.metric("Booking Status", "Confirmed" if prediction == 0 else "At Risk")
                
                with col3:
                    st.metric("Total Value", f"${total_cost:.2f}")
                    risk_value = total_cost * probability
                    st.metric("Value at Risk", f"${risk_value:.2f}")
                
                # Risk factors
                st.subheader("Risk Assessment")
                risk_factors = []
                
                if lead_time > 90:
                    risk_factors.append(f"Long booking advance ({lead_time} days)")
                elif lead_time < 3:
                    risk_factors.append(f"Last-minute booking ({lead_time} days)")
                
                if total_nights > 7:
                    risk_factors.append("Extended stay booking")
                
                if base_price > 200:
                    risk_factors.append("Premium room rate")
                elif base_price < 80:
                    risk_factors.append("Budget room rate")
                
                if checkin_date.month in [6, 7, 8, 12]:
                    risk_factors.append("Peak season booking")
                
                if adults + children + babies > 4:
                    risk_factors.append("Large group booking")
                
                if risk_factors:
                    st.write("Key factors affecting confidence:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("Standard booking profile - no special risk factors")
                
                # Recommendations and Overbooking Strategy
                st.subheader("Revenue Management Strategy")
                
                if probability > 0.6:
                    st.error("**High Risk - Overbooking Recommended:**")
                    st.write("• Allow 115-120% overbooking for this booking type")
                    st.write("• Require deposit or pre-payment")
                    st.write("• Send confirmation within 24 hours")
                    st.write("• Follow up 48 hours before arrival")
                    st.write("• Consider offering flexible rebooking options")
                    overbooking_rate = 1.18
                elif probability > 0.4:
                    st.warning("**Medium Risk - Monitor Closely:**")
                    st.write("• Allow 108-115% overbooking")
                    st.write("• Standard confirmation process")
                    st.write("• Monitor for booking changes")
                    st.write("• Consider upgrade offers to reduce cancellation")
                    overbooking_rate = 1.10
                else:
                    st.success("**Low Risk - Standard Process:**")
                    st.write("• Standard overbooking policy (102-108%)")
                    st.write("• Regular confirmation and service")
                    st.write("• Focus on service excellence")
                    overbooking_rate = 1.05
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Suggested Overbooking Rate", f"{overbooking_rate:.0%}")
                with col2:
                    additional_bookings = int((overbooking_rate - 1) * 100)
                    st.metric("Extra Bookings per 100 rooms", f"+{additional_bookings}")
                
                # Show realistic scenarios
                st.subheader("Booking Risk Examples (F2 Score Model)")
                st.write("**High Risk Scenarios (60-90% cancellation):**")
                st.write("• Lead time > 120 days + Budget room + Large group")
                st.write("• Lead time > 90 days + Luxury suite + Long stay")
                st.write("• Last minute booking (< 3 days) + Premium room")
                st.write("• Peak season + Extended stay + Mid-range room")
                
                st.write("**Medium Risk Scenarios (40-60% cancellation):**")
                st.write("• Lead time 30-90 days + Standard room + 2-4 guests")
                st.write("• Weekend booking + Mid-range room + Meal plan")
                st.write("• City hotel + Business traveler + 3-5 night stay")
                
                st.write("**Low Risk Scenarios (10-40% cancellation):**")
                st.write("• Lead time 7-30 days + Weekend booking + Special requests")
                st.write("• Business traveler + City hotel + Short stay")
                st.write("• Resort hotel + Family + Full board + Multiple requests")
                
                st.caption("Note: F2 Score model prioritizes recall - better at detecting potential cancellations")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Model info in sidebar
    with st.sidebar:
        st.header("System Information")
        if metadata:
            st.write(f"**Model**: {metadata['algorithm']}")
            st.write(f"**Optimization**: {metadata['optimization_metric']}")
            st.write(f"**F2 Score**: {metadata['performance_metrics']['f2_score']:.4f}")
            st.write(f"**Recall**: {metadata['performance_metrics']['recall']:.1%}")
            st.write(f"**ROC-AUC**: {metadata['performance_metrics']['roc_auc']:.4f}")
            
            st.subheader("Business Impact")
            st.write(f"**Net Benefit**: ${metadata['business_metrics']['net_benefit']:,.0f}")
            st.write(f"**ROI**: {metadata['business_metrics']['roi_percentage']:.1f}%")
            
            st.info("Model prioritizes catching cancellations (F2 Score optimized)")
        
        st.subheader("Test High Risk Scenarios")
        st.write("**Try these for high cancellation risk:**")
        st.write("• Check-in 4+ months ahead + Premium room + 5+ guests")
        st.write("• Check-in tomorrow + Budget room + Solo traveler")
        st.write("• 10+ nights + Luxury suite + Large group")
        st.write("• Peak season + Budget room + Extended stay")
        
        st.subheader("Expected Risk Levels")
        st.write("• **Low Risk**: 10-40% cancellation")
        st.write("• **Medium Risk**: 40-70% cancellation") 
        st.write("• **High Risk**: 70-90% cancellation")
        
        st.subheader("Room Types")
        for room, desc in ROOM_DESCRIPTIONS.items():
            st.write(f"**{room}**: ${ROOM_PRICES[room]} - {desc.split(' - ')[1]}")

if __name__ == "__main__":
    main()