import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration for a wide layout
st.set_page_config(page_title="Food Security Insights Dashboard", layout="wide")

# Use st.cache_data to cache the data loading function for performance
@st.cache_data
def load_data():
    """
    Loads the food security dataset from the uploaded CSV file.
    
    Returns:
        pd.DataFrame: The loaded and preprocessed dataframe.
    """
    try:
       
        df = pd.read_csv("food_security_dashboard_sample.csv")
        
        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
        df['Year'] = df['Date'].dt.year
        
        # Define vulnerability based on Food Consumption Score (FCS) thresholds
        # According to standard WFP/INDDEX guidelines:
        # FCS > 35: Acceptable
        # 21.5 <= FCS <= 35: Borderline
        # FCS < 21.5: Poor
        df['Vulnerability Status'] = pd.cut(
            df['FCS'], 
            bins=[0, 21.5, 35, float('inf')],
            labels=['Poor', 'Borderline', 'Acceptable'],
            right=False
        )
        
        return df
    except FileNotFoundError:
        st.error("Error: The file 'food_security_dashboard_sample.csv' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# Load the data
df = load_data()

# --- Dashboard Title and Description ---
st.title("🌾 Food Security Insights Dashboard")
st.markdown("""
This interactive dashboard provides key insights into food security across different regions and time periods. 
Use the filters on the left to explore trends in crop yields, food prices, and household vulnerability.
""")

# --- Sidebar for Filters ---
st.sidebar.header("Filter Data")

# Define filter options
all_regions = ['All'] + sorted(df['Region'].unique().tolist())
all_crops = ['All'] + sorted(df['Crop'].unique().tolist())
all_years = ['All'] + sorted(df['Year'].unique().tolist())
all_vulnerability = ['All'] + sorted(df['Vulnerability Status'].unique().tolist())

# Create the select boxes
selected_region = st.sidebar.selectbox("Select a Region:", all_regions)
selected_crop = st.sidebar.selectbox("Select a Crop:", all_crops)
selected_year = st.sidebar.selectbox("Select a Year:", all_years)
selected_vulnerability = st.sidebar.multiselect("Select Vulnerability Status:", all_vulnerability, default='All')

# Filter the dataframe based on user selections
filtered_df = df.copy()

if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

if selected_crop != 'All':
    filtered_df = filtered_df[filtered_df['Crop'] == selected_crop]

if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]

if 'All' not in selected_vulnerability:
    filtered_df = filtered_df[filtered_df['Vulnerability Status'].isin(selected_vulnerability)]

# Check for empty dataframe after filtering
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
else:
    # --- Key Metrics (KPIs) ---
    st.header("Key Performance Indicators (KPIs)")

    # Calculate KPIs on the filtered data
    avg_yield = filtered_df['Yield (kg/ha)'].mean()
    avg_price = filtered_df['Price (USD/kg)'].mean()
    poor_fcs_count = (filtered_df['Vulnerability Status'] == 'Poor').sum()
    total_households = filtered_df['Household ID'].nunique()
    poor_fcs_percentage = (poor_fcs_count / total_households) * 100 if total_households > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Average Crop Yield",
            value=f"{avg_yield:,.0f} kg/ha"
        )

    with col2:
        st.metric(
            label="Average Food Price",
            value=f"${avg_price:,.2f} / kg"
        )

    with col3:
        st.metric(
            label="Households with Poor FCS",
            value=f"{poor_fcs_count} of {total_households}",
            delta=f"{poor_fcs_percentage:.1f}%"
        )

    st.markdown("---")

    # --- Charts for Detailed Insights ---
    st.header("Detailed Insights")

    # Create two columns for the charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Crop Yields Over Time")
        # Group data by date and crop to show trends
        yield_data = filtered_df.groupby(['Date', 'Crop'])['Yield (kg/ha)'].mean().reset_index()
        fig_yield = px.line(
            yield_data,
            x='Date',
            y='Yield (kg/ha)',
            color='Crop',
            title='Average Crop Yield (kg/ha) Over Time',
            labels={'Yield (kg/ha)': 'Average Yield (kg/ha)', 'Date': 'Date'},
            template="plotly_white"
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    with chart_col2:
        st.subheader("Food Prices by Commodity")
        # Group data by commodity to show price variations
        price_data = filtered_df.groupby('Commodity')['Price (USD/kg)'].mean().reset_index()
        fig_price = px.bar(
            price_data,
            x='Commodity',
            y='Price (USD/kg)',
            title='Average Price by Commodity',
            labels={'Price (USD/kg)': 'Average Price (USD/kg)', 'Commodity': 'Commodity'},
            template="plotly_white"
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # Vulnerability and Aid Access
    st.subheader("Household Vulnerability and Aid Access")

    vulnerability_col1, vulnerability_col2 = st.columns(2)

    with vulnerability_col1:
        # Plot vulnerability status distribution
        fcs_counts = filtered_df['Vulnerability Status'].value_counts().reset_index()
        fcs_counts.columns = ['Status', 'Count']
        fig_fcs = px.pie(
            fcs_counts,
            values='Count',
            names='Status',
            title='Distribution of Household Vulnerability Status',
            color='Status',
            color_discrete_map={'Poor': 'darkred', 'Borderline': 'orange', 'Acceptable': 'green'}
        )
        st.plotly_chart(fig_fcs, use_container_width=True)

    with vulnerability_col2:
        # Plot vulnerability by aid access
        aid_access_data = filtered_df.groupby(['Vulnerability Status', 'Aid Access']).size().reset_index(name='Count')
        fig_aid = px.bar(
            aid_access_data,
            x='Vulnerability Status',
            y='Count',
            color='Aid Access',
            title='Vulnerability Status by Aid Access',
            labels={'Count': 'Number of Households', 'Vulnerability Status': 'Vulnerability Status'},
            barmode='group',
            template="plotly_white"
        )
        st.plotly_chart(fig_aid, use_container_width=True)

    # Optional: Display the raw data in an expander for validation
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)
