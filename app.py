import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

def load_model():
    """Load the model with deployment-safe path handling"""
    # Try multiple possible paths
    possible_paths = [
        'bike_duration_model_complete.pkl',  # Same directory
        'Project1/bike_duration_model_complete.pkl',  # Your local structure
        './bike_duration_model_complete.pkl',  # Current directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    
    raise FileNotFoundError("Model file not found in any expected location")

#setting page
st.set_page_config(
    page_title= "NYC Bike Share Analytics",
    page_icon= "🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Loading data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data(file_names):
    """
    Load multiple CSV files and combine them into single DF
    """
    all_data = []
    messages = []

    for file_name in file_names:
        file_path = os.path.join(SCRIPT_DIR, "Data", file_name)

        if not os.path.exists(file_path):
            messages.append(f"File not found: {file_name}")
            continue
        try:
            data = pd.read_csv(file_path)
            data['source_file'] = file_name
            all_data.append(data)
            messages.append(f'Successfully loaded {file_name}')
        except Exception as e:
            messages.append(f"Error loading {file_name}:{e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df, messages, "success"
    else:
        return None, messages, 'error'

def preprocess_data(_df):
    df = _df.copy()

    if 'started_at' in df.columns:
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['date'] = df['started_at'].dt.date
    
    if 'ended_at' in df.columns:
        df['ended_at'] = pd.to_datetime(df['ended_at'])
    
    #trip_duration
    if all(col in df.columns for col in ['started_at', 'ended_at']):
        df['trip_duration_sec'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
        df = df[(df['trip_duration_sec'] >= 60) & (df['trip_duration_sec'] <= 24 * 3600)]
    return df

# --- Load multiple files ---
if 'df' not in st.session_state:
    file_names = [
        'citibike_trip_202301.csv',
        'citibike_trip_202302.csv', 
        'citibike_trip_202303.csv',
        'citibike_trip_202304.csv'
    ]
    st.session_state.df, st.session_state.load_messages, st.session_state.message_type = load_data(file_names)

st.sidebar.title("🚴 NYC Bike Share")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate the App",
    ["🏠 Home", "📊 Data Overview","📈 Interactive Dashboard", "👥 Rider Analysis", "🤖 ML Predictor"]
)

#--HomePage--
if page == '🏠 Home':
    st.title("NYC Bike Share Data Explorer")
    st.markdown("---")
    st.subheader("Welcome to my Data Analytics Project!")
    st.write("""
    This web app is built with **StreamLit** and analyzes public data from New York City's Bike Share Program.
    The objective of this web app is:
    - **Wrangle** and **clean** large datasets
    - Create **interactive visualizations** and dashboards
    - Build **data products** and tell data-driven stories
    - Deploy machine learning models        
    """)    
    st.success("👈 **Use the sidebar on the left to navigate through the different sections of the analysis.**")
        #Gonna put image here later
    if st.session_state.df is not None:
        st.write(f"The dataset currently loaded contains **{len(st.session_state.df)} rows** and **{len(st.session_state.df.columns)} columns**.")
    else:
        st.warning("Data not loaded yet")


#-- Data Overview --
elif page == "📊 Data Overview":
    st.title("Data Overview & Preprocessing")
    st.markdown("---")

    if st.session_state.message_type == 'error':
        st.error(st.session_state.load_messages)
        st.info("Please make sure:")
        st.info("1. You've created a 'Data' folder in the same directory as app.py")
        st.info("2. The file is in the Data folder")
        st.info("3. The filename matches exactly: 'citibike_trip_202301.csv'")
    elif st.session_state.message_type == "success":
        if st.session_state.df is not None:
            st.success(st.session_state.load_messages)
            st.write(f"Data loaded successfully! Shape: {st.session_state.df.shape}")
            st.dataframe(st.session_state.df.head())
        else:
            st.warning("Data not loaded yet")

    with st.expander("🔍 Show More Data"):
        if st.session_state.df is not None:
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(st.session_state.df.head(num_rows))
        
    with st.expander("Data Description"):
        st.markdown("""
            | Column Name | Description |
            |-------------|-------------|
            | `ride_id` | Unique ride identifier |
            | `rideable_type` | Type of bike ridden by users(electric/classic) |
            | `started_at` | precise date and time when ride began |
            | `ended_at` | precise date and time when ride concluded |
            | `start_station_name` | The name of the bike's starting location |
            | `start_station_id` | A unique identifier for starting location |
            | `end_station_name` | The name of the bike's ending location |
            | `end_station_id` | A unique identifier for starting location |
            | `member_casual` |  Differentiates between types of users, such as members and casual riders |
            | `start_lat` | The latitude coordinate of the ride's starting point |
            | `start_lng` | The longitude coordinate of the ride's starting point |
            | `end_lat` | The latitude coordinate of the ride's ending point |
            | `end_lng` | The longitude coordinate of the ride's ending point |
                    
        """)

#--- Interactive Dashboard ---
elif page=="📈 Interactive Dashboard":
    st.title("📈 Interactive Dashboard")
    st.markdown("---")
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        #Pre-process datetime
        if 'started_at' in df.columns:
            df['started_at'] = pd.to_datetime(df['started_at'])
            df['date'] = df['started_at'].dt.date

        if 'ended_at' in df.columns:
            df['ended_at'] = pd.to_datetime(df['ended_at'])

        if all(col in df.columns for col in ['started_at', 'ended_at']):
            df['trip_duration_sec'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
            #filtering less than 60 s or more than 24 hours
            df = df[(df['trip_duration_sec'] >= 60) & (df['trip_duration_sec'] <= 24 * 3600)]

        #Sidebar filter
        st.sidebar.header("🔧 Filters")

        #Time range finders
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()

            date_range = st.sidebar.date_input(
                "Data Range",
                value = (min_date, max_date),
                min_value = min_date,
                max_value = max_date
            )

            if len(date_range) == 2:  
                start_date, end_date = date_range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if 'member_casual' in df.columns:
            user_types = df['member_casual'].unique()
            selected_type = st.sidebar.multiselect(
                "User Types",
                options=user_types,
                default=user_types
            )
            df = df[df['member_casual'].isin(selected_type)]
        
        if 'rideable_type' in df.columns:
            bike_types = df['rideable_type'].unique()
            selected_type = st.sidebar.multiselect(
                "Bike Types",
                options=bike_types,
                default=bike_types
            )
            df = df[df['rideable_type'].isin(selected_type)]

        if 'trip_duration_sec' in df.columns:
            min_duration_min = 1 #at least 1 minute
            max_duration_min = int(df['trip_duration_sec'].max()/60)

            min_duration, max_duration= st.sidebar.slider(
                "Trip Duration (minutes)",
                min_value=min_duration_min,
                max_value=min(240, max_duration_min), #Capping at 4 hours for usability
                value=(2,60) #default range 2-60 min
            )

            #convert back to second for filtering
            df = df[(df['trip_duration_sec'] >= min_duration * 60) &
                    (df['trip_duration_sec'] <= max_duration * 60)]
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Rides", f"{len(df):,}")
        
        with col2:
            if 'trip_duration_sec' in df.columns:
                avg_duration_min = df['trip_duration_sec'].mean()/60
                st.metric("Avg Duration", f"{avg_duration_min:.1f} min")
            else:
                st.metric("Avg Duration", "N/A")
        
        with col3:
            if 'member_casual' in df.columns:
                member_pct = (len(df[df['member_casual'] == 'member']) / len(df)) * 100
                st.metric("Members", f"{member_pct:.1f}%")
            else:
                st.metric("Members", "N/A")
        
        with col4:
            if 'rideable_type' in df.columns:
                electric_pct = (len(df[df['rideable_type'] == 'electric_bike']) / len(df)) * 100
                st.metric("Electric Bikes", f"{electric_pct:.1f}%")
            else:
                st.metric("Electric Bikes", "N/A")
        
        # --- Visualizations ---
        tab1, tab2, tab3 = st.tabs(["⌚ Time Analysis", "📍 Station Analysis", "👥 User Analysis", ])

        with tab1:
            st.subheader("Rides Over Time")

            #Daily rides chart
            if 'date' in df.columns:
                daily_rides = df.groupby('date').size()
                fig_daily = px.line(x=daily_rides.index, y = daily_rides.values,
                                    labels={'x':'Date', 'y': 'Number of Rides'},
                                    title='Daily Rides Trend')
                fig_daily.update_layout(xaxis_title='Date', yaxis_title='Number of Rides')
                st.plotly_chart(fig_daily, use_container_width=True)

            #Hourly Pattern
            if 'started_at' in df.columns:
                df['hour'] = df['started_at'].dt.hour
                hourly_rides = df.groupby('hour').size()
                fig_hourly = px.bar(x=hourly_rides.index, y=hourly_rides.values,
                                    labels={'x': 'Hour of Day', 'y': 'Number of Rides'},
                                    title = "Rides by Hour of the Day")
                fig_hourly.update_layout(xaxis_title = 'Hour of Day', yaxis_title = 'Number of Rides')
                st.plotly_chart(fig_hourly, use_container_width=True)

        with tab2:
            st.subheader("Station Analysis")
            #Top start stations
            if 'start_station_name' in df.columns:
                top_start_station = df['start_station_name'].value_counts().head(10)
                fig_start = px.bar(x=top_start_station, y=top_start_station.index,
                                   orientation = 'h', 
                                   labels = {'x': 'Number of Rides', 'y':'Station Names'},
                                   title = 'Top 10 Start Stations')
                fig_start.update_layout(xaxis_title = "Number of Rides", yaxis_title = "Station Name")
                st.plotly_chart(fig_start, use_container_width=True)
            
            #Station Map
            if all(col in df.columns for col in ['start_lat', 'start_lng']):
                station_coords = df[['start_station_name','start_lat', 'start_lng']].drop_duplicates()
                st.map(station_coords.rename(columns={
                    'start_lat':'lat',
                    'start_lng':'lon'
                }))

        with tab3:
            st.subheader("User Behavior Analysis")

            if 'member_casual' in df.columns:
                user_dist = df['member_casual'].value_counts()
                fig_users = px.pie(values=user_dist.values, names = user_dist.index,
                                   title = "Member vs Casual Riders")
                st.plotly_chart(fig_users, use_container_width=True)

            if 'rideable_type' in df.columns:
                bike_dist = df['rideable_type'].value_counts()
                fig_bikes = px.pie(values=bike_dist.values, names=bike_dist.index,
                                   title = 'Bike Type Distribution')
                st.plotly_chart(fig_bikes, use_container_width=True)
            
            if 'trip_duration_sec' in df.columns:
                #convert to minute for better readabiliy
                df['trip_duration_min'] = df['trip_duration_sec']/60
                #filtering extremely long trip
                duration_filtered = df[df['trip_duration_min']<= 120] #2h max
                fig_duration = px.histogram(duration_filtered, x='trip_duration_min', nbins=50,
                                           title = 'Trip Duration Distribution(minutes)',
                                           labels ={'trip_duration_min': 'Duration (minutes)'})
                st.plotly_chart(fig_duration, use_container_width=True)

        st.sidebar.info(f"Showing {len(df):,} rides after filtering")

        
       
elif page == "👥 Rider Analysis":
    st.title("👥 Rider Behaviour Analysis")
    st.markdown('---')
    if st.session_state.df is not None:
        df = preprocess_data(st.session_state.df)
        
        #sampling data for performance
        if len(df) > 50000:
            sample_df = df.sample(50000, random_state= 42)
            st.info("Using sampled data for better performance")
        else:
            sample_df = df

        st.subheader("Member vs Casual Rider Comparison")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if 'member_casual' in sample_df.columns:
                member_count = len(sample_df[sample_df['member_casual'] == 'member'])
                casual_count = len(sample_df[sample_df['member_casual'] == 'casual'])
                st.metric("Members", f"{member_count:,}")
                st.metric("Casual Riders", f"{casual_count:,}")
        
        with col2:
            if all(col in sample_df.columns for col in ['member_casual', 'trip_duration_sec']):
                member_avg = sample_df[sample_df['member_casual'] == 'member']['trip_duration_sec'].mean()/60
                casual_avg = sample_df[sample_df['member_casual'] == 'casual']['trip_duration_sec'].mean()/60
                st.metric( "Avg Duration (Members)", f"{member_avg:.1f} min")
                st.metric("Avg Duration (Casuals)", f"{casual_avg:.1f} min")

        with col3:
            if 'rideable_type' in sample_df.columns:
                #recalculating member and casual_count to avoid unbound alegation
                member_count = len(sample_df['member_casual'] == 'member')
                casual_count = len(sample_df['member_casual'] == 'casual')

                member_elec = len(sample_df[(sample_df['member_casual'] == 'member') & (sample_df['rideable_type'] == 'electric_bike')])
                casual_elec = len(sample_df[(sample_df['member_casual'] == 'casual') & (sample_df['rideable_type']=='electric_bike')])

                pct_member_elec = (member_elec/member_count * 100) if member_count > 0 else 0
                pct_casual_elec = (casual_elec/casual_count * 100) if casual_count > 0 else 0
                st.metric("Electric bike user(Members)", f"{pct_member_elec:.1f}%")
                st.metric("Electric bike user (Casual)", f"{pct_casual_elec:.1f}%")
        
        with col4:
            if 'rideable_type' in sample_df.columns:
                #just gonna copy and paste the above code and change the ridable type to conventional bike
                member_count = len(sample_df['member_casual'] == 'member')
                casual_count = len(sample_df['member_casual'] == 'casual')

                member_clasc = len(sample_df[(sample_df['member_casual'] == 'member') & (sample_df['rideable_type'] == 'classic_bike')])
                casual_clasc = len(sample_df[(sample_df['member_casual'] == 'casual') & (sample_df['rideable_type']=='classic_bike')])

                pct_member_clasc = (member_clasc/member_count * 100) if member_count > 0 else 0
                pct_casual_clasc = (casual_clasc/casual_count * 100) if casual_count > 0 else 0
                st.metric("Casual bike user(Members)", f"{pct_member_clasc:.1f}%")
                st.metric("Casual bike user (Casual)", f"{pct_casual_clasc:.1f}%")
        with col5:
            if 'started_at' in sample_df.columns:
                sample_df['hour'] = sample_df['started_at'].dt.hour
                member_peak = sample_df[sample_df['member_casual']== 'member']['hour'].mode()[0]
                casual_peak = sample_df[sample_df['member_casual'] == 'casual']['hour'].mode()[0]
                st.metric("Peak Hour (Members)", f"{member_peak:02d}:00")
                st.metric("Peak hour (Casual)", f"{casual_peak:02d}:00")
        
        #visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Usage Pattern by Hour")
            if all (col in sample_df.columns for col in ['member_casual', 'started_at']):
                sample_df['hour'] = sample_df['started_at'].dt.hour
                hourly_usage = sample_df.groupby(['hour', 'member_casual']).size().reset_index(name='count')
                fig_hourly = px.line(
                    hourly_usage,
                    x = 'hour',
                    y = 'count',
                    color='member_casual',
                    title='Rides by Hour and User Type',
                    labels= {'hour': 'Hour of Day', 'count':'Number of Rides'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
        with col2:
            st.subheader("Usage Pattern by Day of Week")
            if 'started_at' in sample_df.columns:
                sample_df['day_of_week'] = sample_df['started_at'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_usage = sample_df.groupby(['day_of_week', 'member_casual']).size().reset_index(name='count')
                daily_usage['day_of_week'] = pd.Categorical(daily_usage['day_of_week'], categories=day_order, ordered=True)
                fig_daily = px.bar(
                    daily_usage,
                    x = 'day_of_week',
                    y = 'count',
                    color = 'member_casual',
                    barmode='group',
                    title='Rides by Day and User by Type',
                    labels={'day_of_week': 'Day of Week', 'count': 'Number of Rides'}
                )
                st.plotly_chart(fig_daily, use_container_width=True)
        
        st.subheader("Bike Type Preference")
        if all(col in sample_df.columns for col in ['member_casual', 'rideable_type']):
            bike_preference =sample_df.groupby(['member_casual', 'rideable_type']).size().reset_index(name='count')
            fig_bike = px.bar(
                bike_preference,
                x = 'member_casual',
                y = 'count',
                color='rideable_type',
                barmode='group',
                title='Bike Type Preference by User Type',
                labels={'member_casual':'User Type', 'count': 'Number of Rides'}
            )
            st.plotly_chart(fig_bike, use_container_width=True)
        
        st.subheader("Trip Duration Distribution")
        if all(col in sample_df.columns for col in ['member_casual', 'trip_duration_sec']):
            duration_df = sample_df[sample_df['trip_duration_sec'] <= 3600]
            fig_duration = px.histogram(
                duration_df,
                x='trip_duration_sec',
                color='member_casual',
                nbins = 30,
                title='Trip Duration Distribution by User Type',
                labels={'trip_duration_sec': 'Duration (seconds)'},
                opacity= 0.7
            )
            st.plotly_chart(fig_duration, use_container_width=True)
    else:
        st.warning("Please load data first to view rider analysis")

elif page == "🤖 ML Predictor":
    import random
    st.title("Trip Duration Predictor")
    st.markdown("---")

    if st.session_state.df is not None:
        st.info("""
             This machine learning model predicts trip duration based on ride characteristics. (Hopefully)
        """)

        #Sample data for training
        ml_df = preprocess_data(st.session_state.df)
        ml_sample = ml_df.sample(20000, random_state=42) if len(ml_df) > 20000 else ml_df

        required_cols = ['start_station_name', 'end_station_name', 'rideable_type', 'member_casual', 'started_at', 'trip_duration_sec']
        if all(col in ml_sample.columns for col in required_cols):

            ml_sample['hour'] = ml_sample['started_at'].dt.hour
            ml_sample['day_of_week'] = ml_sample['started_at'].dt.dayofweek
            ml_sample['month'] = ml_sample['started_at'].dt.month

            with st.expander("🔍 View Features Used for Prediction"):
                st.write("**Features used:**")
                st.write("- Start Station")
                st.write("- End Station") 
                st.write("- Bike Type")
                st.write("- User Type")
                st.write("- Hour of Day")
                st.write("- Day of Week")
                st.write("- Month")
                st.write("**Target:** Trip Duration (seconds)")
                
                st.dataframe(ml_sample[['start_station_name', 'end_station_name', 'rideable_type', 
                                        'member_casual', 'hour', 'trip_duration_sec']].head(10))
            st.subheader("Make a Predicion")

            col1, col2 = st.columns(2)

            with col1:
                start_station = st.selectbox(
                    "Start_station",
                    options = ml_sample['start_station_name'].unique(),
                    index=0
                )

                end_station = st.selectbox(
                    "End Station",
                    options = ml_sample['end_station_name'].unique(),
                    index= 0
                )

                bike_type = st.selectbox(
                    "Bike Type",
                    options=ml_sample['rideable_type'].unique()
                )

            with col2:
                user_type = st.selectbox(
                    "User Type",
                    options=ml_sample['member_casual'].unique()
                )

                hour = st.slider(
                    "Hour of Day",
                    min_value= 0,
                    max_value= 23,
                    value = 12
                )

                day_of_week = st.selectbox(
                    "Day of Week",
                    options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    index=0
                )
                available_months = sorted(ml_sample['started_at'].dt.month.unique())
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April'}

                month = st.selectbox(
                    "Month",
                    options=available_months,
                    format_func=lambda x: month_names[x],
                    index=0  # Default to first available month
                )


            if  st.button("Predict Trip Duration", type="primary"):
                try:
                    model_data = load_model()
                    model = model_data['best_model']
                    feature_names = model_data['feature_names']

                    input_data = {
                        'start_station_name' : start_station,
                        'end_station_name': end_station,
                        'rideable_type' : bike_type,
                        'member_casual' : user_type,
                        'hour': hour,
                        'day_of_week':  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week),
                        'month': pd.Timestamp.now().month
                    }

                    input_df = pd.DataFrame([input_data])
                    input_encoded = pd.get_dummies(input_df)

                    missing_features = [feature for feature in feature_names if feature not in input_encoded.columns]
                    if missing_features:
                        missing_df = pd.DataFrame(0, index=input_encoded.index, columns = missing_features)
                        input_encoded = pd.concat([input_encoded, missing_df], axis=1)

                    input_encoded = input_encoded[feature_names]

                    predicted_minutes = model.predict(input_encoded)[0]
                    predicted_seconds = predicted_minutes * 60
                    predicted_minutes_display = predicted_seconds / 60

                    st.success(f"**Predicted Trip Duration:** {predicted_minutes_display:.1f} minutes")
                
                except FileNotFoundError:
                    base_duration = 600

                    if user_type == 'casual':
                        base_duration *= 1.3

                    if 7 <= hour <= 9 or 16 <= hour <= 18:
                        base_duration *=0.9

                    predicted_seconds = base_duration * random.uniform(0.8, 1.2)
                    predicted_minutes = predicted_seconds/60

                    st.success(f"**Estimated Trip Duration {predicted_minutes:.1f} minutes")
                    st.info(" ℹ️ Using simple heuteristic - train model for better accuracy")
        else:
            st.warning("Please load data first to use predictor")