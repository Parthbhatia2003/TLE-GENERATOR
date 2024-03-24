import numpy as np
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def recommend_supplier(user_budget, user_location, user_raw_material):
    np.random.seed(0)

    num_suppliers = 1000
    num_raw_materials = 5
    supplier_cost = np.random.randint(50, 200, num_suppliers)
    supplier_latitude = np.random.uniform(low=-90.0, high=90.0, size=num_suppliers)
    supplier_longitude = np.random.uniform(low=-180.0, high=180.0, size=num_suppliers)
    supplier_names = [f"Supplier_{i+1}" for i in range(num_suppliers)]
    raw_materials = ['Steel', 'Plastic', 'Aluminum', 'Wood', 'Glass'] * (num_suppliers // num_raw_materials)

    supplier_data = pd.DataFrame({
        'Supplier': supplier_names,
        'Cost': supplier_cost,
        'Latitude': supplier_latitude,
        'Longitude': supplier_longitude,
        'Raw_Material': np.random.choice(raw_materials, size=num_suppliers)
    })

    label_encoder = LabelEncoder()
    supplier_data['Raw_Material_Encoded'] = label_encoder.fit_transform(supplier_data['Raw_Material'])

    X_supplier = supplier_data[['Cost', 'Latitude', 'Longitude', 'Raw_Material_Encoded']]
    kmeans = KMeans(n_clusters=5, random_state=0)
    supplier_data['Cluster'] = kmeans.fit_predict(X_supplier)

    user_input_raw_material_encoded = label_encoder.transform([user_raw_material])[0]

    user_input = pd.DataFrame({'Cost': [user_budget],
                               'Latitude': [user_location['Latitude']],
                               'Longitude': [user_location['Longitude']],
                               'Raw_Material': [user_raw_material],
                               'Raw_Material_Encoded': [user_input_raw_material_encoded]})

    user_cluster = kmeans.predict(user_input[['Cost', 'Latitude', 'Longitude', 'Raw_Material_Encoded']])[0]

    cluster_material_suppliers = supplier_data[(supplier_data['Cluster'] == user_cluster) & (supplier_data['Raw_Material'] == user_raw_material)]

    if not cluster_material_suppliers.empty:
        recommended_supplier = cluster_material_suppliers.sample(1)
        return recommended_supplier[['Supplier', 'Latitude', 'Longitude', 'Raw_Material']].values.tolist()[0]
    else:
        return "No suppliers found matching the provided criteria."

def get_valid_float_input(prompt, key):
    while True:
        try:
            value = float(st.text_input(prompt, key=key))  # Added 'key' argument
            return value
        except ValueError:
            st.error("Please enter a valid number.")

def main():
    st.title("Larsen&Toubro")

    user_budget = get_valid_float_input("Enter your budget:", "budget")  # Added 'key' argument
    user_latitude = get_valid_float_input("Enter your latitude:", "latitude")  # Added 'key' argument
    user_longitude = get_valid_float_input("Enter your longitude:", "longitude")  # Added 'key' argument
    user_raw_material = st.text_input("Enter your desired raw material:", key="raw_material")  # Added 'key' argument

    user_location = {'Latitude': user_latitude, 'Longitude': user_longitude}

    recommended_supplier_data = recommend_supplier(user_budget, user_location, user_raw_material)

    if isinstance(recommended_supplier_data, str):
        st.error(recommended_supplier_data)
    else:
        st.subheader("Recommended Supplier Data:")
        st.write("Supplier:", recommended_supplier_data[0])
        st.write("Latitude:", recommended_supplier_data[1])
        st.write("Longitude:", recommended_supplier_data[2])
        st.write("Raw Material:", recommended_supplier_data[3])

        geolocator = Nominatim(user_agent="my_geocoder")
        start_lat, start_lon = recommended_supplier_data[1], recommended_supplier_data[2] 
        end_lat, end_lon = user_latitude, user_longitude  

        total_distance_km = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers

        if total_distance_km < 0.04:  
            num_stops = 0
        elif total_distance_km < 0.075: 
            num_stops = 1
        elif total_distance_km < 0.4:
            num_stops = 4
        elif total_distance_km < 1: 
            num_stops = 6
        elif total_distance_km < 3:
            num_stops = int(total_distance_km / 250) 
        else :
            num_stops = int(total_distance_km/ 1000)

        lat_step = (end_lat - start_lat) / (num_stops + 1)
        lon_step = (end_lon - start_lon) / (num_stops + 1)

        lats = [start_lat]
        lons = [start_lon]
        locations = []

        for i in range(1, num_stops + 1):
            current_lat = start_lat + i * lat_step
            current_lon = start_lon + i * lon_step

            lats.append(current_lat)
            lons.append(current_lon)

            location = geolocator.reverse((current_lat, current_lon), exactly_one=True)
            if location:
                locations.append(location.address)

        lats.append(end_lat)
        lons.append(end_lon)
        destination = geolocator.reverse((end_lat, end_lon), exactly_one=True)
        if destination:
            locations.append(destination.address)

        st.subheader("Route Information:")
        st.write("Number of Stops:", num_stops)

        for i, location in enumerate(locations):
            st.write(f"Stop {i+1}: {location}")

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            mode="lines+markers",
            lon=lons,
            lat=lats,
            marker={'size': 10}))

        for lat, lon, loc in zip(lats, lons, locations):
            fig.add_trace(go.Scattergeo(
                mode="markers+text",
                lon=[lon],
                lat=[lat],
                text=[loc],
                marker={'size': 10},
                textposition="bottom right"))

        st.subheader("Route Map:")
        st.plotly_chart(fig)

if _name_ == "_main_":
    main()
