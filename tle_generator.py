import numpy as np
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def recommend_supplier(user_budget, user_location, user_raw_materials):
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

    user_input_raw_materials_encoded = label_encoder.transform(user_raw_materials)

    user_input = pd.DataFrame({'Cost': [user_budget] * len(user_raw_materials),
                               'Latitude': [user_location['Latitude']] * len(user_raw_materials),
                               'Longitude': [user_location['Longitude']] * len(user_raw_materials),
                               'Raw_Material': user_raw_materials,
                               'Raw_Material_Encoded': user_input_raw_materials_encoded})

    user_clusters = kmeans.predict(user_input[['Cost', 'Latitude', 'Longitude', 'Raw_Material_Encoded']])

    recommended_suppliers = []

    for cluster, raw_material in zip(user_clusters, user_raw_materials):
        cluster_material_suppliers = supplier_data[(supplier_data['Cluster'] == cluster) & (supplier_data['Raw_Material'] == raw_material)]
        
        if not cluster_material_suppliers.empty:
            recommended_supplier = cluster_material_suppliers.sample(1)
            recommended_suppliers.append(recommended_supplier[['Supplier', 'Latitude', 'Longitude', 'Raw_Material']].values.tolist()[0])
        else:
            recommended_suppliers.append("No suppliers found matching the provided criteria for {}".format(raw_material))

    return recommended_suppliers

def main():
    st.title("Supplier Recommender App")
    
    user_budget = st.number_input("Enter your budget:", value=0.0)
    user_latitude = st.number_input("Enter your latitude:", value=0.0)
    user_longitude = st.number_input("Enter your longitude:", value=0.0)
    user_raw_materials = st.text_input("Enter your desired raw materials separated by commas:")

    user_location = {'Latitude': user_latitude, 'Longitude': user_longitude}

    if st.button("Recommend Suppliers"):
        recommended_suppliers_data = recommend_supplier(user_budget, user_location, user_raw_materials.split(','))

        fig = go.Figure()
        for data in recommended_suppliers_data:
            if isinstance(data, str):
                st.write(data)
            else:
                st.write("Recommended Supplier Data:")
                st.write("Supplier:", data[0])
                st.write("Latitude:", data[1])
                st.write("Longitude:", data[2])
                st.write("Raw Material:", data[3])

                geolocator = Nominatim(user_agent="my_geocoder")

                start_lat, start_lon = data[1], data[2] 
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

                def get_city_name(latitude, longitude):
                    geolocator = Nominatim(user_agent="geo_locator")
                    
                    location = geolocator.reverse((latitude, longitude), exactly_one=True)
                    
                    if location:
                        address = location.raw['address']
                        city = address.get('city', '')
                        if not city:
                            city = address.get('town', '')
                        if not city:
                            city = address.get('village', '')
                        return city
                    else:
                        return "Unknown City"

                lat_step = (end_lat - start_lat) / (num_stops + 1)
                lon_step = (end_lon - start_lon) / (num_stops + 1)
                startCity = get_city_name(start_lat, end_lat)

                lats = [start_lat]
                lons = [start_lon]
                locations = [data[3] + " Supplier " + startCity]
                for i in range(1, num_stops + 1):

                    current_lat = start_lat + i * lat_step
                    current_lon = start_lon + i * lon_step
                    

                    lats.append(current_lat)
                    lons.append(current_lon)
                    

                    location = geolocator.reverse((current_lat, current_lon), exactly_one=True)
                    if location:
                        locations.append(location.address)
                    else :
                        locations.append("Unknown")
                lats.append(end_lat)
                lons.append(end_lon)
                destination = geolocator.reverse((end_lat, end_lon), exactly_one=True)
                if destination:
                    locations.append("Buyer " + destination.address)
                else:
                    locations.append("Destination")

                st.write("Number of Stops:", num_stops)

                for i, location in enumerate(locations):
                    st.write(f"Stop {i+1}: {location}")
                
                fig.add_trace(go.Scattergeo(
                    mode="lines+markers",
                    lon=lons,
                    lat=lats,
                    marker={'size': 10}
                ))

                for lat, lon, loc in zip(lats, lons, locations):
                    fig.add_trace(go.Scattergeo(
                        mode="markers+text",
                        lon=[lon],
                        lat=[lat],
                        text=[loc],
                        marker={'size': 10},
                        textposition="bottom right"))

        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
