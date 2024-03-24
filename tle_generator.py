import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def recommend_supplier(user_budget, user_location, user_raw_materials):
    # Arbitrary data generation for suppliers
    np.random.seed(0)

    # Generate supplier data
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

    # Encoding raw material
    label_encoder = LabelEncoder()
    supplier_data['Raw_Material_Encoded'] = label_encoder.fit_transform(supplier_data['Raw_Material'])

    # Perform KMeans clustering
    X_supplier = supplier_data[['Cost', 'Latitude', 'Longitude', 'Raw_Material_Encoded']]
    kmeans = KMeans(n_clusters=5, random_state=0)
    supplier_data['Cluster'] = kmeans.fit_predict(X_supplier)

    # Encode raw materials in user input
    user_input_raw_materials_encoded = label_encoder.transform(user_raw_materials)

    # User input DataFrame including raw materials
    user_input = pd.DataFrame({'Cost': [user_budget] * len(user_raw_materials),
                               'Latitude': [user_location['Latitude']] * len(user_raw_materials),
                               'Longitude': [user_location['Longitude']] * len(user_raw_materials),
                               'Raw_Material': user_raw_materials,
                               'Raw_Material_Encoded': user_input_raw_materials_encoded})

    # Predict clusters for user input
    user_clusters = kmeans.predict(user_input[['Cost', 'Latitude', 'Longitude', 'Raw_Material_Encoded']])

    recommended_suppliers = []

    for cluster, raw_material in zip(user_clusters, user_raw_materials):
        # Filter suppliers belonging to the same cluster and having the required raw material
        cluster_material_suppliers = supplier_data[(supplier_data['Cluster'] == cluster) & (supplier_data['Raw_Material'] == raw_material)]
        
        if not cluster_material_suppliers.empty:
            recommended_supplier = cluster_material_suppliers.sample(1)
            recommended_suppliers.append(recommended_supplier[['Supplier', 'Latitude', 'Longitude', 'Raw_Material']].values.tolist()[0])
        else:
            recommended_suppliers.append("No suppliers found matching the provided criteria for {}".format(raw_material))

    return recommended_suppliers

def main():
    st.title('Supplier Recommendation App')
    
    # Divide the page into two columns
    col1, col2 = st.columns([3, 2])

    # Input window on the left column
    with col1:
        user_budget = st.number_input("Enter your budget:", min_value=0.0, step=1.0)
        user_latitude = st.number_input("Enter your latitude:", min_value=-90.0, max_value=90.0)
        user_longitude = st.number_input("Enter your longitude:", min_value=-180.0, max_value=180.0)
        user_raw_materials = st.text_input("Enter your desired raw materials separated by commas:")
        if st.button("Recommend"):
            # User location dictionary
            user_location = {'Latitude': user_latitude, 'Longitude': user_longitude}

            # Split user_raw_materials into list
            user_raw_materials = [material.strip() for material in user_raw_materials.split(',')]

            # Recommend suppliers based on user input
            recommended_suppliers_data = recommend_supplier(user_budget, user_location, user_raw_materials)

    # Output window on the right column
    with col2:
        if 'recommended_suppliers_data' in locals():
            for data in recommended_suppliers_data:
                if isinstance(data, str):
                    st.write(data)
                else:
                    st.write("Recommended Supplier Data:")
                    st.write("Supplier:", data[0])
                    st.write("Latitude:", data[1])
                    st.write("Longitude:", data[2])
                    st.write("Raw Material:", data[3])

if _name_ == "_main_":
    main()
