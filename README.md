# DVProject – Member 3: Earthquake Analysis and Clustering

## 📌 Project Objective
This project aims to analyze **Turkey’s earthquakes from 1915 to February 2024** by:

- Investigating relationships between different magnitude types (MD, ML, Mw),  
- Clustering earthquakes based on **depth, magnitude, and location (K-Means)**,  
- Visualizing the relationship between magnitude and depth.  

The project is implemented as an **interactive Streamlit dashboard**.

## 📊 Member 3 Dashboard Contents

### 1. Magnitude Types Relationship – Parallel Coordinates
- **Purpose:** Explore relationships between MD, ML, and Mw magnitude types  
- **Features:** Axis brushing, hover highlight, zoom  
- **Insight:** Visualizes correlations and measurement differences between magnitude types  

### 2. K-Means Clustering – Earthquake Clusters
- **Purpose:** Cluster earthquakes by magnitude, depth, and location  
- **Features:** Select K value via slider (2–6), 3D scatter plot, cluster color coding  
- **Insight:** Discover patterns and natural clusters in earthquake data  

### 3. Bubble Chart – Magnitude vs Depth
- **Purpose:** Analyze whether large earthquakes occur shallow or deep  
- **Features:** Year range filter, Bubble size = Magnitude, Hover tooltip  
- **Insight:** Visualizes magnitude–depth relationships
