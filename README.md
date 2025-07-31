🌊 AI-Powered Geospatial Flood Analysis Agent
This project presents an intelligent agent capable of performing complex geospatial flood analysis workflows from natural language queries. Built as a Streamlit application, it combines the power of a local, open-source Large Language Model (LLM) with a robust set of Python geospatial tools to provide a seamless and intuitive user experience.

The core goal of this project is to automate the multi-step process of identifying, analyzing, and visualizing flood-prone areas, a task that traditionally requires specialized software and expertise.

🌟 Key Features
Natural Language Interface: Users can describe complex geospatial analysis tasks in plain English (e.g., "Find flood prone areas and mark them on a Maharashtra map").

Intelligent Workflow Planning: An open-source LLM (Mistral-7B) acts as a planner, generating a step-by-step, executable JSON plan based on the user's query and a predefined set of tools.

Modular Toolset: The agent is powered by a suite of modular Python functions built on libraries like geopandas, rasterio, and matplotlib to handle tasks such as DEM processing, flood simulation, and spatial clipping.

Advanced Visualization: The final output is a high-quality map visualization, combining simulated flood extents and stream networks overlaid on a detailed basemap (e.g., OpenStreetMap) for clear geographic context.

Performance Metrics: The system includes a framework for evaluating key performance metrics like runtime efficiency and plan accuracy, demonstrating the advantages of this automated approach.

🎯 Advanced Workflow Demonstration
The primary demonstration for this project is a detailed flood risk analysis for the state of Maharashtra, India. The workflow, which is automatically generated and executed by the agent, involves the following steps:

Load DEM Data: Load a high-resolution Digital Elevation Model for Maharashtra.

Hydrological Analysis: Calculate flow direction, flow accumulation, and extract the stream network to understand water pathways.

Flood Simulation: Simulate flood inundation at a specified water level (e.g., 160 meters).

Spatial Filtering: Load Maharashtra's administrative boundary and precisely clip both the flood extent and the stream network to this boundary.

Visualization: Display a final map combining the clipped flood areas and stream network on an interactive basemap (using contextily for OpenStreetMap tiles).

This complex, multi-step process is executed from a single natural language prompt, highlighting the agent's ability to automate specialized GIS tasks.

🚀 How to Run the Application
Prerequisites
You need the following installed:

Python 3.8+

Git

The required Python libraries listed in requirements.txt.

Step-by-Step Instructions
Clone the repository:

git clone https://github.com/your-username/Geospatial-Analysis-Streamlit.git
cd Geospatial-Analysis-Streamlit

Set up your Python environment:

python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements.txt

Download and place the geospatial data:

DEM: Download a high-resolution DEM for Maharashtra (e.g., SRTM or ALOS PALSAR) and rename it to maharashtra_dem.tif.

Maharashtra Boundary: Download the administrative boundary shapefile for India (e.g., from GADM), and filter it to Maharashtra. Ensure all shapefile components (.shp, .shx, .dbf, .prj, etc.) are present and renamed to maharashtra_boundary.*.

Place these files in the data/ directory.

Run the Streamlit application:

streamlit run app.py

The application will open in your browser, where you can interact with the agent using natural language.

📂 Repository Contents
app.py: The main Streamlit application code.

data/: Directory to hold all input geospatial data files.

flood_analysis_outputs/: Directory for storing all generated intermediate and final output files. This is ignored by Git.

requirements.txt: Lists all Python dependencies.

🛠️ Technical Stack
LLM: Mistral-7B-Instruct-v0.3

Framework: Streamlit for the web UI.

Geospatial Libraries: geopandas, rasterio, matplotlib, contextily, pyproj, shapely.

LLM Integration: transformers, accelerate, bitsandbytes, langchain-community.

Version Control: Git & GitHub.
