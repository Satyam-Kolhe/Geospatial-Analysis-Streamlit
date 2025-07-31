# 🌊 AI-Powered Geospatial Flood Analysis Agent

Welcome to an intelligent agent designed to automate complex geospatial flood analysis workflows using natural language commands. Built as a Streamlit application, this project seamlessly blends the power of a local, open-source Large Language Model (LLM) with robust Python geospatial libraries to deliver an intuitive and highly capable GIS experience.

---

## ✨ Key Features

- **Natural Language Interface:**  
  Describe sophisticated geospatial analysis tasks in plain English  
  _Example:_ “Find flood prone areas and mark them on a Maharashtra map.”

- **Intelligent Workflow Planning:**  
  Uses an open-source LLM (_Mistral-7B_) to generate step-by-step, executable JSON plans based on your queries.

- **Modular Geospatial Toolset:**  
  Employs Python functions built on libraries like `geopandas`, `rasterio`, and `matplotlib` for DEM processing, flood simulation, spatial clipping, and more.

- **Advanced Visualization:**  
  Produces high-quality map outputs, combining simulated flood extents and stream networks, overlaid on detailed basemaps (e.g., OpenStreetMap).

- **Performance Metrics:**  
  Built-in framework for evaluating runtime efficiency and plan accuracy, highlighting the benefits of automation.

---

## 🎯 Advanced Workflow Demonstration

**Primary Demo:**  
_Detailed flood risk analysis for Maharashtra, India._

**Automated Workflow Steps:**
1. **Load DEM Data:**  
   Import a high-resolution Digital Elevation Model (DEM) for Maharashtra.
2. **Hydrological Analysis:**  
   Compute flow direction, flow accumulation, and extract stream networks to trace water pathways.
3. **Flood Simulation:**  
   Simulate flood inundation for a specified water level (e.g., 160 meters).
4. **Spatial Filtering:**  
   Clip flood extents and stream networks precisely to Maharashtra’s administrative boundary.
5. **Visualization:**  
   Display results on an interactive basemap using OpenStreetMap tiles.

All steps are executed from a single, natural language prompt—demonstrating automation of expert-level GIS workflows.

---

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8+
- Git
- All Python libraries in `requirements.txt`

### **Setup Instructions**
```bash
# 1. Clone the repository
git clone https://github.com/your-username/Geospatial-Analysis-Streamlit.git

# 2. Enter the project directory
cd Geospatial-Analysis-Streamlit

# 3. Set up your Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# 4. Install dependencies
pip install -r requirements.txt
```

### **Prepare Geospatial Data**
- **DEM:**  
  Download a high-resolution DEM for Maharashtra (SRTM or ALOS PALSAR).  
  _Rename to:_ `maharashtra_dem.tif`

- **Maharashtra Boundary:**  
  Download the administrative boundary shapefile for India (e.g., from GADM), filter to Maharashtra, and ensure all shapefile components (`.shp`, `.shx`, `.dbf`, `.prj`, etc.) are present and renamed to `maharashtra_boundary.*`

- Place all files in the `data/` directory.

### **Run the Application**
```bash
streamlit run app.py
```
_The app will launch in your browser. Interact with the agent using natural language!_

---

## 📁 Repository Structure

```
app.py                   # Main Streamlit application
data/                    # Input geospatial data files
flood_analysis_outputs/   # Stores generated outputs (intermediate & final, gitignored)
requirements.txt         # Python dependencies
```

---

## 🛠️ Technical Stack

- **LLM:** Mistral-7B-Instruct-v0.3
- **Web Framework:** Streamlit
- **Geospatial Libraries:** geopandas, rasterio, matplotlib, contextily, pyproj, shapely
- **LLM Integration:** transformers, accelerate, bitsandbytes, langchain-community
- **Version Control:** Git & GitHub

---

## 🌐 License

This project is [MIT Licensed](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

---

## 📣 Acknowledgements

- [OpenStreetMap](https://www.openstreetmap.org/)
- [GADM Database of Global Administrative Areas](https://gadm.org/)
- [Mistral AI](https://mistral.ai/)
