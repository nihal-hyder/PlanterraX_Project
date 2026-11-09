# PlanterraX 🌱

**Making smarter agriculture easier to understand through NASA Earth observation data, interactive learning, and AI.**

PlanterraX is an agricultural learning platform developed for the NASA Space Apps Challenge. It brings together Earth observation data, an interactive irrigation simulation, and an AI-powered farming assistant to explore how satellite data and technology can support better agricultural decisions.

The project focuses on an important agricultural challenge: understanding soil conditions, rainfall, and crop water requirements to make more informed irrigation decisions.

## 🌍 Project Overview

Agriculture depends heavily on water availability and soil conditions. However, understanding when and how much to irrigate can be difficult.

PlanterraX approaches this problem through two connected experiences:

- **Earth Observation:** Retrieve environmental information using NASA data sources, including soil moisture, precipitation, humidity, and land-cover information.
- **Interactive Learning:** Explore irrigation decisions through a game where users manage soil moisture according to different soil types, crops, and rainfall conditions.

An AI-powered farming assistant provides an additional way to explore agricultural concepts and ask questions.

## ✨ Features

### 🛰️ NASA Earth Observation Data

PlanterraX integrates multiple NASA data sources to explore environmental conditions:

- **NASA SMAP L4:** Retrieves surface and root-zone soil moisture information.
- **NASA MODIS MCD12Q1:** Provides land-cover classification and helps identify cropland areas.
- **NASA POWER:** Retrieves precipitation and relative humidity information.

Users can request environmental data using geographic coordinates.

### 💧 Interactive Irrigation Simulation

Learn about irrigation through an interactive, level-based simulation.

The simulation includes:

- Different soil types: Sandy, Loamy, and Clay.
- Multiple crops, including Wheat, Maize, Rice, Soybean, and Potato.
- Crop-specific ideal soil moisture ranges.
- Randomized initial soil moisture conditions.
- Rainfall considerations when evaluating irrigation decisions.
- Feedback explaining whether the watering decision was insufficient, appropriate, or excessive.

Correct decisions allow users to progress through levels.

### 🤖 AI Farming Assistant

PlanterraX includes an AI-powered chatbot designed to answer questions about:

- Agricultural practices.
- NASA satellite data concepts.
- Soil and water conditions.
- Evapotranspiration basics.
- Practical farming guidance.

The assistant uses the Hugging Face Inference API with the Zephyr 7B Beta model.

### 📊 User Progress Tracking

The platform keeps track of learning progress, including:

- Highest level reached.
- Total attempts.
- Correct attempts.
- Current correct-answer streak.
- Best correct-answer streak.

Progress is stored using SQLite.

### 👤 User Accounts

The application includes:

- User registration and login.
- Password hashing.
- Session-based authentication.
- Password change and reset interfaces.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Authentication | Flask-Login, Werkzeug |
| Database | SQLite |
| AI Integration | Hugging Face Inference API |
| AI Model | HuggingFaceH4/zephyr-7b-beta |
| NASA Data | SMAP, MODIS, NASA POWER |
| Data Processing | Earthaccess, Xarray, NumPy |
| HTTP Requests | Requests |

## 🏗️ Project Structure

```text
PlanterraX_Project/
│
├── app.py
│
├── chatbot/
│
├── templates/
│
├── progress.db
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or newer
- Git
- A Hugging Face API token for chatbot functionality
- NASA Earthdata credentials for accessing applicable Earth observation datasets

### 1. Clone the Repository

```bash
git clone https://github.com/nihal-hyder/PlanterraX_Project.git

cd PlanterraX_Project
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv

venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install Flask Flask-Login Werkzeug requests python-dotenv earthaccess xarray numpy
```

### 4. Configure Environment Variables

Create a `.env` file in the project directory:

```env
HF_TOKEN=your_huggingface_token
```

The chatbot requires a valid Hugging Face token.

For NASA Earthdata access, configure your Earthdata credentials using the authentication method supported by Earthaccess.

### 5. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 🧪 How the Irrigation Simulation Works

The simulation generates a scenario containing:

1. A soil type and its water-holding capacity.
2. An initial soil moisture level.
3. A crop with its ideal moisture range.
4. A rainfall condition.

Users decide how much additional water to apply.

The application calculates the resulting moisture level and evaluates the decision against the crop's ideal range, soil capacity, and rainfall conditions.

Correct decisions unlock the next level and update the user's progress.

**Note:** The irrigation simulation uses generated scenarios and predefined crop and soil parameters. It is an educational simulation, not a field-tested irrigation recommendation system.

## 🛰️ NASA Data Sources

PlanterraX uses the following NASA data services:

- [NASA SMAP](https://smap.jpl.nasa.gov/) — Soil Moisture Active Passive mission.
- [NASA MODIS](https://modis.gsfc.nasa.gov/) — Moderate Resolution Imaging Spectroradiometer.
- [NASA POWER](https://power.larc.nasa.gov/) — Prediction of Worldwide Energy Resources.

These datasets provide environmental information that can help users explore the relationship between land conditions, water availability, and agriculture.

## 🔐 Security and Deployment Notes

PlanterraX is an experimental educational project. Before deploying it publicly, review and improve its security configuration.

In particular:

- Move the Flask secret key into environment variables.
- Replace hardcoded user credentials with persistent account storage.
- Implement secure, verified password-reset procedures.
- Configure secure session cookies and production settings.
- Disable Flask debug mode in production.
- Review input validation and authentication controls.

Never commit API tokens, passwords, or other secrets to the repository.

## 🎯 Project Purpose

PlanterraX explores how space-based environmental data, artificial intelligence, and interactive learning can make agricultural concepts more accessible.

The goal is to help users better understand the factors involved in irrigation decisions and the potential of Earth observation technology in agriculture.

## 📄 License

Add your chosen open-source license here before distributing the project.

---

**Built for the NASA Space Apps Challenge.**
