# =============================
# Project: PLANTERRAX – Farm Assistant (No AI/ML)
# Stack: Python (Flask) + Vanilla HTML/CSS/JS
# Theme: Pixel/farm retro (matches your provided design)
# Behavior: Simple rule-based intents + external APIs (no LLMs)
# =============================

# ────────────────────────────────────────────────────────────────────────────────
# File: app.py
# ────────────────────────────────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify
import os, requests, re
from datetime import datetime

app = Flask(__name__)

# ▼ Read API keys from environment (set in .env or your hosting panel)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")  # optional if you use open-meteo
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/point"
SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_time(ts: datetime) -> str:
    return ts.strftime("%b %d, %I:%M %p")


def get_weather_by_city(city: str):
    """Weather via OpenWeatherMap (needs OPENWEATHER_API_KEY). Falls back to Open‑Meteo (no key)."""
    city = city.strip()
    try:
        if OPENWEATHER_API_KEY:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
            )
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            desc = data["weather"][0]["description"].title()
            temp = round(data["main"]["temp"], 1)
            feels = round(data["main"]["feels_like"], 1)
            hum = data["main"]["humidity"]
            wind = data["wind"]["speed"]
            return (
                f"Weather for {data['name']}: {desc}. Temp {temp}°C (feels {feels}°C), "
                f"Humidity {hum}%, Wind {wind} m/s."
            )
        else:
            # Open‑Meteo geocoding → forecast (no API key)
            g = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1},
                timeout=15,
            )
            g.raise_for_status()
            gj = g.json()
            if not gj.get("results"):
                return f"Couldn't find a location named '{city}'."
            loc = gj["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]
            fc = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "hourly": "relativehumidity_2m,apparent_temperature,precipitation",
                },
                timeout=15,
            )
            fc.raise_for_status()
            j = fc.json()
            cw = j.get("current_weather", {})
            temp = cw.get("temperature")
            wind = cw.get("windspeed")
            desc = "Clear/Clouds/Rain"  # Open‑Meteo has no text desc; keep generic
            return (
                f"Weather for {loc['name']}: {desc}. Temp {temp}°C, Wind {wind} km/h."
            )
    except Exception as e:
        return f"Weather error: {str(e)}"


def get_nasa_power(lat: float, lon: float):
    """NASA POWER Agroclimatology sample (no key). Returns last 7 days average T and solar."""
    try:
        params = {
            "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": "19810101",  # POWER requires a range; we use climatology "recent" via temporal averaging
            "end": datetime.utcnow().strftime("%Y%m%d"),
            "format": "JSON",
            "temporal": "daily",
        }
        r = requests.get(NASA_POWER_BASE, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        series = j.get("properties", {}).get("parameter", {})
        # pick last few days intersecting across keys
        keys = sorted(set().union(*[set(series.get(k, {}).keys()) for k in series.keys()]))[-7:]
        if not keys:
            return "No recent NASA POWER data found for that point."
        # compute simple averages
        def avg_for(k):
            vals = [series.get(k, {}).get(d) for d in keys if series.get(k, {}).get(d) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None
        t = avg_for("T2M")
        rh = avg_for("RH2M")
        sun = avg_for("ALLSKY_SFC_SW_DWN")
        return (
            f"NASA POWER (last ~{len(keys)} days) → Avg Temp: {t}°C, Avg RH: {rh}%, Solar Irradiance: {sun} kWh/m²/day."
        )
    except Exception as e:
        return f"NASA POWER error: {str(e)}"


def get_soil_summary(lat: float, lon: float):
    """Soil via ISRIC SoilGrids (no key). Return organic carbon and texture at topsoil."""
    try:
        params = {
            "lon": lon,
            "lat": lat,
            "property": ["soc", "clay", "sand", "silt"],
            "depth": ["0-5cm"],
            "value": "mean",
        }
        r = requests.get(SOILGRIDS_BASE, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        layers = j.get("properties", {}).get("layers", [])
        vals = {ly["name"]: ly["depths"][0]["values"].get("mean") for ly in layers}
        soc = vals.get("soc")  # g/kg
        clay = vals.get("clay")
        sand = vals.get("sand")
        silt = vals.get("silt")
        texture = None
        if None not in (clay, sand, silt):
            # very rough textural class (triangles are complex; we'll do a simple hint)
            if clay > 40:
                texture = "Clayey"
            elif sand > 60:
                texture = "Sandy"
            elif silt > 60:
                texture = "Silty"
            else:
                texture = "Loam"
        return (
            f"Soil (0–5 cm): SOC ~ {round(soc,1) if soc is not None else 'n/a'} g/kg, "
            f"Clay {round(clay,1) if clay is not None else 'n/a'}%, Sand {round(sand,1) if sand is not None else 'n/a'}%, "
            f"Silt {round(silt,1) if silt is not None else 'n/a'}% — Texture: {texture or 'n/a'}."
        )
    except Exception as e:
        return f"SoilGrids error: {str(e)}"


# ── Very simple rule-based NLU ─────────────────────────────────────────────────
CITY_RE = re.compile(r"weather\s+(?:in|for)?\s*([A-Za-z\s]+)$", re.I)
LATLON_RE = re.compile(r"(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)")


def route_message(msg: str) -> str:
    m = CITY_RE.search(msg)
    if m:
        return get_weather_by_city(m.group(1))

    if "weather" in msg.lower():
        return "Try: ‘weather Lahore’ or ‘weather in Multan’."

    if any(k in msg.lower() for k in ["nasa", "power", "solar", "radiation", "agro"]):
        ll = LATLON_RE.search(msg)
        if ll:
            lat, lon = float(ll.group(1)), float(ll.group(2))
            return get_nasa_power(lat, lon)
        return "Send coordinates like: ‘NASA 31.5204, 74.3587’. (lat, lon)"

    if any(k in msg.lower() for k in ["soil", "soilgrid", "soilgrids"]):
        ll = LATLON_RE.search(msg)
        if ll:
            lat, lon = float(ll.group(1)), float(ll.group(2))
            return get_soil_summary(lat, lon)
        return "Send coordinates like: ‘soil 30.3753, 69.3451’. (lat, lon)"

    # default canned tips
    if any(k in msg.lower() for k in ["wheat", "cotton", "maize", "rice"]):
        return (
            "General tip: Water early morning, monitor 7‑day rainfall before irrigation, and keep soil moisture 20–40% field capacity during vegetative growth."
        )

    if "help" in msg.lower():
        return (
            "I can do:\n • weather <city>\n • NASA <lat,lon> (POWER)\n • soil <lat,lon> (SoilGrids)\n • simple crop tips (wheat/cotton/rice/maize)."
        )

    return "Say ‘help’ to see commands. Try: weather Karachi | NASA 31.5, 74.3 | soil 30.4, 69.3"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.post("/chatbot")
def chatbot():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"response": "Please type a message."})
    reply = route_message(user_msg)
    return jsonify({"response": reply, "timestamp": fmt_time(datetime.now())})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))


# ────────────────────────────────────────────────────────────────────────────────
# File: requirements.txt
# ────────────────────────────────────────────────────────────────────────────────
Flask==3.0.3
requests==2.32.3
python-dotenv==1.0.1


# ────────────────────────────────────────────────────────────────────────────────
# File: .env.example  (copy to .env and fill in)
# ────────────────────────────────────────────────────────────────────────────────
# Optional (use OpenWeather) — otherwise it will fall back to Open‑Meteo which needs no key
OPENWEATHER_API_KEY=your_key_here

# Flask
FLASK_DEBUG=1
PORT=5000


# ────────────────────────────────────────────────────────────────────────────────
# File: templates/index.html
# (Your original HTML, slightly adapted: fixed one JS bug, pointed fetch to /chatbot)
# ────────────────────────────────────────────────────────────────────────────────
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PLANTERRAX - Farm Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'VT323', monospace; background: linear-gradient(180deg, #87CEEB 0%, #B0E0E6 50%, #8B7355 100%); height: 100vh; overflow: hidden; image-rendering: pixelated; display: flex; justify-content: center; align-items: center; position: relative; }
        .cloud { position: fixed; background: white; border-radius: 50%; opacity: 0.9; animation: float-cloud 25s linear infinite; z-index: 1; }
        .cloud::before, .cloud::after { content: ''; position: absolute; background: white; border-radius: 50%; }
        .cloud1 { width: 80px; height: 30px; top: 10%; left: -100px; }
        .cloud1::before { width: 40px; height: 40px; top: -15px; left: 10px; }
        .cloud1::after { width: 50px; height: 35px; top: -10px; right: 10px; }
        @keyframes float-cloud { from { transform: translateX(0); } to { transform: translateX(calc(100vw + 200px)); } }
        .sun { position: fixed; top: 30px; right: 10%; width: 60px; height: 60px; background: #FFD700; border-radius: 50%; box-shadow: 0 0 20px #FFD700, 0 0 40px #FFA500; animation: pulse-sun 3s ease-in-out infinite; z-index: 1; }
        @keyframes pulse-sun { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        .chatbot-container { width: 90%; max-width: 800px; height: 85vh; background: #D2B48C; border: 6px solid #8B4513; border-radius: 15px; box-shadow: 10px 10px 0 #000; display: flex; flex-direction: column; position: relative; z-index: 10; animation: slideUp 0.5s ease; }
        @keyframes slideUp { from { transform: translateY(50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .chatbot-container::before { content: ''; position: absolute; top: -6px; left: -6px; right: -6px; bottom: -6px; background: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(139,69,19,0.1) 10px, rgba(139,69,19,0.1) 20px); z-index: -1; border-radius: 15px; }
        .chat-header { background: #2d5016; border-bottom: 4px solid #1a3009; padding: 20px; border-radius: 9px 9px 0 0; display: flex; align-items: center; gap: 15px; }
        .bot-avatar { width: 60px; height: 60px; background: #8B4513; border: 4px solid #654321; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 3px 3px 0 #000; }
        .bot-info { flex: 1; }
        .bot-name { font-family: 'Press Start 2P', cursive; font-size: 1.3rem; color: #90EE90; text-shadow: 2px 2px 0 #000; margin-bottom: 5px; }
        .bot-status { font-size: 1.2rem; color: #FFD700; }
        .close-btn { width: 40px; height: 40px; background: #ff6b6b; border: 3px solid #c92a2a; border-radius: 5px; color: white; font-size: 1.5rem; cursor: pointer; box-shadow: 3px 3px 0 #000; transition: all 0.1s; display: flex; align-items: center; justify-content: center; }
        .close-btn:hover { transform: translate(-2px, -2px); box-shadow: 5px 5px 0 #000; }
        .close-btn:active { transform: translate(1px, 1px); box-shadow: 2px 2px 0 #000; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 25px; background: #F5DEB3; display: flex; flex-direction: column; gap: 15px; }
        .chat-messages::-webkit-scrollbar { width: 12px; }
        .chat-messages::-webkit-scrollbar-track { background: #D2B48C; border: 2px solid #8B4513; }
        .chat-messages::-webkit-scrollbar-thumb { background: #8B4513; border: 2px solid #654321; border-radius: 5px; }
        .message { display: flex; gap: 10px; animation: messageSlide 0.3s ease; }
        @keyframes messageSlide { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.bot { align-self: flex-start; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message-avatar { width: 45px; height: 45px; background: #2d5016; border: 3px solid #1a3009; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 3px 3px 0 #000; flex-shrink: 0; }
        .message.user .message-avatar { background: #4CAF50; border-color: #2d5016; }
        .message-content { max-width: 70%; background: white; padding: 15px; border: 4px solid #8B4513; border-radius: 10px; box-shadow: 4px 4px 0 rgba(0,0,0,0.3); font-size: 1.4rem; line-height: 1.6; color: #2d5016; white-space: pre-wrap; }
        .message.user .message-content { background: #90EE90; border-color: #4CAF50; }
        .message-time { font-size: 1rem; color: #654321; margin-top: 5px; }
        .typing-indicator { display: none; align-items: center; gap: 10px; }
        .typing-indicator.active { display: flex; }
        .typing-dots { display: flex; gap: 5px; padding: 15px; background: white; border: 4px solid #8B4513; border-radius: 10px; box-shadow: 4px 4px 0 rgba(0,0,0,0.3); }
        .typing-dot { width: 10px; height: 10px; background: #2d5016; border-radius: 50%; animation: typing 1.4s infinite; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }
        .chat-input-area { background: #2d5016; border-top: 4px solid #1a3009; padding: 20px; display: flex; gap: 15px; border-radius: 0 0 9px 9px; }
        .chat-input { flex: 1; padding: 15px; background: #F5DEB3; border: 4px solid #8B4513; border-radius: 8px; font-family: 'VT323', monospace; font-size: 1.4rem; color: #2d5016; outline: none; box-shadow: inset 3px 3px 0 rgba(0,0,0,0.2); resize: none; }
        .chat-input:focus { border-color: #4CAF50; box-shadow: inset 3px 3px 0 rgba(0,0,0,0.2), 0 0 0 3px rgba(76,175,80,0.3); }
        .send-btn { padding: 15px 30px; background: #FFD700; border: 4px solid #FFA500; border-radius: 8px; font-family: 'Press Start 2P', cursive; font-size: 1rem; color: #2d5016; cursor: pointer; box-shadow: 4px 4px 0 #000; transition: all 0.1s; white-space: nowrap; }
        .send-btn:hover { background: #FFA500; transform: translate(-2px, -2px); box-shadow: 6px 6px 0 #000; }
        .send-btn:active { transform: translate(2px, 2px); box-shadow: 2px 2px 0 #000; }
        .send-btn:disabled { background: #A9A9A9; border-color: #696969; cursor: not-allowed; opacity: 0.6; }
        .quick-actions { display: none; }
        @media (max-width: 768px) { .chatbot-container { width: 95%; height: 90vh; } .bot-name { font-size: 1rem; } .message-content { max-width: 85%; font-size: 1.2rem; } .send-btn { padding: 15px 20px; font-size: 0.8rem; } .quick-actions { padding: 0 15px 10px; } }
    </style>
</head>
<body>
    <div class="cloud cloud1"></div>
    <div class="sun"></div>

    <div class="chatbot-container">
        <div class="chat-header">
            <div class="bot-avatar">🤖</div>
            <div class="bot-info">
                <div class="bot-name">FARM ASSISTANT</div>
                <div class="bot-status">Online - Ready to help!</div>
            </div>
            <button class="close-btn" onclick="window.location.href='/'">✕</button>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    Welcome to PLANTERRAX Farm Assistant! I can help you with NASA satellite data, weather patterns, soil analysis, and farming tips. How can I help you today?
                    <div class="message-time">Just now</div>
                </div>
            </div>

            <div class="message bot typing-indicator" id="typingIndicator">
                <div class="message-avatar">🤖</div>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>

        <div class="chat-input-area">
            <textarea class="chat-input" id="chatInput" placeholder="Ask about farming, weather, or NASA data..." rows="1" maxlength="500"></textarea>
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">SEND</button>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const typingIndicator = document.getElementById('typingIndicator');

        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function sendMessage() {
            const message = chatInput.value.trim();
            if (!message) return;

            addMessage(message, 'user');
            chatInput.value = '';
            chatInput.style.height = 'auto';

            typingIndicator.classList.add('active');
            scrollToBottom();

            fetch('/chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            })
            .then(response => response.json())
            .then(data => {
                typingIndicator.classList.remove('active');
                addMessage(data.response || 'Sorry, I encountered an error. Please try again.', 'bot');
            })
            .catch(error => {
                typingIndicator.classList.remove('active');
                addMessage('Sorry, I am having trouble connecting. Please try again later.', 'bot');
                console.error('Error:', error);
            });
        }

        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            // FIXED bug: use template string so class becomes "message user" or "message bot"
            messageDiv.className = `message ${sender}`;

            const now = new Date();
            const time = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            const avatar = sender === 'bot' ? '🤖' : '👤';

            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">${text}
                    <div class="message-time">${time}</div>
                </div>
            `;
            chatMessages.insertBefore(messageDiv, typingIndicator);
            scrollToBottom();
        }

        function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }
        scrollToBottom();
    </script>
</body>
</html>


# ────────────────────────────────────────────────────────────────────────────────
# File: README.md (quick start)
# ────────────────────────────────────────────────────────────────────────────────
# PLANTERRAX – Farm Assistant (No AI/ML)

## Run locally
1. Python 3.10+
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and (optionally) set `OPENWEATHER_API_KEY`.
5. `python app.py` → open http://localhost:5000

## Commands to try
- `weather Lahore`
- `NASA 31.5204, 74.3587`
- `soil 30.3753, 69.3451`
- `help`

## APIs used (no ML)
- **Weather:** OpenWeatherMap (with API key) or fallback **Open‑Meteo** (no key).
- **NASA Agroclimatology:** NASA POWER (no key) for T2M/RH2M/solar.
- **Soils:** ISRIC SoilGrids (no key) for topsoil SOC and texture.

> Swap or extend intents easily in `route_message()`.
