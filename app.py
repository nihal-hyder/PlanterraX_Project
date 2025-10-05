from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, session
)
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import os
import time
import requests
import random
import sqlite3
from contextlib import closing
from datetime import datetime
import math
import json

# -----------------------------
# Hugging Face Inference API (simple + free)
# -----------------------------

# If you installed python-dotenv, this will load .env automatically in dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HF_TOKEN = os.getenv("HF_TOKEN") or ""   # put your token in .env as HF_TOKEN=hf_xxx
if not HF_TOKEN.startswith("hf_"):
    print("WARNING: HF_TOKEN missing/invalid. Set HF_TOKEN in environment for /chatbot to work.")

HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def hf_generate(user_msg: str) -> str:
    if not HF_TOKEN:
        return "HF token not configured on server. Please set HF_TOKEN in environment."

    system_prompt = (
        "You are PLANTERRAX, a friendly farm assistant. "
        "Help with NASA satellite data concepts, local weather patterns, soil/water/ET basics, "
        "and practical farming tips. Keep answers clear and concise."
    )
    prompt = f"{system_prompt}\n\nUser: {user_msg}\nAssistant:"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    for attempt in range(1, 4):
        try:
            r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=60)
        except Exception as e:
            return f"Network error contacting HF: {type(e).__name__}: {e}"

        if r.status_code == 503:
            time.sleep(4 * attempt)
            continue
        if r.status_code == 401:
            return "HF says 401 Unauthorized. Check your HF token value."
        if r.status_code == 403:
            return "HF says 403 Forbidden. This model is gated for your account (Zephyr shouldn't be)."
        if r.status_code == 404:
            return ("HF says 404 Not Found. Model id must be exact:\n"
                    "HuggingFaceH4/zephyr-7b-beta (case-sensitive, no spaces).")
        if not r.ok:
            return f"HF error {r.status_code}: {r.text[:200].replace(chr(10),' ')}"

        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return (data[0]["generated_text"] or "").strip()
        return str(data)[:1000]

    return "Model is busy—please try again in a moment."

# -----------------------------
# Flask app setup
# -----------------------------

app = Flask(__name__)
app.secret_key = "@secret_key_nasa_HACAKTON"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # SESSION_COOKIE_SECURE=True,  # enable on HTTPS
)

# -----------------------------
# Simple SQLite for progress persistence
# -----------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "progress.db")

def db_connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    with closing(db_connect()) as conn, conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            username TEXT PRIMARY KEY,
            best_level INTEGER NOT NULL DEFAULT 1,
            total_attempts INTEGER NOT NULL DEFAULT 0,
            total_correct INTEGER NOT NULL DEFAULT 0,
            current_streak INTEGER NOT NULL DEFAULT 0,
            best_streak INTEGER NOT NULL DEFAULT 0
        );
        """)

def ensure_progress_row(username: str):
    with closing(db_connect()) as conn, conn:
        cur = conn.execute("SELECT username FROM progress WHERE username = ?", (username,))
        if not cur.fetchone():
            conn.execute("""
                INSERT INTO progress (username, best_level, total_attempts, total_correct, current_streak, best_streak)
                VALUES (?, 1, 0, 0, 0, 0)
            """, (username,))

def get_progress(username: str) -> dict:
    ensure_progress_row(username)
    with closing(db_connect()) as conn:
        cur = conn.execute("""
            SELECT best_level, total_attempts, total_correct, current_streak, best_streak
            FROM progress WHERE username = ?
        """, (username,))
        row = cur.fetchone()
        keys = ["best_level", "total_attempts", "total_correct", "current_streak", "best_streak"]
        return dict(zip(keys, row))

def update_progress_on_attempt(username: str, correct: bool, new_level_reached: int | None = None):
    ensure_progress_row(username)
    with closing(db_connect()) as conn, conn:
        cur = conn.execute("""
            SELECT best_level, total_attempts, total_correct, current_streak, best_streak
            FROM progress WHERE username = ?
        """, (username,))
        best_level, total_attempts, total_correct, current_streak, best_streak = cur.fetchone()

        total_attempts += 1
        if correct:
            total_correct += 1
            current_streak += 1
            if current_streak > best_streak:
                best_streak = current_streak
            if new_level_reached is not None and new_level_reached > best_level:
                best_level = new_level_reached
        else:
            current_streak = 0

        conn.execute("""
            UPDATE progress
               SET best_level = ?,
                   total_attempts = ?,
                   total_correct = ?,
                   current_streak = ?,
                   best_streak = ?
             WHERE username = ?
        """, (best_level, total_attempts, total_correct, current_streak, best_streak, username))

# -----------------------------
# Earthdata helpers (SMAP L4 + MODIS LC + POWER precip)
# -----------------------------
def _round(x, n=3):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)
    except Exception:
        return None

def _landcover_label_from_mcd12(type1_value: int | None) -> str | None:
    """
    MODIS MCD12Q1 IGBP legend (LC_Type1); key classes:
    12 = Croplands
    14 = Cropland/Natural Mosaic
    """
    if type1_value is None:
        return None
    lc_map = {
        0: "Water",
        1: "Evergreen Needleleaf Forest",
        2: "Evergreen Broadleaf Forest",
        3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest",
        5: "Mixed Forests",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up",
        14: "Cropland/Natural Mosaic",
        15: "Snow and Ice",
        16: "Barren or Sparsely Vegetated",
        254: "Unclassified",
        255: "Fill"
    }
    return lc_map.get(int(type1_value), None)

def _is_cropland(type1_value: int | None) -> bool:
    return type1_value in (12, 14)

init_db()

# -----------------------------
# Flask-Login
# -----------------------------
login_manager = LoginManager(app)

@login_manager.user_loader
def load_user(user_id: str):
    if user_id in USER:
        return User(user_id)
    return None

login_manager.login_view = "login"
login_manager.login_message_category = "info"

USER = {
    "nihal":  generate_password_hash("#Shah1125"),
    "Abbass": generate_password_hash("123456789"),
    "Ayaan":  generate_password_hash("7899456"),
    "shayan": generate_password_hash("nooneknows"),
}

class User(UserMixin):
    def __init__(self, username: str):
        self.id = username

# -----------------------------
# Game Logic Constants
# -----------------------------

SOIL_CAPACITY = {"Sandy": 50, "Loamy": 70, "Clay": 90}
SOIL_TYPES = list(SOIL_CAPACITY.keys())

CROP_IDEAL_RANGES = {
    "Wheat": (35, 60),
    "Maize": (40, 70),
    "Rice":  (60, 80),
    "Soybean": (45, 65),
    "Potato": (65, 80)
}
DEFAULT_IDEAL_RANGE = (40, 70)
RAIN_BUFFER = 10  # % points

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def pick_crop_for_level(level: int) -> str:
    crops = list(CROP_IDEAL_RANGES.keys())
    return crops[(level - 1) % len(crops)]

def gen_level_state(level: int) -> dict:
    soil = random.choice(SOIL_TYPES)
    capacity = SOIL_CAPACITY[soil]
    start_moisture = random.randint(0, capacity)
    rain = bool(random.getrandbits(1))
    crop = pick_crop_for_level(level)
    lo, hi = CROP_IDEAL_RANGES.get(crop, DEFAULT_IDEAL_RANGE)

    return {
        "level": int(level),
        "soilType": soil,
        "capacity": capacity,
        "moisture": start_moisture,
        "crop": crop,
        "idealLow": int(lo),
        "idealHigh": int(hi),
        "forecastRain": rain,
    }

def ensure_game_session(initial_level: int = 1) -> dict:
    game = session.get("game")
    if not game:
        game = {
            "currentLevel": int(initial_level),
            "state": gen_level_state(int(initial_level)),
        }
        session["game"] = game
    return game

# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        if request.form.get("start-button") is not None or True:
            if current_user.is_authenticated:
                return redirect(url_for("game"))
            else:
                return redirect(url_for("login"))
    return render_template("main.html")

@app.route("/main", methods=["GET", "POST"])
@login_required
def game():
    return render_template("game.html", usr=current_user.id)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        usr = (request.form.get("usr") or "").strip()
        passw = (request.form.get("passw") or "")

        stored_hash = USER.get(usr)
        if stored_hash and check_password_hash(stored_hash, passw):
            login_user(User(usr), remember=bool(request.form.get("remember")))
            ensure_progress_row(usr)
            next_url = request.args.get("next")
            return redirect(next_url or url_for("game"))
        else:
            flash("Invalid credentials.", "danger")
            return render_template("login.html", wrong="Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        _email  = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        confirm  = request.form.get("confirm_password") or ""

        if not username or not password:
            return render_template("signup.html", error="Username and password are required.")
        if password != confirm:
            return render_template("signup.html", error="Passwords do not match.")
        if username in USER:
            return render_template("signup.html", error="Username already exists. Try another.")

        USER[username] = generate_password_hash(password)
        login_user(User(username))
        ensure_progress_row(username)
        return render_template("signup.html", success="Account created !, Kindly Login Again")
    return render_template("signup.html")

@app.route("/api/nasa_power")
@login_required
def api_nasa_power():
    from datetime import datetime as _dt
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Provide lat and lon query params"}), 400

    today = _dt.utcnow().strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR,RH2M&community=AG&longitude={lon}&latitude={lat}"
        f"&start={today}&end={today}&format=JSON"
    )
    try:
        r = requests.get(url, timeout=25)
        if not r.ok:
            return jsonify({"ok": False, "error": f"POWER error {r.status_code}"}), 502
        data = r.json()
        p = data.get("properties", {}).get("parameter", {})
        precip = (p.get("PRECTOTCORR") or {}).get(today)  # mm/day
        rh = (p.get("RH2M") or {}).get(today)             # %
        rain_today = bool(precip is not None and float(precip) >= 1.0)
        return jsonify({
            "ok": True,
            "precip_mm": None if precip is None else round(float(precip), 2),
            "rel_humidity_pct": None if rh is None else round(float(rh), 1),
            "rain_today": rain_today
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"POWER request failed: {e}"}), 502

@app.route("/api/earthdata")
@login_required
def api_earthdata():
    """
    GET /api/earthdata?lat=..&lon=..
    Returns:
      {
        ok: true,
        power: { precip_mm, rel_humidity_pct, rain_today },
        smap: { rootzone_sm_m3m3, surface_sm_m3m3, time_iso },
        modis: { lc_type1, landcover_label, is_cropland, year }
      }
    """
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Provide lat and lon query params"}), 400

    # ---- 1) NASA POWER (precip/humidity; free, no auth)
    power_part = None
    try:
        from datetime import datetime as _dt
        today = _dt.utcnow().strftime("%Y%m%d")
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=PRECTOTCORR,RH2M&community=AG&longitude={lon}&latitude={lat}"
            f"&start={today}&end={today}&format=JSON"
        )
        r = requests.get(url, timeout=25)
        p = r.json().get("properties", {}).get("parameter", {})
        precip = (p.get("PRECTOTCORR") or {}).get(today)
        rh = (p.get("RH2M") or {}).get(today)
        rain_today = bool(precip is not None and float(precip) >= 1.0)
        power_part = {
            "precip_mm": None if precip is None else round(float(precip), 2),
            "rel_humidity_pct": None if rh is None else round(float(rh), 1),
            "rain_today": rain_today
        }
    except Exception as e:
        power_part = {"error": f"POWER request failed: {e}"}

    # ---- 2) SMAP L4 (root-zone & surface soil moisture) via Earthdata Cloud
    smap_part = None
    try:
        import earthaccess, xarray as xr
        import numpy as np
        from datetime import datetime as _dt2, timedelta

        earthaccess.login(strategy="netrc")

        end = _dt2.utcnow()
        start = end - timedelta(days=1)

        results = earthaccess.search_data(
            short_name="SPL4SMGP",
            temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
            bounding_box=(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1),
            cloud_hosted=True
        )
        if not results:
            results = earthaccess.search_data(
                short_name="SPL4SMGP",
                temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                cloud_hosted=True
            )

        if results:
            urls = earthaccess.sign(results[0:1])
            ds = xr.open_dataset(urls[0])

            var_candidates_surface = [
                "SurfaceSoilMoisture_inst", "sm_surface", "surface_soil_moisture", "ssm"
            ]
            var_candidates_root = [
                "RootZoneSoilMoisture_inst", "sm_rootzone", "root_zone_soil_moisture", "rzsm"
            ]

            vs = next((v for v in var_candidates_surface if v in ds.variables), None)
            vr = next((v for v in var_candidates_root if v in ds.variables), None)

            lat_name = "lat" if "lat" in ds.dims else [d for d in ds.dims if "lat" in d][0]
            lon_name = "lon" if "lon" in ds.dims else [d for d in ds.dims if "lon" in d][0]
            nearest = {lat_name: float(lat), lon_name: float(lon)}

            def _pick(var):
                try:
                    da = ds[var].sel(method="nearest", **nearest)
                    if "time" in da.dims:
                        da = da.isel(time=-1)
                    return float(da.values)
                except Exception:
                    return None

            sm_surface = _pick(vs) if vs else None
            sm_root = _pick(vr) if vr else None
            tiso = None
            try:
                if "time" in ds.dims:
                    t = ds["time"].values[-1]
                    tiso = str(np.datetime_as_string(t, unit='s'))
            except Exception:
                pass

            smap_part = {
                "surface_sm_m3m3": _round(sm_surface, 3),
                "rootzone_sm_m3m3": _round(sm_root, 3),
                "time_iso": tiso
            }
            ds.close()
        else:
            smap_part = {"error": "No SMAP L4 granules found in the last 24h for this location."}

    except Exception as e:
        smap_part = {"error": f"SMAP L4 access failed: {e}"}

    # ---- 3) MODIS Land Cover (MCD12Q1) to infer cropland/non-cropland
    modis_part = None
    try:
        import earthaccess, xarray as xr
        earthaccess.login(strategy="netrc")

        results = earthaccess.search_data(
            short_name="MCD12Q1",
            version="061",
            cloud_hosted=True,
            bounding_box=(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01),
            temporal=("2022-01-01", "2100-01-01")
        )
        if not results:
            modis_part = {"error": "No MODIS MCD12Q1 granule near this location."}
        else:
            urls = earthaccess.sign(results[0:1])
            ds = xr.open_dataset(urls[0])
            var = "LC_Type1" if "LC_Type1" in ds.variables else None
            if var is None:
                modis_part = {"error": "LC_Type1 not found in granule."}
            else:
                lat_name = "lat" if "lat" in ds.dims else [d for d in ds.dims if "lat" in d][0]
                lon_name = "lon" if "lon" in ds.dims else [d for d in ds.dims if "lon" in d][0]
                val = ds[var].sel(method="nearest", **{lat_name: float(lat), lon_name: float(lon)}).values
                v = int(val) if hasattr(val, "item") else int(val)
                label = _landcover_label_from_mcd12(v)
                ds_year = None
                try:
                    ds_year = int(str(ds.attrs.get("StartDate", ""))[0:4]) if "StartDate" in ds.attrs else None
                except Exception:
                    pass
                modis_part = {
                    "lc_type1": v,
                    "landcover_label": label,
                    "is_cropland": _is_cropland(v),
                    "year": ds_year
                }
            ds.close()
    except Exception as e:
        modis_part = {"error": f"MODIS access failed: {e}"}

    return jsonify({
        "ok": True,
        "power": power_part,
        "smap": smap_part,
        "modis": modis_part
    })

@app.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current = request.form.get("current") or ""
        new     = request.form.get("new") or ""
        confirm = request.form.get("confirm") or ""
        user = current_user.id

        if not check_password_hash(USER.get(user), current):
            return render_template("change_password.html", wrong="Current password is incorrect.")
        if not new:
            return render_template("change_password.html", wrong="New password cannot be empty.")
        if new != confirm:
            return render_template("change_password.html", wrong="New passwords do not match.")
        if check_password_hash(USER.get(user), new):
            return render_template("change_password.html", wrong="New password must be different.")

        USER[user] = generate_password_hash(new)
        return render_template("change_password.html", ok="Password updated successfully!")
    return render_template("change_password.html")

@app.route("/forget_password", methods=["GET", "POST"])
def forget_password():
    usr_name = ""
    found = False
    masked = None
    reset_ok = None
    error = None

    if request.method == "POST":
        action = request.form.get("action") or "lookup"

        if action == "lookup":
            usr_name = (request.form.get("usr_name") or "").strip()
            found = usr_name in USER
            if found:
                masked = "********"
            else:
                error = "No account exists with that username."

        elif action == "reset":
            usr_name = (request.form.get("usr_name") or "").strip()
            if usr_name not in USER:
                error = "User disappeared… please try again."
            else:
                new = request.form.get("new_password") or ""
                confirm = request.form.get("confirm_password") or ""
                if not new:
                    error = "New password cannot be empty."
                elif new != confirm:
                    error = "Passwords do not match."
                else:
                    USER[usr_name] = generate_password_hash(new)
                    reset_ok = "Password reset successfully! You can now log in."
                    found = True
                    masked = "********"

    return render_template(
        "forget_password.html",
        usr_name=usr_name,
        found=found,
        masked=masked,
        reset_ok=reset_ok,
        error=error,
    )

# -----------------------------
# LEVEL MAP (shared template for 1–5)
# -----------------------------

@app.route("/level_1", methods=["GET", "POST"])
@login_required
def level_1():
    return render_template("level_map.html", initial_level=1)

@app.route("/level_2", methods=["GET", "POST"])
@login_required
def level_2():
    return render_template("level_map.html", initial_level=2)

@app.route("/level_3", methods=["GET", "POST"])
@login_required
def level_3():
    return render_template("level_map.html", initial_level=3)

@app.route("/level_4", methods=["GET", "POST"])
@login_required
def level_4():
    return render_template("level_map.html", initial_level=4)

@app.route("/level_5", methods=["GET", "POST"])
@login_required
def level_5():
    return render_template("level_map.html", initial_level=5)

# -----------------------------
# Other pages
# -----------------------------

@app.route("/level_6", methods=["GET", "POST"])
@login_required
def level_6():
    return render_template("level_6.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    return render_template("feedback.html")

@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")

# -----------------------------
# Chatbot endpoint
# -----------------------------
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        user_msg = (data.get("message") or "").strip()
        if not user_msg:
            return jsonify({"response": "Please type a message."}), 400

        reply = hf_generate(user_msg)
        return jsonify({"response": reply}), 200

    return render_template("chatbot.html")

@app.route("/show")
def show():
    return render_template(
        "show.html",
        effective_rainfall="(demo data)",
        field_capacity="(demo data)",
        soil_mostureis="(demo data)",
        crop_evapotranspiration="(demo data)"
    )

# -----------------------------
# Game API (Level + Watering Logic + Progress HUD)
# -----------------------------

@app.route("/api/state", methods=["GET"])
@login_required
def api_state():
    game = ensure_game_session(1)
    progress = get_progress(current_user.id)
    return jsonify({
        "ok": True,
        "currentLevel": game["currentLevel"],
        "state": game["state"],
        "progress": progress
    })

@app.route("/api/set_level", methods=["POST"])
@login_required
def api_set_level():
    data = request.get_json(silent=True) or {}
    level = int(data.get("level") or 1)

    game = session.get("game")
    if not game or game.get("currentLevel") != level:
        game = {
            "currentLevel": level,
            "state": gen_level_state(level),
        }
        session["game"] = game

    progress = get_progress(current_user.id)
    return jsonify({
        "ok": True,
        "currentLevel": game["currentLevel"],
        "state": game["state"],
        "progress": progress
    })

@app.route("/api/progress", methods=["GET"])
@login_required
def api_progress():
    progress = get_progress(current_user.id)
    return jsonify({"ok": True, "progress": progress})

@app.route("/api/reset_progress", methods=["POST"])
@login_required
def api_reset_progress():
    with closing(db_connect()) as conn, conn:
        conn.execute("""
            UPDATE progress
               SET best_level=1, total_attempts=0, total_correct=0, current_streak=0, best_streak=0
             WHERE username=?
        """, (current_user.id,))
    return jsonify({"ok": True})

@app.route("/api/evaluate", methods=["POST"])
@login_required
def api_evaluate():
    """
    Evaluate the player's watering decision.
    Input: {"waterAdded": <number>} in percentage points.
    Output:
      - decision: "Correct" | "Too little" | "Overwatered"
      - feedback: { soilType, oldMoisture, waterAdded, newMoisture, capacity, idealRange, forecastRain, crop }
      - advance: bool
      - nextState (if advance)
      - progress: updated HUD
    """
    game = ensure_game_session(1)
    state = game["state"]
    data = request.get_json(silent=True) or {}

    try:
        water_added = float(data.get("waterAdded", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "waterAdded must be a number."}), 400

    old_m = float(state["moisture"])
    cap = float(state["capacity"])
    soil = state["soilType"]
    crop = state["crop"]
    lo, hi = float(state["idealLow"]), float(state["idealHigh"])
    rain = bool(state["forecastRain"])

    new_m = clamp(old_m + water_added, 0, cap)

    adjusted_hi = min(hi, cap)
    if rain:
        adjusted_hi = max(lo, adjusted_hi - RAIN_BUFFER)

    if new_m < lo:
        decision = "Too little"
    elif new_m <= adjusted_hi:
        decision = "Correct"
    else:
        decision = "Overwatered"

    feedback = {
        "soilType": soil,
        "crop": crop,
        "oldMoisture": round(old_m, 1),
        "waterAdded": round(water_added, 1),
        "newMoisture": round(new_m, 1),
        "capacity": int(cap),
        "idealRange": [int(lo), int(hi)],
        "forecastRain": rain,
        "decision": decision,
    }

    if decision == "Correct":
        next_level = int(game["currentLevel"]) + 1
        game["currentLevel"] = next_level
        game["state"] = gen_level_state(next_level)
        session["game"] = game

        update_progress_on_attempt(current_user.id, correct=True, new_level_reached=next_level)
        progress = get_progress(current_user.id)

        return jsonify({
            "ok": True,
            "decision": decision,
            "feedback": feedback,
            "advance": True,
            "currentLevel": next_level,
            "nextState": game["state"],
            "progress": progress
        })

    update_progress_on_attempt(current_user.id, correct=False, new_level_reached=None)
    progress = get_progress(current_user.id)

    return jsonify({
        "ok": True,
        "decision": decision,
        "feedback": feedback,
        "advance": False,
        "currentLevel": game["currentLevel"],
        "progress": progress
    })

if __name__ == "__main__":
    app.run(debug=True)
