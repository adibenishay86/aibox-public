import os
import sys
import time
import threading
import traceback
import locale
import platform
import socket
import time as pytime
from datetime import datetime
import speech_recognition as sr
from gtts import gTTS
import subprocess
from flask import Flask, request, jsonify
import logging
try:
    import google.genai as genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
import requests
import re
import ST7789
from PIL import Image, ImageDraw, ImageFont

# ====== CONFIGURATION =======
LANGUAGES = ['he-IL', 'en-US']  # Hebrew first, then English fallback
TTS_LANG = 'iw'
USE_OPENAI = False
SCRIPT_UPDATE_URL = 'https://github.com/adibenishay86/aibox-public/blob/main/ai_box.py'
VERSION_URL = 'https://github.com/adibenishay86/aibox-public/blob/main/version.txt'

LOCAL_VERSION = "1.0.37"
UPDATE_CHECK_INTERVAL = 300
SESSION_EXPIRE = 300
REST_API_PORT = 5000
LOG_FILENAME = "ai_box.log"
MAX_CONTEXT_TURNS = 60
BUTTON_POLL_INTERVAL = 0.1  # seconds

# Models ranked from best to worst quality for automatic fallback
MODEL_PRIORITY = [
    "Gemini 3.1 Pro (High)",
    "Gemini 3.1 Pro (Low)",
    "Claude Opus 4.6 (Thinking)",
    "Claude Sonnet 4.6 (Thinking)",
    "GPT-OSS 120B (Medium)",
    "Gemini 3.5 Flash (High)",
    "Gemini 3.5 Flash (Medium)",
    "Gemini 3.5 Flash (Low)",
]
QUOTA_CACHE_TTL = 3600  # Skip exhausted models for 1 hour
# ============================

# Cache of exhausted models: {model_name: timestamp_when_exhausted}
exhausted_models = {}

logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1
session_context = None
last_interaction = time.time()
last_update_check = 0
tts_process = None
listening = False

app = Flask(__name__)

# Explicitly load environment variables from /etc/environment if not already set
env_file = "/etc/environment"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if not os.getenv(key):  # Only set if not already in the environment
                    os.environ[key] = value
GOOGLE_API_KEY = "AIzaSyDRbvvpXAd6AcYZrVcbzLRI26zNcBSjqa8"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_API_URL = f"{GROQ_API_BASE}/responses"

# Debugging environment variables
logging.info(f"GOOGLE_API_KEY from environment: {os.getenv('GOOGLE_API_KEY')}")
logging.info(f"GITHUB_TOKEN from environment: {os.getenv('GITHUB_TOKEN')}")
logging.info(f"GROQ_API_KEY is set via environment: {bool(os.getenv('GROQ_API_KEY'))}")

# Initialize Google Gemini client (optional fallback / not used for agy CLI)
genai_client = None
grounding_tool = None
generate_config = None
if genai and types:
    try:
        if GOOGLE_API_KEY.startswith("otk_"):
            logging.warning("Using injected one-time placeholder GOOGLE_API_KEY. This is NOT a valid Google Cloud API key.")
        else:
            genai_client = genai.Client(api_key=GOOGLE_API_KEY)
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            generate_config = types.GenerateContentConfig(tools=[grounding_tool])
    except Exception as e:
        logging.warning(f"Could not initialize genai client: {e}")

# Initialize display
disp = ST7789.ST7789()
disp.Init()
disp.clear()
disp.bl_DutyCycle(50)

image1 = Image.new("RGB", (disp.width, disp.height), "WHITE")
draw = ImageDraw.Draw(image1)
prev_button_states = {
    'UP': 0,
    'LEFT': 0,
    'RIGHT': 0,
    'DOWN': 0,
    'CENTER': 0,
    'KEY1': 0,
    'KEY2': 0,
    'KEY3': 0,
}
try:
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    font = ImageFont.truetype(font_path, 32)
except Exception as e:
    logging.warning(f"Failed to load font, fallback: {e}")
    font = ImageFont.load_default()


def log_error(where, e):
    logging.error(f"Error in {where}: {e}")
    traceback.print_exc()
def check_for_update():
    global last_update_check
    now = time.time()
    if now - last_update_check < UPDATE_CHECK_INTERVAL:
        return
    try:
        headers = {}
        if GITHUB_TOKEN:
            headers = {"Authorization": f"token {GITHUB_TOKEN}"}

        version_raw_url = 'https://raw.githubusercontent.com/adibenishay86/aibox-public/main/version.txt'
        script_raw_url = 'https://raw.githubusercontent.com/adibenishay86/aibox-public/main/ai_box.py'

        remote_version_resp = requests.get(version_raw_url, headers=headers, timeout=5)
        remote_version_resp.raise_for_status()
        remote_version = remote_version_resp.text.strip()
        logging.info(f"Remote version: {remote_version}")
        logging.info(f"Local version: {LOCAL_VERSION}")
        if remote_version != LOCAL_VERSION:
            logging.info("New version found! Downloading update...")

            script_resp = requests.get(script_raw_url, headers=headers, timeout=10)
            script_resp.raise_for_status()
            new_code = script_resp.text

            script_path = os.path.abspath(__file__)
            backup_path = script_path + ".backup"

            if os.path.exists(script_path):
                os.replace(script_path, backup_path)
                logging.info(f"Backup of current script saved as {backup_path}")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(new_code)

            logging.info("Update applied. Exiting to let systemd restart...")
            os._exit(0)
            logging.info("process couldn't exit")
        else:
            logging.info("No update needed.")
    except Exception as e:
        log_error("update check", e)
        logging.info("Update skipped due to error.")
    last_update_check = now

def recognize_multilang(audio):
    texts = []
    for lang in LANGUAGES:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text.strip():
                logging.info(f"Recognized ({lang}): {text}")
                texts.append(text)
        except sr.UnknownValueError:
            continue
        except Exception as e:
            log_error("speech recognition", e)
            break

    if not texts:
        return "", None

    chosen_text = texts[0]
    detected_lang = detect_language_from_text(chosen_text)
    logging.info(f"Detected language from text heuristic: {detected_lang}")
    return chosen_text, detected_lang


# Attempt to load a TTF font that supports Hebrew
try:
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"  # adjust if needed
    font = ImageFont.truetype(font_path, 32)
except Exception as e:
    logging.warning(f"Failed to load TTF font, fallback to default font: {e}")
    font = None

def clear_display():
    disp.clear()

def display_text(text, fill=(0, 0, 0)):
    image1.paste((255, 255, 255), [0, 0, disp.width, disp.height])
    draw.rectangle((0, 0, disp.width, disp.height), fill=(255, 255, 255))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except AttributeError:
        w, h = draw.textsize(text, font=font)
    x = (disp.width - w) // 2
    y = (disp.height - h) // 2
    draw.text((x, y), text, font=font, fill=fill)
    disp.ShowImage(image1)


def display_colored_text(user_text, ai_text, user_color=(0, 0, 255), ai_color=(255, 0, 0), font_size=20):
    font_small = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    max_width = disp.width - 20  # horizontal padding

    def wrap_text(text, font, max_w):
        words = text.split()
        lines = []
        current_line = ''
        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            if w <= max_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    # Clear screen
    image1.paste((255, 255, 255), [0, 0, disp.width, disp.height])
    draw.rectangle((0, 0, disp.width, disp.height), fill=(255, 255, 255))

    y = 10
    line_spacing = 4

    # Draw user text wrapped lines
    user_lines = wrap_text(user_text, font_small, max_width)
    for line in user_lines:
        bbox = draw.textbbox((0, 0), line, font=font_small)
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        x = (disp.width - w) // 2
        draw.text((x, y), line, font=font_small, fill=user_color)
        y += h + line_spacing

    y += 8  # extra vertical gap

    # Draw AI text wrapped lines
    ai_lines = wrap_text(ai_text, font_small, max_width)
    for line in ai_lines:
        bbox = draw.textbbox((0, 0), line, font=font_small)
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        x = (disp.width - w) // 2
        draw.text((x, y), line, font=font_small, fill=ai_color)
        y += h + line_spacing

    disp.ShowImage(image1)





import threading

def listen_in_background(result_container):
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source,timeout=2)
        result_container['text'], result_container['used_lang'] = recognize_multilang(audio)
    except Exception as e:
        log_error("speech recognition background", e)
        result_container['text'], result_container['used_lang'] = "", LANGUAGES[0]

def listen_for_command():
    global listening
    listening = True
    try:
        logging.info("Listening for speech...")

        result_container = {}
        t = threading.Thread(target=listen_in_background, args=(result_container,))
        t.start()

        # Countdown while recognizer listens in parallel
        for i in range(1, -1, -1):
            display_text(f":מאזין בעוד {i}", fill=(0, 0, 0))
            time.sleep(1)
        display_text(f":מאזין ", fill=(0, 0, 0))

        t.join()  # Wait for recognition to complete

        text = result_container.get('text', "")
        used_lang = result_container.get('used_lang', LANGUAGES[0])
        logging.info(f"Speech recognized: {text} (lang: {used_lang})")
    except Exception as e:
        log_error("speech recognition", e)
        text, used_lang = "", LANGUAGES[0]

    listening = False
    disp.clear()
    return text, used_lang



def recognize_multilang(audio):
    texts = []
    for lang in LANGUAGES:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text.strip():
                logging.info(f"Recognized ({lang}): {text}")
                texts.append(text)
        except sr.UnknownValueError:
            continue
        except Exception as e:
            log_error("speech recognition", e)
            break
    if not texts:
        return "", None
    chosen_text = texts[0]
    detected_lang = detect_language_from_text(chosen_text)
    logging.info(f"Detected language from text heuristic: {detected_lang}")
    return chosen_text, detected_lang


def detect_language_from_text(text):
    if re.search(r"[\u0590-\u05FF]", text):
        return "he-IL"
    else:
        return "en-US"


def get_system_context_message():
    try:
        locale_str, _ = locale.getlocale()
        if locale_str is None:
            locale_str = "he-IL"
    except Exception:
        locale_str = "he-IL"
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    tz_name = pytime.tzname[pytime.localtime().tm_isdst]
    device_info = "Raspberry Pi voice assistant"
    location = "Yavne, Israel"
    format_pref = "Prefers 24-hour time, DD/MM/YYYY dates."
    context_msg = (
        f"Environment: Locale={locale_str}; Time={local_time} {tz_name}; "
        f"Location={location}; Device={device_info}; {format_pref}"
    )
    return context_msg


def get_diagnostics():
    groq_enabled = bool(GROQ_API_KEY)
    agy_path = "/home/rnela/.local/bin/agy"
    agy_available = os.path.exists(agy_path)
    try:
        groq_connectivity = "reachable"
        with socket.create_connection(("api.groq.com", 443), timeout=5):
            pass
    except Exception as e:
        groq_connectivity = f"unreachable: {e}"

    diag = {
        "local_version": LOCAL_VERSION,
        "python_version": platform.python_version(),
        "flask_port": REST_API_PORT,
        "use_openai": USE_OPENAI,
        "groq_api_key_present": groq_enabled,
        "groq_model": GROQ_MODEL,
        "groq_api_url": GROQ_API_URL,
        "groq_connectivity": groq_connectivity,
        "agy_path": agy_path,
        "agy_available": agy_available,
        "model_priority": MODEL_PRIORITY,
        "last_interaction": last_interaction,
        "session_context_turns": len(session_context) if session_context else 0,
        "languages": LANGUAGES,
        "tts_lang": TTS_LANG,
        "button_poll_interval": BUTTON_POLL_INTERVAL,
        "update_check_interval": UPDATE_CHECK_INTERVAL,
        "google_api_key_present": bool(GOOGLE_API_KEY),
    }
    return diag


def query_groq_ai(text, used_lang):
    global session_context
    if not GROQ_API_KEY:
        logging.warning("Groq Cloud API key is missing, skipping Groq query.")
        return None, session_context, False

    try:
        use_continue = (session_context is not None and len(session_context) > 0)
        language_label = "Hebrew" if used_lang.startswith("he") else "English"
        
        # Build comprehensive grounding context
        grounding_context = get_system_context_message()
        
        # Check if query is asking about weather and fetch real-time data
        if any(word in text.lower() for word in ["weather", "מזג אוויר", "טמפרטורה", "מזג", "אוויר", "תחזוקה", "סערה", "גשם"]):
            try:
                # Use Open-Meteo free weather API for Yavne, Israel
                import requests as req
                weather_response = req.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": 31.93,
                        "longitude": 34.76,
                        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                        "temperature_unit": "celsius",
                        "timezone": "Asia/Jerusalem"
                    },
                    timeout=5
                )
                if weather_response.status_code == 200:
                    weather_data = weather_response.json().get("current", {})
                    temp = weather_data.get("temperature_2m", "unknown")
                    humidity = weather_data.get("relative_humidity_2m", "unknown")
                    wind = weather_data.get("wind_speed_10m", "unknown")
                    weather_code = weather_data.get("weather_code", 0)
                    
                    # Simple weather code to text mapping
                    weather_descriptions = {
                        0: "clear", 1: "partly cloudy", 2: "mostly cloudy", 3: "overcast",
                        45: "foggy", 48: "foggy", 51: "drizzle", 53: "drizzle", 55: "drizzle",
                        61: "rain", 63: "rain", 65: "heavy rain", 71: "snow", 73: "snow", 75: "heavy snow",
                        77: "snow", 80: "showers", 81: "showers", 82: "heavy showers", 85: "snow showers",
                        86: "snow showers", 95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm"
                    }
                    condition = weather_descriptions.get(weather_code, "varied")
                    weather_info = f"Current weather: {temp}°C, {humidity}% humidity, {wind} km/h wind, {condition}."
                    grounding_context += f" {weather_info}"
                    logging.info(f"Weather API result: {weather_info}")
            except Exception as e:
                logging.warning(f"Weather API fetch failed: {e}")
        
        # Build prompt with grounding context
        prompt = (
            f"System: You are a voice assistant. Respond in {language_label} only. "
            f"Give a direct, concise answer. No reasoning, no explanations. "
            f"Context: {grounding_context}\n"
            f"User: {text.strip()}\n"
            f"Assistant:"
        )

        payload = {
            "model": GROQ_MODEL,
            "input": prompt,
            "temperature": 0.0,
            "top_p": 0.0,
            "max_output_tokens": 128,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        logging.info(f"Querying Groq Cloud URL {GROQ_API_URL} as first priority")
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            logging.warning(f"Groq Cloud request failed: {response.status_code} {response.text}")
            return None, session_context, False

        result = response.json()
        output = result.get("output")
        if not output:
            logging.warning(f"Groq Cloud response missing output: {result}")
            return None, session_context, False

        answer_parts = []
        for item in output:
            if isinstance(item, dict):
                for content in item.get("content", []):
                    if isinstance(content, dict) and content.get("type") in ("output_text", "reasoning_text"):
                        text = content.get("text")
                        if text:
                            answer_parts.append(text.strip())
                    elif isinstance(content, str):
                        answer_parts.append(content.strip())
        answer = "\n".join([part for part in answer_parts if part])
        if not answer:
            # Last resort: try OpenAI-style chat completion format
            choices = result.get("choices") or []
            for choice in choices:
                message = choice.get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    answer_parts.append(content.strip())
            answer = "\n".join([part for part in answer_parts if part])
        if not answer:
            logging.warning(f"Groq Cloud returned empty response data: {result}")
            return None, session_context, False

        if session_context is None:
            session_context = []
        session_context.append({"parts": [{"text": text}], "role": "user"})
        session_context.append({"parts": [{"text": answer}], "role": "model"})
        if len(session_context) > MAX_CONTEXT_TURNS:
            session_context = session_context[-MAX_CONTEXT_TURNS:]

        logging.info(f"Groq Cloud success. Answer: \"{answer}\"")
        return answer, session_context, True
    except Exception as e:
        log_error("Groq Cloud query failed", e)
        return None, session_context, False


def query_google_ai(text, used_lang):
    global session_context
    try:
        # Check if we should continue the conversation
        use_continue = (session_context is not None and len(session_context) > 0)
        
        # Build comprehensive grounding context
        grounding_context = get_system_context_message()
        
        # Prepare the query text with grounding
        language_label = "Hebrew" if used_lang.startswith("he") else "English"
        language_instruction = (
            f"You are a helpful assistant. Answer the user question directly in {language_label}. "
            f"Do not repeat the instructions, do not mention the prompt, and do not restate the question. "
            f"Use only plain text suitable for text-to-speech synthesis. Avoid special characters, emojis, or formatting. "
            f"Use only textual characters and numbers."
        )
        # If starting a new session, we can prefix it with system context
        if not use_continue:
            prompt = (
                f"Context: {grounding_context}\n"
                f"{language_instruction}\n\n"
                f"{text.strip()}"
            )
        else:
            prompt = f"{language_instruction}\n\n{text.strip()}"
        
        # Try each model in priority order, falling back on quota exhaustion
        now = time.time()
        for model_index, model_name in enumerate(MODEL_PRIORITY):
            # Skip models that are cached as exhausted
            if model_name in exhausted_models:
                cached_at = exhausted_models[model_name]
                if now - cached_at < QUOTA_CACHE_TTL:
                    remaining = int(QUOTA_CACHE_TTL - (now - cached_at))
                    logging.info(f"Skipping model '{model_name}' — quota cached as exhausted ({remaining}s remaining)")
                    continue
                else:
                    # Cache expired, remove and retry this model
                    del exhausted_models[model_name]
                    logging.info(f"Model '{model_name}' quota cache expired, retrying...")
            
            logging.info(f"Querying agy CLI with model '{model_name}' (continue={use_continue}, attempt {model_index + 1}/{len(MODEL_PRIORITY)}) prompt: '{prompt[:100]}...'")
            
            # Build command with model flag
            cmd = ["/home/rnela/.local/bin/agy", "--model", model_name, "--print", prompt]
            if use_continue:
                cmd.insert(1, "--continue")
                
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            combined_output = (res.stderr or "") + (res.stdout or "")
            
            # Check for quota exhaustion — cache and try next model
            if "RESOURCE_EXHAUSTED" in combined_output or "quota" in combined_output.lower():
                exhausted_models[model_name] = now
                logging.warning(f"Model '{model_name}' quota exhausted, cached for {QUOTA_CACHE_TTL}s. Trying next model...")
                continue
            
            if res.returncode != 0:
                err_msg = res.stderr.strip()
                logging.error(f"agy CLI returned error code {res.returncode} with model '{model_name}': {err_msg}")
                if "Authentication required" in combined_output:
                    return "Please run the command 'agy' in the terminal on the Raspberry Pi once to authenticate your Google account.", session_context
                # For non-quota errors, try the next model too
                logging.warning(f"Model '{model_name}' failed, trying next model...")
                continue
                
            answer = res.stdout.strip()
            
            # If answer is empty, the model might have silently failed — cache and try next
            if not answer:
                exhausted_models[model_name] = now
                logging.warning(f"Model '{model_name}' returned empty response, cached for {QUOTA_CACHE_TTL}s. Trying next model...")
                continue
            
            # Success — keep local session context for --continue
            if session_context is None:
                session_context = []
            session_context.append({"parts": [{"text": text}], "role": "user"})
            session_context.append({"parts": [{"text": answer}], "role": "model"})
            if len(session_context) > MAX_CONTEXT_TURNS:
                session_context = session_context[-MAX_CONTEXT_TURNS:]
                
            logging.info(f"agy CLI success with model '{model_name}'. Answer: \"{answer}\"")
            return answer, session_context
        
        # All models exhausted
        logging.error("All models exhausted or returned errors.")
        return "All AI models are currently unavailable. Please try again later.", session_context
    except Exception as e:
        log_error("agy CLI query failed", e)
        return "There was an error communicating with the local agy CLI.", session_context


def query_ai(text, used_lang):
    if USE_OPENAI:
        # OpenAI branch placeholder
        pass

    answer, session_context, groq_success = query_groq_ai(text, used_lang)
    if groq_success:
        return answer, session_context

    return query_google_ai(text, used_lang)


def speak_text(text, lang):
    global tts_process
    try:
        reply_path = "reply.mp3"
        if os.path.exists(reply_path):
            os.remove(reply_path)
        tts = gTTS(text=text, lang=(lang if lang != "he-IL" else "iw"), slow=False)
        tts.save(reply_path)

        def play():
            global tts_process
            try:
                tts_process = subprocess.Popen(["mpg123", reply_path])
                tts_process.wait()
            except Exception as e:
                log_error("TTS playback", e)

        thread = threading.Thread(target=play)
        thread.start()
        return thread
    except Exception as e:
        log_error("TTS generation", e)


def stop_tts():
    global tts_process
    try:
        if tts_process and tts_process.poll() is None:
            tts_process.terminate()
            logging.info("TTS playback stopped.")
    except Exception as e:
        log_error("stopping TTS", e)


def process_text_query(user_text, used_lang, source="unknown"):
    global last_interaction, session_context
    if time.time() - last_interaction > SESSION_EXPIRE:
        session_context = None
    last_interaction = time.time()
    if user_text:
        logging.info(f"Received query from {source}: {user_text}")
        answer, session_context = query_ai(user_text, used_lang)
        display_colored_text(user_text, answer, user_color=(0, 0, 255), ai_color=(255, 0, 0))
        tts_lang = "iw" if used_lang.startswith("he") else "en"
        speak_text(answer, tts_lang)
        logging.info(f"AI Response to {source}: {answer}")
        return answer
    else:
        logging.info(f"No text provided in {source} query.")
        return "No text provided!"


def button_pressed():
    logging.info("Button pressed")
    try:
        global last_interaction, session_context
        if tts_process and tts_process.poll() is None:
            stop_tts()
            last_interaction = time.time()
            logging.info("Stopped playback by button.")
            return
        user_text, used_lang = listen_for_command()
        process_text_query(user_text, used_lang, source="button")
    except Exception as e:
        log_error("button pressed handler", e)

# Track previous button states to detect new presses


def check_buttons_polling():
    global prev_button_states
    try:
        states = {
            'UP': disp.digital_read(disp.GPIO_KEY_UP_PIN),
            'LEFT': disp.digital_read(disp.GPIO_KEY_LEFT_PIN),
            'RIGHT': disp.digital_read(disp.GPIO_KEY_RIGHT_PIN),
            'DOWN': disp.digital_read(disp.GPIO_KEY_DOWN_PIN),
            'CENTER': disp.digital_read(disp.GPIO_KEY_PRESS_PIN),
            'KEY1': disp.digital_read(disp.GPIO_KEY1_PIN),
            'KEY2': disp.digital_read(disp.GPIO_KEY2_PIN),
            'KEY3': disp.digital_read(disp.GPIO_KEY3_PIN),
        }
        for btn, state in states.items():
            if state != prev_button_states.get(btn, 0) and state != 0:
                logging.info(f"{btn} button pressed (poll detection)")
                button_pressed()
            prev_button_states[btn] = state
    except Exception as e:
        log_error("button polling", e)


def button_polling_thread():
    while True:
        check_buttons_polling()
        time.sleep(BUTTON_POLL_INTERVAL)


def periodic_update_check():
    while True:
        logging.info(f"Checking for updates at time {time.time()}")
        try:
            check_for_update()
        except Exception as e:
            log_error("periodic update check", e)
        time.sleep(UPDATE_CHECK_INTERVAL)


@app.route('/query', methods=['POST'])
def rest_query():
    try:
        data = request.get_json(force=True)
        user_text = data.get('text', '')
        logging.info(f"Received query via REST: {user_text}")
        user_lang = detect_language_from_text(user_text) if user_text else 'en-US'
        if not user_lang:
            user_lang = 'en-US'
        answer = process_text_query(user_text, user_lang, source="REST")
        return jsonify({'answer': answer})
    except Exception as e:
        log_error("REST endpoint", e)
        return jsonify({'answer': "Error occurred!"}), 500


@app.route('/simulate_button', methods=['POST'])
def simulate_button():
    try:
        user_text, used_lang = listen_for_command()
        logging.info(f"Simulated button: recognized '{user_text}' lang={used_lang}")
        answer = process_text_query(user_text, used_lang, source="simulated_button")
        return jsonify({'answer': answer, 'input': user_text})
    except Exception as e:
        log_error("simulate_button", e)
        return jsonify({'answer': "Error occurred!"}), 500


@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    try:
        return jsonify(get_diagnostics())
    except Exception as e:
        log_error("diagnostics endpoint", e)
        return jsonify({'error': 'Diagnostics failed', 'details': str(e)}), 500


def run_flask():
    app.run(host='0.0.0.0', port=REST_API_PORT)


# Colored text display helper function (requires PIL)

# Start background threads
update_thread = threading.Thread(target=periodic_update_check, daemon=True)
update_thread.start()

button_thread = threading.Thread(target=button_polling_thread, daemon=True)
button_thread.start()

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

logging.info("AI Box initialized and ready.")
logging.info("Running version " + LOCAL_VERSION)
print("AI Box initialized and ready.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("Exiting by keyboard interrupt.")
    print("Exiting.")
except Exception as e:
    log_error("main loop", e)
finally:
    disp.module_exit()
