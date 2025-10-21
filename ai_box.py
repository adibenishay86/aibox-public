import os
import sys
import time
import threading
import traceback
import locale
import platform
import time as pytime
from datetime import datetime
import speech_recognition as sr
from gtts import gTTS
import subprocess
from flask import Flask, request, jsonify
import logging
import google.genai as genai
from google.genai import types
import requests
import re
import ST7789
from PIL import Image, ImageDraw, ImageFont

# ====== CONFIGURATION =======
LANGUAGES = ['he-IL', 'en-US']  # Hebrew first, then English fallback
TTS_LANG = 'iw'
USE_OPENAI = False
SCRIPT_UPDATE_URL = 'https://github.com/adibenishay86/PyCharmMiscProject/blob/main/ai_box.py'
VERSION_URL = 'https://github.com/adibenishay86/PyCharmMiscProject/blob/main/version.txt'

LOCAL_VERSION = "1.0.28"
UPDATE_CHECK_INTERVAL = 300
SESSION_EXPIRE = 300
REST_API_PORT = 5000
LOG_FILENAME = "ai_box.log"
MAX_CONTEXT_TURNS = 60
BUTTON_POLL_INTERVAL = 0.1  # seconds
# ============================

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Debugging environment variables
logging.info(f"GOOGLE_API_KEY from environment: {os.getenv('GOOGLE_API_KEY')}")
logging.info(f"GITHUB_TOKEN from environment: {os.getenv('GITHUB_TOKEN')}")

# Initialize Google Gemini client
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY is not set. Exiting.")
    sys.exit(1)
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
grounding_tool = types.Tool(google_search=types.GoogleSearch())
generate_config = types.GenerateContentConfig(tools=[grounding_tool])

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

        version_raw_url = 'https://raw.githubusercontent.com/adibenishay86/PyCharmMiscProject/main/version.txt'
        script_raw_url = 'https://raw.githubusercontent.com/adibenishay86/PyCharmMiscProject/main/ai_box.py'

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


def query_google_ai(text, used_lang):
    global session_context
    try:
        system_context_text = get_system_context_message()
        if not session_context:
            session_context = []
        full_context = [
            {"parts": [{"text": "system context and previous questions context is: "+system_context_text}], "role": "user"}
        ]
        full_context.extend(session_context)
        full_context.append({
            "parts": [
                {"text": (
                    "The following message is the current user query to answer. "
                    "Please answer it directly in the same language : "+used_lang +
                    ". Use the previous text as context."
                    + "Please respond with plain text suitable for text-to-speech synthesis. Avoid special characters, emojis, or formatting"
                    + "use only textual characters and numbers, no asterisks or bullets"
                )}
            ],
            "role": "user"
        })
        full_context.append({"parts": [{"text": "the question is: " + text.strip()}], "role": "user"})
        if len(full_context) > MAX_CONTEXT_TURNS:
            full_context = full_context[-MAX_CONTEXT_TURNS:]
        logging.info("about to query Google AI with the following context: %s", full_context)
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_context,
            config=generate_config
        )
        answer = response.text.strip()
        new_session_context = []
        for entry in full_context:
            text_parts = entry["parts"][0]["text"].lower()
            if (
                "the following message is the current user query to answer" in text_parts
            ):
                continue
            if text_parts.startswith(
                "system context and previous questions context is:"
            ):
                continue
            if entry["role"] == "user":
                new_session_context.append(entry)
            elif entry["role"] == "model":
                new_session_context.append(entry)
        new_session_context.append({"parts": [{"text": answer}], "role": "model"})
        session_context = new_session_context[-MAX_CONTEXT_TURNS:]
        logging.info(
            f"Google Gemini AI Query cleaned session saving: \"{text}\" | Answer: \"{answer}\""
        )
        return answer, session_context
    except Exception as e:
        log_error("Google Gemini AI SDK with grounding (explicit instruction)", e)
        return "There was an error with Google Gemini AI service.", session_context


def query_ai(text, used_lang):
    if USE_OPENAI:
        # OpenAI branch placeholder
        pass
    else:
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
        user_lang = 'he-IL'
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
