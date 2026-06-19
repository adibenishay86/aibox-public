import http.server
import socketserver
import json
import subprocess
import os
import time
import sys

PORT = 5001
AGENTAPI_PATH = os.path.expanduser("~/.gemini/antigravity/bin/agentapi")
BRAIN_DIR = os.path.expanduser("~/.gemini/antigravity/brain")

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    # Store the active conversation ID at the class level
    conversation_id = None
    
    def log_message(self, format, *args):
        # Override to log to stdout/stderr instead of default stderr format
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format%args))
        sys.stdout.flush()

    def do_POST(self):
        if self.path == '/query':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                req = json.loads(post_data.decode('utf-8'))
            except Exception as e:
                self.send_error_response(400, f"Invalid JSON: {e}")
                return

            text = req.get('text', '').strip()
            reset = req.get('reset', False)
            
            if not text:
                self.send_error_response(400, "Missing 'text' parameter")
                return

            print(f"\n--- New Request: reset={reset} ---")
            print(f"User Query: {text}")

            # Start or retrieve conversation
            if reset or not ProxyHandler.conversation_id:
                print("Starting new conversation...")
                conv_id = self.start_new_conversation(text)
                if not conv_id:
                    self.send_error_response(500, "Failed to start new conversation")
                    return
                ProxyHandler.conversation_id = conv_id
                print(f"Created Conversation ID: {ProxyHandler.conversation_id}")
                # For new conversation, step 0 is the USER_INPUT. We poll for step > 0.
                last_step_index = 0
            else:
                print(f"Using existing Conversation ID: {ProxyHandler.conversation_id}")
                # For existing conversation, we first find the current max step_index in transcript
                last_step_index = self.get_max_step_index(ProxyHandler.conversation_id)
                print(f"Last step index before query: {last_step_index}")
                success = self.send_message(ProxyHandler.conversation_id, text)
                if not success:
                    print("Send-message failed, attempting to start new conversation instead...")
                    conv_id = self.start_new_conversation(text)
                    if not conv_id:
                        self.send_error_response(500, "Failed to send message or restart conversation")
                        return
                    ProxyHandler.conversation_id = conv_id
                    print(f"Created new Conversation ID: {ProxyHandler.conversation_id}")
                    last_step_index = 0

            # Poll for response
            response_text = self.poll_for_response(ProxyHandler.conversation_id, last_step_index)
            if response_text:
                print(f"AI Response: {response_text}")
                self.send_success_response({"answer": response_text})
            else:
                print("Timeout waiting for response from Antigravity agent")
                self.send_error_response(504, "Timeout waiting for agent response")
        else:
            self.send_error_response(404, "Not Found")

    def send_success_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))

    def start_new_conversation(self, prompt):
        try:
            # We request model flash or default, using agentapi new-conversation
            cmd = [AGENTAPI_PATH, "new-conversation", prompt]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            conv_id = data.get("response", {}).get("newConversation", {}).get("conversationId")
            return conv_id
        except Exception as e:
            print(f"Error in start_new_conversation: {e}")
            if 'res' in locals():
                print(f"Stderr: {res.stderr}")
                print(f"Stdout: {res.stdout}")
            return None

    def send_message(self, conv_id, content):
        try:
            cmd = [AGENTAPI_PATH, "send-message", conv_id, content]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"send-message command returned code {res.returncode}")
                print(f"Stderr: {res.stderr}")
                return False
            data = json.loads(res.stdout)
            if "error" in data:
                print(f"Error in send-message response: {data['error']}")
                return False
            return True
        except Exception as e:
            print(f"Exception in send_message: {e}")
            return False

    def get_max_step_index(self, conv_id):
        transcript_path = os.path.join(BRAIN_DIR, conv_id, ".system_generated", "logs", "transcript.jsonl")
        if not os.path.exists(transcript_path):
            return -1
        max_idx = -1
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        idx = data.get("step_index", -1)
                        if idx > max_idx:
                            max_idx = idx
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error getting max step index: {e}")
        return max_idx

    def poll_for_response(self, conv_id, last_step_index, timeout=60):
        transcript_path = os.path.join(BRAIN_DIR, conv_id, ".system_generated", "logs", "transcript.jsonl")
        start_time = time.time()
        
        print("Polling transcript.jsonl for response...")
        while time.time() - start_time < timeout:
            if os.path.exists(transcript_path):
                try:
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    # Search from bottom to top
                    for line in reversed(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            step_idx = data.get("step_index", -1)
                            if step_idx > last_step_index:
                                # We look for MODEL source and PLANNER_RESPONSE type
                                if data.get("source") == "MODEL" and data.get("type") == "PLANNER_RESPONSE":
                                    content = data.get("content", "")
                                    # Ensure it's not a tool-calling intermediate step
                                    if content and not data.get("tool_calls"):
                                        return content
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Error reading transcript while polling: {e}")
            
            time.sleep(0.5)
        return None

def run():
    # Make sure brain directory exists
    if not os.path.exists(BRAIN_DIR):
        print(f"Warning: Brain directory {BRAIN_DIR} not found.")
    
    server_address = ('', PORT)
    with socketserver.TCPServer(server_address, ProxyHandler) as httpd:
        print(f"Mac Proxy Server running on port {PORT}...")
        print(f"Ensure Raspberry Pi points to http://<mac_ip>:{PORT}/query")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down proxy server.")

if __name__ == '__main__':
    run()
