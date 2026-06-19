# Generative AI (Gemini) API Key — How to obtain and configure

I can't create Google Cloud API keys on your behalf. Follow these steps to create a key and configure the app to use Gemini (Generative AI):

1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create or select a project.
3. Enable the "Generative AI API" (sometimes named "Generative" or "Gemini API").
4. Open "APIs & Services" → "Credentials" → "Create Credentials" → "API key".
5. (Recommended) Restrict the key to the Generative AI API and by IP or HTTP referrer.
6. Copy the key string.

To configure the app locally:

- Put the key into `/etc/environment` as:

  GOOGLE_API_KEY=your_key_here
  GROQ_API_KEY=your_groq_key_here

- Or export it in your shell before running the app:

  export GOOGLE_API_KEY=your_key_here
  export GROQ_API_KEY=your_groq_key_here

- Or create a `.env` from `.env.example` and load it with your environment loader.

- Do not commit the Groq API key or any other secret into source control.

Programmatic key creation (optional helper)
-----------------------------------------
I included a helper script `create_google_api_key.py` that can create an API key for a project using a service account. It must be run locally and requires:

- A service account JSON key file with permissions to manage API keys in the project (roles/serviceusage.serviceUsageAdmin or similar).
- The `apikeys.googleapis.com` API enabled in the project.
- Python dependencies:

  pip install google-api-python-client google-auth

Example:

```bash
python create_google_api_key.py --sa ./service-account.json --project your-gcp-project-id --name "AI Box Gemini Key"
```

The script starts the key-creation operation and polls until it completes. If the created key string (`keyString`) is returned by the API it will be printed; otherwise check the Cloud Console to retrieve the key.

Security note: do not commit service account JSON files to source control.

