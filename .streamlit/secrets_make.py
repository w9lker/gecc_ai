import json
from pathlib import Path

# 1) Point this to your downloaded service-account JSON file
SERVICE_ACCOUNT_JSON = r"D:/Архив Final Version\Desktop/CU/Year 4/CC FYP/capable-mind-468913-f0-ba62e15eec48.json"

# 2) Put your Gemini API key here (or paste later into the output file)
GEMINI_API_KEY = "AIzaSyDowDpOo6qdumX3ZFXyF4mARze1IcWWs0o"

# Load JSON
with open(SERVICE_ACCOUNT_JSON, "r", encoding="utf-8") as f:
    svc = json.load(f)

# Normalize private key to a single-line TOML string with literal \n
pk = svc["private_key"].replace("\r\n", "\n").replace("\n", r"\n")

# Optional fields
universe_domain = svc.get("universe_domain", "googleapis.com")

toml = f"""[gcp]
project_id = "{svc['project_id']}"

[gemini]
api_key = "{GEMINI_API_KEY}"

[firestore]
type = "{svc['type']}"
project_id = "{svc['project_id']}"
private_key_id = "{svc['private_key_id']}"
private_key = "-----BEGIN PRIVATE KEY-----\\n{pk.split('-----BEGIN PRIVATE KEY-----\\n')[-1]}"
client_email = "{svc['client_email']}"
client_id = "{svc['client_id']}"
auth_uri = "{svc['auth_uri']}"
token_uri = "{svc['token_uri']}"
auth_provider_x509_cert_url = "{svc['auth_provider_x509_cert_url']}"
client_x509_cert_url = "{svc['client_x509_cert_url']}"
universe_domain = "{universe_domain}"
"""

out = Path(__file__).with_name("secrets.toml")
out.write_text(toml, encoding="utf-8")
print(f"Wrote: {out.resolve()}")
