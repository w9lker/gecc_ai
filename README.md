# GECC AI — Music & Focus Study 🎵

This Streamlit app runs a study on how background music influences reading comprehension and focus.  
It integrates with **Google Vertex AI Lyria (lyria-002)** to generate background music,  
and with **Gemini** to generate reading passages and questions.  
Final responses are saved to **Firestore**.

---

## ✨ Features
- Interactive multi-page Streamlit app
- Gemini-powered reading comprehension passages
- Lyria-002 powered background music generation
- Personalized music prompts (genre, tempo, mood, instruments, etc.)
- Firestore integration for saving study results
- Graceful error handling & silent fallback audio if API calls fail

---

## ⚙️ Requirements
- Python 3.10+
- Google Cloud project with:
  - Vertex AI API enabled
  - Firestore enabled
  - Billing (free credits are fine)
  - Service account with roles:
    - **Vertex AI User** (`roles/aiplatform.user`)
    - **Cloud Datastore User** (`roles/datastore.user`)

---

## 🚀 Setup

```powershell
# Clone repo
git clone https://github.com/shaytanaraba/gecc_ai.git
cd gecc_ai

# Create & activate virtual environment
python -m venv ai_music
& ".\ai_music\Scripts\Activate.ps1"

# Install requirements
pip install -r requirements.txt
