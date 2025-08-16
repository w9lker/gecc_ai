import base64
import json
import time
import traceback

import requests
import streamlit as st
from google import genai
from google.cloud import firestore
from google.oauth2 import service_account
import google.auth

# --- HELPER FUNCTIONS  ---
PROJECT_ID = st.secrets["lyria"]["project_id"]
TEXT_GENERATION_PROMPT = """
    Generate a short reading passage for a focus test, and provide 3 comprehension questions.
    Return your response strictly as JSON (give me a json in a format it is stored in the file, don't give markdown as output)
    - "generated_text": the passage as a string,
    - "questions": a list of objects, each with "text" (the question) and "correct_response" ("Yes" or "No").
"""
LYRIA_MODEL = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/lyria-002:predict"


def load_test_from_gemini(max_retries=7, retry_delay=5):
    """
    Calls Gemini to generate a reading passage and questions.
    Retries on rate limit errors.
    Returns:
        tuple: (generated_text: str, questions: list of dicts)
    """
    with st.spinner("Loading test from Gemini..."):
        for attempt in range(1, max_retries + 1):
            try:
                client = genai.Client(api_key=st.secrets["gemini"]["api_key"])

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=TEXT_GENERATION_PROMPT,
                ).text

                # Parse the JSON from Gemini's response
                data = json.loads(response)
                generated_text = data["generated_text"]
                questions = data["questions"]
                return generated_text, questions
            except Exception as e:
                error_msg = str(e)
                if (
                    "rate limit" in error_msg.lower() or "429" in error_msg
                ) and attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    st.error("Rate limit reached. Maximum retries exceeded.")
                    return "Error loading test (rate limit).", []
                else:
                    st.error(f"Failed to load test from Gemini: {e}")
                    return "Error loading test.", []


def load_music(prompt: str):
    """
    Dummy function to simulate generating or fetching music.
    In your real app, this could call a music generation API.
    For this demo, it generates a 5-second silent WAV file.
    Args:
        prompt (str): The user's music preference (e.g., "Classical").
    Returns:
        bytes: The audio data in bytes.
    """
    st.toast("Generating music based on: prompt...")

    def send_request_to_google_api(api_endpoint, data=None):
        """
        Sends an HTTP request to a Google API endpoint.

        Args:
            api_endpoint: The URL of the Google API endpoint.
            data: (Optional) Dictionary of data to send in the request body (for POST, PUT, etc.).

        Returns:
            The response from the Google API.
        """

        # Get access token calling API
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]

        # 2. LOAD CREDENTIALS WITH THE SPECIFIED SCOPE
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["lyria"],
            scopes=scopes,  # <-- THE FIX IS HERE
        )
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        access_token = creds.token

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        print(api_endpoint)
        response = requests.post(api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def generate_music(request: dict):
        req = {"instances": [request], "parameters": {}}
        print(req)
        resp = send_request_to_google_api(LYRIA_MODEL, req)
        return resp["predictions"]

    pred = generate_music({"prompt": prompt})[0]
    bytes_b64 = dict(pred)["bytesBase64Encoded"]
    decoded_audio_data = base64.b64decode(bytes_b64)
    st.audio(decoded_audio_data, autoplay=True)


def submit_to_firestore(data: dict):
    """
    Function to submit the final collected data to Google Cloud Firestore.
    This function requires authentication to be set up.
    """
    try:
        # This is how you would initialize with secrets

        creds = service_account.Credentials.from_service_account_info(
            st.secrets["firestore"]
        )
        db = firestore.Client(
            credentials=creds, project=st.secrets["firestore"]["project_id"]
        )

        collection_ref = db.collection("user_responses")
        collection_ref.add(data)
        st.success("Your response were saved. Thank you 👻")

        return True
    except Exception:
        st.error(f"Failed to submit to Firestore:\n{traceback.format_exc()}")
        return False


# --- STATE INITIALIZATION ---

# Use session_state to store data across reruns and pages
if "page_number" not in st.session_state:
    st.session_state.page_number = 1
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "test_answers" not in st.session_state:
    st.session_state.test_answers = {}


# --- PAGE RENDERING FUNCTIONS ---


def render_page_1():
    """Renders the initial user information gathering page."""
    st.header("Welcome! Let's get to know you.")

    # Use a form to batch inputs together
    with st.form("user_info_form"):
        music_style = st.text_input(
            "What is your favourite music style?",
            placeholder="e.g., Classical, Lo-fi, Rock",
        )
        music_while_studying = st.radio(
            "Do you listen to music while studying?", ("Yes", "No"), horizontal=True
        )
        music_volume = st.select_slider(
            "Do you prefer quiet or loud music?", options=["Quiet", "Moderate", "Loud"]
        )

        submitted = st.form_submit_button("Submit")

        if submitted:
            # Save the data to the session_state dictionary
            st.session_state.user_info = {
                "favourite_music_style": music_style,
                "music_while_studying": music_while_studying,
                "preferred_volume": music_volume,
            }
            # Move to the next page
            st.session_state.page_number = 2
            st.rerun()  # Use rerun to immediately show the next page


def render_test_page(page_num: int, with_music: bool):
    """A generic function to render a test page."""
    st.header(f"Test Section {page_num - 1}")

    # Use session_state to cache test content per page
    test_key = f"test_content_page_{page_num}"
    if test_key not in st.session_state:
        test_text, question_obj_list = load_test_from_gemini()
        st.session_state[test_key] = (test_text, question_obj_list)
    else:
        test_text, question_obj_list = st.session_state[test_key]

    st.markdown(test_text)

    if with_music:
        # Use the user's preference from the first page as a prompt
        music_prompt = st.session_state.user_info.get("favourite_music_style", "calm")
        load_music(str(music_prompt))

    st.divider()

    # Store answers in a dictionary for this page
    page_answers = {}
    for i, question_obj in enumerate(question_obj_list):
        # The key for each widget must be unique across the entire app
        q = question_obj["text"]
        page_answers[q] = st.radio(
            q, ("Yes", "No"), key=f"p{page_num}_q{i}", horizontal=True
        )

    if st.button("Next", key=f"next_p{page_num}"):
        # evaluate the test answers
        correct_count = 0
        for question, correct_response in [
            (question_obj["text"], question_obj["correct_response"])
            for question_obj in question_obj_list
        ]:
            if page_answers[question] == correct_response:
                correct_count += 1
        page_answers["correct_count"] = correct_count

        # Save this page's answers to the main state
        st.session_state.test_answers[f"page_{page_num}"] = page_answers

        # Increment page number and rerun
        st.session_state.page_number += 1
        st.rerun()


def render_final_page():
    """Renders the final thank you and submission page."""
    st.header("Thank You!")
    st.markdown("You have completed all sections. You can review your data below.")

    # Display all collected data for user review
    st.subheader("Your Preferences")
    st.json(st.session_state.user_info)

    st.subheader("Your Test Answers")
    st.json(st.session_state.test_answers)

    st.divider()

    if st.button("Submit Final Data", type="primary"):
        # Combine all data into one dictionary for submission
        final_data = {
            "user_info": st.session_state.user_info,
            "test_answers": st.session_state.test_answers,
        }

        with st.spinner("Submitting your data to the database..."):
            success = submit_to_firestore(final_data)
            if success:
                st.success(
                    "Your submission was successful! Thank you for participating."
                )
                st.balloons()
                # Optionally clear the state after submission
                for key in st.session_state.keys():
                    del st.session_state[key]


# --- MAIN APP ROUTER ---
# kdkfj
# jfkdjfdkj
st.title("Interactive Music & Focus Study")

# This acts as a simple router based on the page number in the session state
page = st.session_state.page_number

if page == 1:
    render_page_1()
elif page == 2:
    render_test_page(page_num=2, with_music=False)
elif page == 3:
    render_test_page(page_num=3, with_music=True)
elif page == 4:
    render_test_page(page_num=4, with_music=True)
elif page == 5:
    render_final_page()
else:
    st.warning("Something went wrong. Please refresh.")
    # Add a button to reset the session
    if st.button("Start Over"):
        st.session_state.page_number = 1
        st.rerun()
