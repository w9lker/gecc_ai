import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import io

# --- DUMMY & HELPER FUNCTIONS (You will replace these) ---

def load_test_from_gemini():
    """
    Dummy function to simulate a call to Gemini.
    In your real app, this will make an API call.
    Returns:
        tuple: A tuple containing (str: generated_text, list: questions)
    """
    # This is a placeholder. Replace with your actual Gemini API call.
    generated_text = "Studies show that focus can be influenced by auditory stimuli. Some people find that certain types of music help them concentrate, while for others, silence is golden. The effect can depend on the complexity of the task and the individual's personality."
    questions = [
        "Do you agree that music can influence focus?",
        "Have you ever used music to help you concentrate?",
        "Do you believe silence is always better for complex tasks?"
    ]
    return generated_text, questions

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
    st.toast(f"Generating music based on: {prompt}...")
    # Create a silent audio track as a placeholder
    samplerate = 44100  # 44.1kHz
    duration = 5  # seconds
    # Generate a silent numpy array
    silent_array = np.zeros(int(samplerate * duration))
    
    # Use an in-memory bytes buffer
    buffer = io.BytesIO()
    write(buffer, samplerate, silent_array.astype(np.int16))
    
    # Return the bytes from the buffer
    return buffer.getvalue()

def submit_to_firestore(data: dict):
    """
    Function to submit the final collected data to Google Cloud Firestore.
    This function requires authentication to be set up.
    """
    try:
        # Use st.secrets for secure credential handling in deployed apps
        # In your secrets.toml:
        # [firestore]
        # type = "service_account"
        # project_id = "your-gcp-project-id"
        # ... (rest of your service account JSON)
        
        # This is how you would initialize with secrets
        # from google.oauth2 import service_account
        # from google.cloud import firestore
        # creds = service_account.Credentials.from_service_account_info(st.secrets["firestore"])
        # db = firestore.Client(credentials=creds, project=st.secrets["firestore"]["project_id"])
        
        # For now, we just print the data and simulate success
        st.write("---")
        st.write("`submit_to_firestore` called. In a real app, this would be sent to the database:")
        st.json(data)
        
        # Example of what the real code would look like:
        # collection_ref = db.collection("user_responses")
        # doc_ref = collection_ref.add(data)
        # st.success(f"Data successfully submitted to Firestore with Document ID: {doc_ref.id}")

        return True
    except Exception as e:
        st.error(f"Failed to submit to Firestore: {e}")
        return False

# --- STATE INITIALIZATION ---

# Use session_state to store data across reruns and pages
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'test_answers' not in st.session_state:
    st.session_state.test_answers = {}


# --- PAGE RENDERING FUNCTIONS ---

def render_page_1():
    """Renders the initial user information gathering page."""
    st.header("Welcome! Let's get to know you.")
    
    # Use a form to batch inputs together
    with st.form("user_info_form"):
        music_style = st.text_input("What is your favourite music style?", placeholder="e.g., Classical, Lo-fi, Rock")
        music_while_studying = st.radio("Do you listen to music while studying?", ("Yes", "No"), horizontal=True)
        music_volume = st.select_slider("Do you prefer quiet or loud music?", options=["Quiet", "Moderate", "Loud"])
        
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            # Save the data to the session_state dictionary
            st.session_state.user_info = {
                "favourite_music_style": music_style,
                "music_while_studying": music_while_studying,
                "preferred_volume": music_volume
            }
            # Move to the next page
            st.session_state.page_number = 2
            st.rerun() # Use rerun to immediately show the next page

def render_test_page(page_num: int, with_music: bool):
    """A generic function to render a test page."""
    st.header(f"Test Section {page_num - 1}")

    # Load test content
    test_text, questions = load_test_from_gemini()
    st.markdown(test_text)
    
    if with_music:
        # Use the user's preference from the first page as a prompt
        music_prompt = st.session_state.user_info.get("favourite_music_style", "calm")
        audio_bytes = load_music(music_prompt)
        st.audio(audio_bytes, format='audio/wav')

    st.divider()
    
    # Store answers in a dictionary for this page
    page_answers = {}
    for i, q in enumerate(questions):
        # The key for each widget must be unique across the entire app
        page_answers[q] = st.radio(q, ("Yes", "No"), key=f"p{page_num}_q{i}", horizontal=True)
        
    if st.button("Next", key=f"next_p{page_num}"):
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
            "test_answers": st.session_state.test_answers
        }
        
        with st.spinner("Submitting your data to the database..."):
            success = submit_to_firestore(final_data)
            if success:
                st.success("Your submission was successful! Thank you for participating.")
                st.balloons()
                # Optionally clear the state after submission
                for key in st.session_state.keys():
                    del st.session_state[key]


# --- MAIN APP ROUTER ---

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