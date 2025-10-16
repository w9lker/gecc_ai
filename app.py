import json
import random
import time
import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import io
import base64
from google.oauth2 import service_account
from google.cloud import firestore
import google.auth.transport.requests
import traceback
import requests
import wave
from google import genai

# --- HELPER FUNCTIONS  ---
TEXT_GENERATION_PROMPT = """
    Generate a short reading passage for a focus test, and provide 3 comprehension questions.
    Return your response strictly as JSON (give me a json in a format it is stored in the file, don't give markdown as output)
    - "generated_text": the passage as a string,
    - "questions": a list of objects, each with "text" (the question) and "correct_response" ("Yes" or "No").
"""


def get_access_token_for_lyria() -> str:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["lyria"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    req = google.auth.transport.requests.Request()
    creds.refresh(req)
    return creds.token


def decode_prediction_to_wav_bytes(pred_bytes_b64: str) -> bytes:
    try:
        raw = base64.b64decode(pred_bytes_b64)
        pcm = np.frombuffer(raw, dtype=np.int16)
        # Ensure even number of samples for stereo reshape
        if pcm.size % 2 != 0:
            pcm = pcm[:-1]
        stereo = pcm.reshape(-1, 2)

        # Build a WAV using the standard library
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            nchannels = 2
            sampwidth = 2  # 16-bit
            framerate = 48000
            nframes = stereo.shape[0]
            w.setnchannels(nchannels)
            w.setsampwidth(sampwidth)
            w.setframerate(framerate)
            w.writeframes(stereo.tobytes())
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error decoding audio: {e}")
        return None


def create_music_prompt(music_params: dict) -> tuple:
    prompt = music_params.get("prompt", "").strip()
    return prompt, music_params.get("negative_prompt", "")


def load_passage():
    passage = st.session_state.available_passages.pop()  # guaranteed not to end
    return passage["generated_text"], passage["questions"]


def load_music(music_params: dict, max_retries=3):
    try:
        # Get access token
        token = get_access_token_for_lyria()
        if not token:
            return create_silent_audio()

        # Create detailed prompt
        music_prompt, negative_prompt = create_music_prompt(music_params)

        with st.expander("🎵 Music Generation Details", expanded=False):
            st.write(f"**Main Prompt:** {music_prompt}")
            st.write(f"**Negative Prompt:** {negative_prompt}")

        # Set up API endpoint
        project_id = st.secrets["lyria"]["project_id"]
        endpoint = (
            f"https://us-central1-aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/us-central1/publishers/google/models/lyria-002:predict"
        )

        # Prepare request - following official Lyria API format
        st.info(music_prompt)
        instance = {
            "prompt": music_prompt,
            "sample_count": 1,
        }

        # Add negative prompt if provided
        if negative_prompt.strip():
            instance["negative_prompt"] = negative_prompt

        payload = {"instances": [instance], "parameters": {}}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Make request with progress indicator
        with st.spinner(
            "🎼 Creating your personalized study music... This may take 30-60 seconds"
        ):
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.post(
                        endpoint, headers=headers, json=payload, timeout=120
                    )
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt < max_retries:
                        st.warning(
                            f"API error, retrying... (attempt {attempt}/{max_retries}): {str(e)[:100]}"
                        )
                        time.sleep(3)
                        continue
                    else:
                        st.error(
                            f"Failed to generate music after {max_retries} attempts: {str(e)[:200]}"
                        )
                        return create_silent_audio()

        # Parse response
        try:
            data = response.json()
            predictions = data.get("predictions", [])

            if not predictions:
                st.error("No music generated. Using silence instead.")
                return create_silent_audio()

            # Decode the first prediction to WAV bytes
            pred_bytes_b64 = predictions[0]["bytesBase64Encoded"]
            wav_bytes = decode_prediction_to_wav_bytes(pred_bytes_b64)

            if wav_bytes:
                st.success("🎵 Music generated successfully!")
                return wav_bytes
            else:
                return create_silent_audio()

        except json.JSONDecodeError:
            st.error("Invalid response format from API")
            return create_silent_audio()

    except Exception as e:
        st.session_state.last_error = str(e)  # Store error for refinement
        st.error(f"Error generating music: {e}")
        return create_silent_audio()


def create_silent_audio(duration=30):
    samplerate = 44100  # 44.1kHz
    # Generate a silent numpy array
    silent_array = np.zeros(int(samplerate * duration))

    # Use an in-memory bytes buffer
    buffer = io.BytesIO()
    write(buffer, samplerate, silent_array.astype(np.int16))

    return buffer.getvalue()


def submit_to_firestore(data: dict):
    """
    Function to submit the final collected data to Google Cloud Firestore.
    This function requires authentication to be set up.
    """
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["firestore"]
        )
        db = firestore.Client(
            credentials=creds, project=st.secrets["firestore"]["project_id"]
        )

        collection_ref = db.collection("user_responses")
        doc_ref = collection_ref.add(data)
        st.success("Your responses were saved. Thank you! 💻")
        return True
    except Exception as e:
        st.error(f"Failed to submit to Firestore: {str(e)[:200]}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return False


def restart_app():
    """Clear all session state and restart the app."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def finetune_text(prompt):
    finetune_prompt = f"""
            When prompting Lyria 2 it's helpful to consider the overall style of music you want to generate. Consider options such as: classical, electronic, rock, jazz, hip hop, or pop. You can even describe more general styles that include cinematic, ambient, or lo-fi.
            With this in mind, please give me back a detailed pure finetuned prompt, given the original prompt: {prompt}.
            Again just return the finetuned prompt (no extra punctuation or markdown), make the prompt around 50 words and describe the music through style, instruments, genre, tone, intensity.
        """
    try:
        client = genai.Client(api_key=st.secrets["gemini"]["api_key"])
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=finetune_prompt,
        )
    except Exception as e:
        st.error(f"Error refining prompt: {e}")
        return prompt  # Return original if error
    return response.text if response else prompt


# --- STATE INITIALIZATION ---
if "page_index" not in st.session_state:
    st.session_state.page_index = 1
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "music_params" not in st.session_state:
    st.session_state.music_params = {}
if "test_answers" not in st.session_state:
    st.session_state.test_answers = {}
if "generated_music_cache" not in st.session_state:
    st.session_state.generated_music_cache = {}
if "page_order" not in st.session_state:
    test_pages = [3, 2, 4]
    # random.shuffle(test_pages)
    st.session_state.page_order = [1] + test_pages + [5]
if "available_passages" not in st.session_state:
    with open("collection.json", "r", encoding="utf-8") as f:
        available_passages = json.load(f)["passages"]
    print(available_passages)
    random.shuffle(available_passages)
    st.session_state.available_passages = available_passages


# --- PAGE RENDERING FUNCTIONS ---
def render_page_1():
    """Renders the initial user information gathering page."""
    st.header("Welcome! Let's get to know you.")
    email = st.text_input("What is your email")

    st.markdown("#### 🎯 Music Prompt")
    prompt = st.text_area(
        "Enter a complete custom prompt for music generation:",
        placeholder="Example: Create a dreamy soundscape with soft rain sounds, distant thunder, and ethereal synthesizers at 60 BPM, reminiscent of Brian Eno's ambient works",
        height=80,
    )

    if st.button("🔄 Refine Prompt with AI"):
        with st.spinner("Refining your prompt..."):
            refined = finetune_text(prompt.strip())
            st.session_state["refined"] = refined
        st.info(f"Refined Prompt: {refined}")

    # Basic user info form
    with st.form("user_info_form"):
        st.subheader("📋 Basic Information")
        music_while_studying = st.radio(
            "Do you usually listen to music while studying?",
            ("Yes", "No"),
            horizontal=True,
        )

        # Advanced options in expander
        with st.expander("🔧 Advanced Music Parameters (Optional)"):
            negative_prompt = st.text_area(
                "What should the music NOT include?",
                placeholder="e.g., vocals, sudden changes, aggressive sounds",
                help="Describe what you want to avoid in the generated music",
            )

        submitted = st.form_submit_button(
            "🎵 Continue to Study Sessions", type="primary"
        )

        if submitted:
            # Check if using alternative prompt or regular parameters
            if prompt.strip():
                # Using alternative prompt - store it and minimal other info
                st.session_state.user_info = {
                    "music_while_studying": music_while_studying,
                    "email": email,
                }
                st.session_state.music_params = {
                    "prompt": prompt.strip()
                    if "refined" not in st.session_state
                    else st.session_state["refined"],
                    "negative_prompt": negative_prompt.strip(),
                }
            else:
                st.error("Please enter a prompt")
                return

            # Move to the next page
            st.session_state.page_index += 1
            st.rerun()


def render_test_page(page_num: int, with_music: bool):
    """A generic function to render a test page."""

    # Different test types for variety
    test_types = {
        2: {
            "title": "Reading Comprehension - Baseline",
            "icon": "📚",
            "description": "First, let's establish your baseline reading performance without any music.",
        },
        3: {
            "title": "Reading Comprehension - AI Background Music",
            "icon": "🎵",
            "description": "Now let's see how background music affects your focus and comprehension.",
        },
        4: {
            "title": "Reading Comprehension - External Music Session",
            "icon": "🎼",
            "description": "Final test with a different passage and the same music style to confirm results.",
        },
    }

    test_info = test_types[page_num]

    st.header(f"{test_info['title']} {test_info['icon']}")
    st.markdown(f"*{test_info['description']}*")

    # Use session_state to cache test content per page
    test_key = f"test_content_page_{page_num}"
    if test_key not in st.session_state:
        test_text, question_obj_list = load_passage()
        st.session_state[test_key] = (test_text, question_obj_list)
    else:
        test_text, question_obj_list = st.session_state[test_key]

    if with_music:
        st.markdown("### 🎵 Background Music")
        # Generate music based on user preferences
        music_cache_key = f"music_page_{page_num}"

        if music_cache_key not in st.session_state.generated_music_cache:
            audio_bytes = load_music(st.session_state.music_params)
            st.session_state.generated_music_cache[music_cache_key] = audio_bytes

        audio_bytes = st.session_state.generated_music_cache[music_cache_key]

        if audio_bytes and len(audio_bytes) > 0:
            st.audio(audio_bytes, format="audio/wav", loop=True, autoplay=True)
            st.caption(
                "🎧 You can adjust the volume and loop the music using the controls above. Start the music before reading."
            )
        else:
            st.warning("Music generation failed. Continuing with silent study session.")

        st.divider()

    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("### 📖 Reading Passage")
        st.markdown(test_text)

    with col_right:
        st.markdown("### ❓ Comprehension Questions")

        # Store answers in a dictionary for this page
        page_answers = {}
        for i, question_obj in enumerate(question_obj_list):
            q = question_obj["text"]
            page_answers[q] = st.radio(
                q, ("Yes", "No"), key=f"p{page_num}_q{i}", horizontal=True
            )

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button(
            "📝 Complete This Section", key=f"next_p{page_num}", type="primary"
        ):
            # evaluate the test answers
            correct_count = 0
            for question, correct_response in [
                (question_obj["text"], question_obj["correct_response"])
                for question_obj in question_obj_list
            ]:
                if page_answers[question] == correct_response:
                    correct_count += 1

            page_answers["correct_count"] = correct_count
            page_answers["total_questions"] = len(question_obj_list)
            page_answers["had_music"] = with_music
            page_answers["test_type"] = test_info["title"]

            # Save this page's answers to the main state
            st.session_state.test_answers[f"page_{page_num}"] = page_answers

            # Show quick feedback
            accuracy = (correct_count / len(question_obj_list)) * 100
            st.success(
                f"✅ Section completed! Accuracy: {correct_count}/{len(question_obj_list)} ({accuracy:.1f}%)"
            )
            time.sleep(1.5)

            # Increment page number and rerun
            st.session_state.page_index += 1
            st.info(st.session_state.page_index)
            st.rerun()

    with col2:
        if st.button(
            "🔄 Restart Study",
            key=f"restart_p{page_num}",
            help="Start over from the beginning",
        ):
            restart_app()


def render_final_page():
    """Renders the final thank you and submission page."""
    st.header("🎉 Study Complete!")
    st.markdown("Thank you for participating in our music and focus study!")

    # Calculate and display results
    st.subheader("📊 Your Performance Summary")

    results_summary = {}
    music_sections = []
    no_music_sections = []

    for page_key, answers in st.session_state.test_answers.items():
        section_name = answers.get("test_type", "Unknown")
        had_music = answers.get("had_music", False)
        score = answers.get("correct_count", 0)
        total = answers.get("total_questions", 3)
        percentage = (score / total * 100) if total > 0 else 0

        result_data = {
            "score": score,
            "total": total,
            "percentage": percentage,
            "section_name": section_name,
        }

        if had_music:
            music_sections.append(result_data)
        else:
            no_music_sections.append(result_data)

        results_summary[section_name] = result_data

    # Combine all data into one dictionary for submission
    final_data = {
        "timestamp": time.time(),
        "user_info": st.session_state.user_info,
        "music_params": st.session_state.music_params,
        "test_answers": st.session_state.test_answers,
        "results_summary": results_summary,
    }

    with st.spinner("Submitting your data to the research database..."):
        success = submit_to_firestore(final_data)
        if success:
            st.success(
                "✅ Your results have been submitted successfully! Thank you for contributing to our research."
            )
            st.balloons()

    with col2:
        if st.button(
            "🔄 Start New Study", help="Clear all data and start a new study session"
        ):
            restart_app()


# --- MAIN APP ROUTER ---
st.set_page_config(
    page_title="Music & Focus Study",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Header with restart option
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🎵 Interactive Music & Focus Study")
with col2:
    if st.button("🔄 Restart", key="header_restart", help="Start over from beginning"):
        restart_app()

# Add progress indicator
if st.session_state.page_index <= 5:
    progress = (st.session_state.page_index - 1) / 4
    st.progress(progress, text=f"Step {st.session_state.page_index} of 5")

# Error boundary wrapper
try:
    page = st.session_state.page_order[st.session_state.page_index - 1]

    if page == 1:
        render_page_1()
    elif page == 2:
        render_test_page(page_num=2, with_music=False)
    elif page == 3:
        render_test_page(page_num=3, with_music=True)
    elif page == 4:
        render_test_page(page_num=4, with_music=False)
    elif page == 5:
        render_final_page()
    else:
        st.error("Invalid page state detected.")
        if st.button("🔄 Reset Application"):
            restart_app()

except Exception as e:
    st.error("An unexpected error occurred. Please restart the application.")
    st.code(f"Error: {str(e)}")

    with st.expander("Error Details"):
        st.code(traceback.format_exc())

    if st.button("🔄 Restart Application", type="primary"):
        restart_app()
