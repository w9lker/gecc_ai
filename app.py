import json
import time
import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import io
import base64
import re
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
    """Get OAuth token for Lyria API using separate credentials."""
    try:
        # Use separate lyria credentials for Lyria API access
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["lyria"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        req = google.auth.transport.requests.Request()
        creds.refresh(req)
        return creds.token
    except Exception as e:
        st.error(f"Failed to get access token for Lyria: {e}")
        return None

def decode_prediction_to_wav_bytes(pred_bytes_b64: str) -> bytes:
    """
    Lyria predict returns raw 48kHz 16-bit PCM stereo as base64-encoded bytes.
    We wrap that raw PCM into a WAV container so media players can open it.
    """
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
    """
    Create a detailed music prompt based on user preferences.
    Returns: (prompt, negative_prompt)
    """
    try:
        # Base style/genre
        genre = music_params.get('genre', 'ambient')
        
        # Tempo mapping
        tempo_map = {
            "Very Slow": "very slow tempo, meditative pace",
            "Slow": "slow tempo, relaxed pace", 
            "Moderate": "moderate tempo, steady rhythm",
            "Fast": "fast tempo, energetic pace",
            "Very Fast": "very fast tempo, high energy"
        }
        tempo_desc = tempo_map.get(music_params.get('tempo', 'Moderate'))
        
        # Mood mapping
        mood = music_params.get('mood', 'calm')
        
        # Instrumentation
        instruments = music_params.get('instruments', [])
        instrument_desc = ""
        if instruments:
            if len(instruments) == 1:
                instrument_desc = f", featuring {instruments[0].lower()}"
            elif len(instruments) == 2:
                instrument_desc = f", featuring {instruments[0].lower()} and {instruments[1].lower()}"
            else:
                instrument_desc = f", featuring {', '.join(instruments[:-1]).lower()}, and {instruments[-1].lower()}"
        
        # Volume intensity mapping
        volume_map = {
            "Very Quiet": "very soft and gentle, barely audible",
            "Quiet": "soft and gentle, background ambiance",
            "Moderate": "moderate volume, balanced dynamics",
            "Loud": "full volume, rich and present",
            "Very Loud": "powerful and intense, strong presence"
        }
        volume_desc = volume_map.get(music_params.get('volume', 'Moderate'))
        
        # Build main prompt
        main_prompt = f"{genre} music, {mood} and peaceful, {tempo_desc}{instrument_desc}, {volume_desc}"
        
        # Add study-specific characteristics
        main_prompt += ", instrumental, non-distracting, perfect for concentration and focus"
        
        # Handle negative prompt
        negative_items = []
        
        # Add user's custom negative prompt
        custom_negative = music_params.get('negative_prompt', '').strip()
        if custom_negative:
            negative_items.append(custom_negative)
            
        # Add automatic negative prompts based on selections
        if music_params.get('tempo') == 'Very Slow':
            negative_items.extend(["fast", "energetic", "upbeat"])
        elif music_params.get('tempo') in ['Fast', 'Very Fast']:
            negative_items.extend(["slow", "meditative", "sleepy"])
            
        if music_params.get('mood') == 'calm':
            negative_items.extend(["aggressive", "chaotic", "dissonant"])
        elif music_params.get('mood') == 'energetic':
            negative_items.extend(["sleepy", "boring", "monotonous"])
            
        # Always exclude distracting elements for study music
        negative_items.extend(["vocals", "lyrics", "singing", "sudden changes", "jarring transitions"])
        
        negative_prompt = ", ".join(set(negative_items))  # Remove duplicates
        
        return main_prompt, negative_prompt
        
    except Exception as e:
        st.error(f"Error creating music prompt: {e}")
        return "ambient calm music for studying", "vocals, lyrics"

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
                client = genai.Client(api_key=st.secrets['gemini']['api_key'])

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
                if ("rate limit" in error_msg.lower() or "429" in error_msg) and attempt < max_retries:
                    st.warning(f"Rate limit hit, retrying in {retry_delay} seconds... (attempt {attempt}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    st.error("Rate limit reached. Maximum retries exceeded.")
                    return "Error loading test (rate limit). Please try again later.", []
                else:
                    st.error(f"Failed to load test from Gemini: {str(e)[:200]}")
                    return "Error loading test. Please try again.", []

def load_music(music_params: dict, max_retries=3):
    """
    Generate music using Lyria API based on user preferences.
    Args:
        music_params (dict): Dictionary containing all music parameters
    Returns:
        bytes: The audio data in bytes, or None if generation fails.
    """
    try:
        # Get access token
        token = get_access_token_for_lyria()
        if not token:
            st.error("Failed to authenticate with Lyria API")
            return create_silent_audio()
            
        # Create detailed prompt
        music_prompt, negative_prompt = create_music_prompt(music_params)
        
        with st.expander("ðŸŽµ Music Generation Details", expanded=False):
            st.write(f"**Main Prompt:** {music_prompt}")
            st.write(f"**Negative Prompt:** {negative_prompt}")
            
        # Set up API endpoint - use your project for Lyria
        project_id = st.secrets["lyria"]["project_id"]
        endpoint = (
            f"https://us-central1-aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/us-central1/publishers/google/models/lyria-002:predict"
        )
        
        # Prepare request - following official Lyria API format
        instance = {
            "prompt": music_prompt
        }
        
        # Add negative prompt if provided
        if negative_prompt.strip():
            instance["negative_prompt"] = negative_prompt
            
        # Add seed if provided (for reproducibility)
        seed = music_params.get('seed')
        if seed is not None and seed > 0:
            instance["seed"] = int(seed)
        else:
            # Use sample_count instead of seed for variety
            instance["sample_count"] = 1
        
        payload = {"instances": [instance], "parameters": {}}
        headers = {
            "Authorization": f"Bearer {token}", 
            "Content-Type": "application/json"
        }
        
        # Make request with progress indicator
        with st.spinner("ðŸŽ¼ Creating your personalized study music... This may take 30-60 seconds"):
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt < max_retries:
                        st.warning(f"Request timeout, retrying... (attempt {attempt}/{max_retries})")
                        time.sleep(3)
                        continue
                    else:
                        st.error("Music generation timed out. Using silence instead.")
                        return create_silent_audio()
                except requests.exceptions.HTTPError as e:
                    error_msg = f"HTTP Error: {e.response.status_code}"
                    if e.response.status_code == 429:  # Rate limit
                        st.error("API rate limit reached. Using silence instead.")
                    elif e.response.status_code == 403:  # Permission denied
                        st.error("API access denied. Check your credentials and permissions.")
                    else:
                        st.error(f"API Error: {error_msg}")
                    return create_silent_audio()
                except Exception as e:
                    if attempt < max_retries:
                        st.warning(f"API error, retrying... (attempt {attempt}/{max_retries}): {str(e)[:100]}")
                        time.sleep(3)
                        continue
                    else:
                        st.error(f"Failed to generate music after {max_retries} attempts: {str(e)[:200]}")
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
                st.success("ðŸŽµ Music generated successfully!")
                return wav_bytes
            else:
                return create_silent_audio()
                
        except json.JSONDecodeError:
            st.error("Invalid response format from API")
            return create_silent_audio()
        
    except Exception as e:
        st.error(f"Unexpected error in music generation: {str(e)[:200]}")
        return create_silent_audio()

def create_silent_audio(duration=30):
    """
    Create a silent audio track as fallback.
    Args:
        duration (int): Duration in seconds
    Returns:
        bytes: Silent WAV audio data
    """
    try:
        samplerate = 44100  # 44.1kHz
        # Generate a silent numpy array
        silent_array = np.zeros(int(samplerate * duration))
        
        # Use an in-memory bytes buffer
        buffer = io.BytesIO()
        write(buffer, samplerate, silent_array.astype(np.int16))
        
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error creating silent audio: {e}")
        return b''

def submit_to_firestore(data: dict):
    """
    Function to submit the final collected data to Google Cloud Firestore.
    This function requires authentication to be set up.
    """
    try:
        creds = service_account.Credentials.from_service_account_info(st.secrets["firestore"])
        db = firestore.Client(credentials=creds, project=st.secrets["firestore"]["project_id"])

        collection_ref = db.collection("user_responses")
        doc_ref = collection_ref.add(data)
        st.success("Your responses were saved. Thank you! ðŸ’»")
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

# --- STATE INITIALIZATION ---

# Use session_state to store data across reruns and pages
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'music_params' not in st.session_state:
    st.session_state.music_params = {}
if 'test_answers' not in st.session_state:
    st.session_state.test_answers = {}
if 'generated_music_cache' not in st.session_state:
    st.session_state.generated_music_cache = {}

# --- PAGE RENDERING FUNCTIONS ---

def render_page_1():
    """Renders the initial user information gathering page."""
    st.header("Welcome! Let's get to know you.")
    
    st.markdown("""
    This study will test how different types of music affect your focus and comprehension. 
    We'll generate personalized music based on your preferences using AI.
    """)
    
    # Basic user info form
    with st.form("user_info_form"):
        st.subheader("ðŸ“‹ Basic Information")
        music_while_studying = st.radio(
            "Do you usually listen to music while studying?", 
            ("Yes", "No"), 
            horizontal=True
        )
        
        st.subheader("ðŸŽµ Music Preferences")
        
        # Genre selection
        genre = st.selectbox(
            "What is your favorite music genre?",
            ["Classical", "Lo-fi", "Jazz", "Ambient", "Electronic", "Acoustic", "Piano", "Rock", "Pop", "Hip Hop", "Other"],
            help="Select the genre that best matches your preference"
        )
        
        other_genre = ""
        if genre == "Other":
            other_genre = st.text_input("Please specify your preferred genre:")
        
        # Volume with proper slider points
        volume = st.select_slider(
            "Preferred music volume level:",
            options=["Very Quiet", "Quiet", "Moderate", "Loud", "Very Loud"],
            value="Moderate",
            help="Choose how loud you prefer your study music"
        )
        
        # Tempo
        tempo = st.select_slider(
            "Preferred tempo (speed):",
            options=["Very Slow", "Slow", "Moderate", "Fast", "Very Fast"],
            value="Moderate",
            help="Choose the speed/rhythm you prefer"
        )
        
        # Mood
        mood = st.selectbox(
            "What mood do you prefer for study music?",
            ["Calm", "Peaceful", "Energetic", "Focused", "Relaxed", "Uplifting", "Meditative"],
            help="Select the emotional tone you find most helpful for studying"
        )
        
        # Instruments (multi-select)
        instruments = st.multiselect(
            "Preferred instruments (optional):",
            ["Piano", "Guitar", "Strings", "Synthesizer", "Flute", "Saxophone", "Drums", "Violin", "Cello", "Harp"],
            help="Select specific instruments you'd like to hear (leave empty for any)"
        )
        
        st.subheader("âš™ï¸ Advanced Options")
        
        # Advanced options in expander
        with st.expander("ðŸ”§ Advanced Music Parameters (Optional)"):
            negative_prompt = st.text_area(
                "What should the music NOT include?",
                placeholder="e.g., vocals, sudden changes, aggressive sounds",
                help="Describe what you want to avoid in the generated music"
            )
            
            seed = st.number_input(
                "Seed for reproducible music (optional):",
                min_value=0,
                max_value=9999,
                value=0,
                help="Use the same seed to get similar music each time (0 = random)"
            )
        
        submitted = st.form_submit_button("ðŸŽµ Continue to Study Sessions", type="primary")
        
        if submitted:
            final_genre = other_genre if genre == "Other" and other_genre.strip() else genre
            
            if not final_genre.strip():
                st.error("Please specify your favorite music genre to continue.")
                return
            
            # Save all parameters
            st.session_state.user_info = {
                "music_while_studying": music_while_studying,
            }
            
            st.session_state.music_params = {
                "genre": final_genre.strip(),
                "volume": volume,
                "tempo": tempo,
                "mood": mood.lower(),
                "instruments": instruments,
                "negative_prompt": negative_prompt.strip(),
                "seed": seed if seed > 0 else None
            }
            
            # Move to the next page
            st.session_state.page_number = 2
            st.rerun()

def render_test_page(page_num: int, with_music: bool):
    """A generic function to render a test page."""
    
    # Different test types for variety
    test_types = {
        2: {"title": "Reading Comprehension - Baseline", "icon": "ðŸ“š", "description": "First, let's establish your baseline reading performance without any music."},
        3: {"title": "Reading Comprehension - With Background Music", "icon": "ðŸŽµ", "description": "Now let's see how background music affects your focus and comprehension."},
        4: {"title": "Reading Comprehension - Extended Music Session", "icon": "ðŸŽ¼", "description": "Final test with a different passage and the same music style to confirm results."}
    }
    
    test_info = test_types[page_num]
    
    st.header(f"{test_info['title']} {test_info['icon']}")
    st.markdown(f"*{test_info['description']}*")

    # Use session_state to cache test content per page
    test_key = f"test_content_page_{page_num}"
    if test_key not in st.session_state:
        test_text, question_obj_list = load_test_from_gemini()
        st.session_state[test_key] = (test_text, question_obj_list)
    else:
        test_text, question_obj_list = st.session_state[test_key]

    if with_music:
        st.markdown("### ðŸŽµ Background Music")
        # Generate music based on user preferences
        music_cache_key = f"music_page_{page_num}"
        
        if music_cache_key not in st.session_state.generated_music_cache:
            audio_bytes = load_music(st.session_state.music_params)
            st.session_state.generated_music_cache[music_cache_key] = audio_bytes
        
        audio_bytes = st.session_state.generated_music_cache[music_cache_key]
        
        if audio_bytes and len(audio_bytes) > 0:
            st.audio(audio_bytes, format='audio/wav', loop=True)
            st.caption("ðŸŽ§ You can adjust the volume and loop the music using the controls above. Start the music before reading.")
        else:
            st.warning("Music generation failed. Continuing with silent study session.")
            
        st.divider()

    st.markdown("### ðŸ“– Reading Passage")
    st.markdown(test_text)
    
    st.divider()
    st.markdown("### â“ Comprehension Questions")
    
    # Store answers in a dictionary for this page
    page_answers = {}
    for i, question_obj in enumerate(question_obj_list):
        q = question_obj['text']
        page_answers[q] = st.radio(
            q, 
            ("Yes", "No"), 
            key=f"p{page_num}_q{i}", 
            horizontal=True
        )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("ðŸ“ Complete This Section", key=f"next_p{page_num}", type="primary"):
            # evaluate the test answers
            correct_count = 0
            for question, correct_response in [(question_obj['text'], question_obj['correct_response']) for 
                                                question_obj in question_obj_list]:
                if page_answers[question] == correct_response: 
                    correct_count += 1
            
            page_answers['correct_count'] = correct_count
            page_answers['total_questions'] = len(question_obj_list)
            page_answers['had_music'] = with_music
            page_answers['test_type'] = test_info['title']
            
            # Save this page's answers to the main state
            st.session_state.test_answers[f"page_{page_num}"] = page_answers
            
            # Show quick feedback
            accuracy = (correct_count / len(question_obj_list)) * 100
            st.success(f"âœ… Section completed! Accuracy: {correct_count}/{len(question_obj_list)} ({accuracy:.1f}%)")
            time.sleep(1.5)
            
            # Increment page number and rerun
            st.session_state.page_number += 1
            st.rerun()
    
    with col2:
        if st.button("ðŸ”„ Restart Study", key=f"restart_p{page_num}", help="Start over from the beginning"):
            restart_app()

def render_final_page():
    """Renders the final thank you and submission page."""
    st.header("ðŸŽ‰ Study Complete!")
    st.markdown("Thank you for participating in our music and focus study!")
    
    # Calculate and display results
    st.subheader("ðŸ“Š Your Performance Summary")
    
    results_summary = {}
    music_sections = []
    no_music_sections = []
    
    for page_key, answers in st.session_state.test_answers.items():
        section_name = answers.get('test_type', 'Unknown')
        had_music = answers.get('had_music', False)
        score = answers.get('correct_count', 0)
        total = answers.get('total_questions', 3)
        percentage = (score / total * 100) if total > 0 else 0
        
        result_data = {
            'score': score,
            'total': total,
            'percentage': percentage,
            'section_name': section_name
        }
        
        if had_music:
            music_sections.append(result_data)
        else:
            no_music_sections.append(result_data)
        
        results_summary[section_name] = result_data
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ðŸ“š Without Music")
        for section in no_music_sections:
            st.metric(
                label="Baseline Score",
                value=f"{section['score']}/{section['total']}",
                delta=f"{section['percentage']:.1f}%"
            )
    
    with col2:
        st.markdown("#### ðŸŽµ With Music")
        music_scores = []
        for section in music_sections:
            music_scores.append(section['percentage'])
            st.metric(
                label=f"Music Session {len(music_scores)}",
                value=f"{section['score']}/{section['total']}",
                delta=f"{section['percentage']:.1f}%"
            )
    
    # Performance comparison
    if no_music_sections and music_sections:
        baseline_avg = sum([s['percentage'] for s in no_music_sections]) / len(no_music_sections)
        music_avg = sum([s['percentage'] for s in music_sections]) / len(music_sections)
        difference = music_avg - baseline_avg
        
        st.markdown("#### ðŸ“ˆ Performance Analysis")
        if difference > 5:
            st.success(f"ðŸŽµ Music improved your performance by {difference:.1f} percentage points!")
        elif difference < -5:
            st.info(f"ðŸ“š You performed {abs(difference):.1f} percentage points better without music.")
        else:
            st.info("ðŸ“Š Music had minimal impact on your performance (within 5% difference).")
    
    # Show music preferences used
    st.subheader("ðŸŽµ Your Music Configuration")
    music_params = st.session_state.music_params
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Genre:** {music_params.get('genre', 'N/A')}")
        st.write(f"**Tempo:** {music_params.get('tempo', 'N/A')}")
        st.write(f"**Mood:** {music_params.get('mood', 'N/A').title()}")
    with col2:
        st.write(f"**Volume:** {music_params.get('volume', 'N/A')}")
        st.write(f"**Instruments:** {', '.join(music_params.get('instruments', [])) or 'Any'}")
        st.write(f"**Study Habit:** {st.session_state.user_info.get('music_while_studying', 'N/A')}")
    
    st.divider()
    
    # Optional: Show detailed data for review (collapsed by default)
    with st.expander("ðŸ“‹ View Detailed Data (Optional)"):
        st.subheader("Music Parameters")
        st.json(st.session_state.music_params)
        
        st.subheader("Test Results")
        st.json(st.session_state.test_answers)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("ðŸ“¤ Submit Results to Research Database", type="primary"):
            # Combine all data into one dictionary for submission
            final_data = {
                "timestamp": time.time(),
                "user_info": st.session_state.user_info,
                "music_params": st.session_state.music_params,
                "test_answers": st.session_state.test_answers,
                "results_summary": results_summary,
                "performance_analysis": {
                    "baseline_avg": baseline_avg if no_music_sections else 0,
                    "music_avg": music_avg if music_sections else 0,
                    "improvement": difference if (no_music_sections and music_sections) else 0
                }
            }
            
            with st.spinner("Submitting your data to the research database..."):
                success = submit_to_firestore(final_data)
                if success:
                    st.success("âœ… Your results have been submitted successfully! Thank you for contributing to our research.")
                    st.balloons()
    
    with col2:
        if st.button("ðŸ”„ Start New Study", help="Clear all data and start a new study session"):
            restart_app()

# --- MAIN APP ROUTER ---

st.set_page_config(
    page_title="Music & Focus Study",
    page_icon="ðŸŽµ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header with restart option
col1, col2 = st.columns([4, 1])
with col1:
    st.title("ðŸŽµ Interactive Music & Focus Study")
with col2:
    if st.button("ðŸ”„ Restart", key="header_restart", help="Start over from beginning"):
        restart_app()

# Add progress indicator
if st.session_state.page_number <= 5:
    progress = (st.session_state.page_number - 1) / 4
    st.progress(progress, text=f"Step {st.session_state.page_number} of 5")

# Error boundary wrapper
try:
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
        st.error("Invalid page state detected.")
        if st.button("ðŸ”„ Reset Application"):
            restart_app()

except Exception as e:
    st.error("An unexpected error occurred. Please restart the application.")
    st.code(f"Error: {str(e)}")
    
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
    
    if st.button("ðŸ”„ Restart Application", type="primary"):
        restart_app()