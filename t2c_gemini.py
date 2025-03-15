import streamlit as st
import requests
import json
import time
import os
import pymongo
from bson import ObjectId

st.set_page_config(page_icon="🛠️", layout="wide",
                   page_title="T2C Gemini")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🛠️Text-2-CAD Assistant🛠️")

st.divider()



# --- Constants and Configuration ---

# MongoDB Connection (Use st.secrets for deployment)
MONGO_URI = st.secrets.get("MONGO_URI", os.environ.get("MONGO_URI"))  # Default to local if not set
DB_NAME = st.secrets.get("MONGO_DB_NAME", os.environ.get("MONGO_DB_NAME")) # Default database name
COLLECTION_NAME = st.secrets.get("MONGO_COLLECTION_NAME", os.environ.get("MONGO_COLLECTION_NAME"))

# --- Helper Functions ---

def get_mongo_client(uri: str):
    """Establishes a connection to MongoDB, handling URI parsing."""
    try:
        # Parse the URI to handle special characters and port issues.

        client = pymongo.MongoClient(uri)
        return client

    except pymongo.errors.ConnectionFailure as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None
    except ValueError as e:  # Catch ValueError for invalid port
        st.error(f"Invalid port specified in MongoDB URI: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred connecting to MongoDB: {e}")
        return None

def log_interaction(db, session_id: str, description: str, feedback: str = "", code: str = ""):
    """Logs interaction data to MongoDB."""
    try:
        collection = db[COLLECTION_NAME]
        doc = {
            "session_id": session_id,
            "timestamp": time.time(),  # Use a numerical timestamp
            "description": description,
            "feedback": feedback,
            "code": code,
        }
        result = collection.insert_one(doc)
        return result.inserted_id
    except Exception as e:
        st.error(f"Error logging to MongoDB: {e}")
        return None


def _send_request_to_gemini(api_endpoint, request_data):
    """Sends a request to the Gemini API."""
    headers = {"Content-Type": "application/json"}
    request_body = json.dumps(request_data)
    response = requests.post(api_endpoint, headers=headers, data=request_body)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Gemini API Error: {response.status_code} - {response.text}")
        return None

def get_code_from_gemini(description: str, api_key, GEMINI_MODEL_NAME) -> str:
    """Gets FreeCAD Python code from Gemini."""
    if not api_key:
        st.error("Cannot generate code without a Google AI Studio API Key.")
        return ""

    api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
    system_prompt = """You are a CAD assistant that converts descriptions into FreeCAD Python code.
Return ONLY executable Python code with no explanations, comments, or markdown formatting (no ```).
The code should create the specified 3D model in FreeCAD.  Be concise but complete."""

    user_prompt = f"""Create FreeCAD Python code for: {description}

The code must:
1. Import necessary FreeCAD modules (e.g., `import FreeCAD as App`, `import Part`).
2. Create a new document if one is not already active (`App.ActiveDocument`).  Use `App.newDocument("ModelName")` if needed.
3. Create the 3D geometry described, using appropriate FreeCAD functions (e.g., `Part.makeBox`, `Part.makeSphere`, `Part.makeCylinder`).
4. Use parametric modeling where appropriate. Define dimensions as variables (e.g., `length = 10`, `width = 5`).
5. If dimensions are not specified, use reasonable default values.  Do *not* leave dimensions undefined.
6. Add the created shape to the active document using `Part.show(shape_name)`.
7. Ensure the code is complete and ready to execute without errors.
8. Recompute the document App.ActiveDocument.recompute()
9. Set the view to fit the object Gui.SendMsgToActiveView("ViewFit")
Return ONLY the Python code. Do not include ANY explanations or extra text.
"""
    request_data = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt + '\n' + user_prompt}]},
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 12000,
        }
    }

    response_data = _send_request_to_gemini(api_endpoint, request_data)
    if not response_data:
        return ""

    if response_data and "candidates" in response_data and response_data["candidates"]:
        code = response_data["candidates"][0]["content"]["parts"][0]["text"]
        return code.replace("```python", "").replace("```", "").strip()
    else:
        st.error("Failed to get code from Gemini.")
        if response_data:
            st.error(json.dumps(response_data, indent=2))
        return ""


def apply_suggestions(code: str, suggestions: str, original_description: str, GEMINI_MODEL_NAME, api_key) -> str:
    """Sends code and user suggestions to LLM for improvement."""
    if not suggestions:
        return code

    api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
    system_prompt = "Refine FreeCAD Python code based on user suggestions.  Return ONLY improved code, no explanations."
    user_prompt = f"""Original Description:\n{original_description}\n\nOriginal Code:\n{code}\n\nUser Suggestions:\n{suggestions}\n\nImproved Code:"""

    request_data = {"contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n{user_prompt}"}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 12000}}

    response = _send_request_to_gemini(api_endpoint, request_data)

    if response and response.get("candidates"):
        improved_code = response["candidates"][0]["content"]["parts"][0]["text"].replace("```python", "").replace("```", "").strip()
        return improved_code

    st.warning("Failed to apply suggestions. Using previous code.")
    return code


# --- Streamlit App ---

def main():
    with st.sidebar:
        url = 'https://computergenerateddesign.com/'

        st.header("Instructions")
        st.write("1. Optional: Select a generative model")
        st.write("2. Optional: Enter your own Google AI Studio API key below or use our shared API key. Our shared API key will be subject to rate limitiations with all other webpage users.")
        st.write("3. Provide specifciations for a CAD model to the LLM assistant in the box below (e.g.Table, 40cm tall) then click \'Generate Code\'. The generated code can then be pasted in a FreeCAD Macro and executed.")
        try:
            gif_path = "https://raw.githubusercontent.com/cmoreno41/GeminiFreeCAD/refs/heads/main/streamlit_gif.gif"
            st.sidebar.image(gif_path, caption="Code Generation")
        except FileNotFoundError:
            st.error("Error loading GIF. Please check the file path.")
        
        st.write("4. In FreeCad, navigate to \"Macro\" in the toolbar and select \"Macros\".")
        try:
            img_path = "https://raw.githubusercontent.com/cmoreno41/CGDFreeCAD/refs/heads/main/streamlit_png/Macro_click.png"
            st.sidebar.image(img_path, caption="Create Macro")
        except FileNotFoundError:
            st.error("Error loading image. Please check the file path.")
        
        st.write("5. Select \"Create Macro\" and name your new macro file.")
        try:
            img_path = "https://raw.githubusercontent.com/cmoreno41/CGDFreeCAD/refs/heads/main/streamlit_png/Create_macro.png"
            st.sidebar.image(img_path, caption="Create Macro")
        except FileNotFoundError:
            st.error("Error loading image. Please check the file path.")

        st.write("6. Paste the generated code in the new FreeCAD Macro." )
        

        st.write("7. Click \"Execute Macro\" in the macros toolbar to generate your model.")
        try:
            img_path = "https://raw.githubusercontent.com/cmoreno41/CGDFreeCAD/refs/heads/main/streamlit_png/execute_macro.png"
            st.sidebar.image(img_path, caption="execute Macro")
        except FileNotFoundError:
            st.error("Error loading image. Please check the file path.")

        st.write("8. Provide revision feedback to the AI agent to improve or modify the original code.")
        try:
            img_path = "https://raw.githubusercontent.com/cmoreno41/CGDFreeCAD/refs/heads/main/streamlit_png/improvement.png"
            st.sidebar.image(img_path, caption="Improve Macro")
        except FileNotFoundError:
            st.error("Error loading image. Please check the file path.")

        st.write("9. Keep generating and revising in the same macro file or refresh the app to generate a new design.")
        st.markdown("---")  # Add a separator line

        st.write("Note: In this webpage we write all prompts and feedback to a MongoDB database with the hopes of fine-tuning an LLM for CAD generation in the future." )
        st.markdown("This is an alternative to downloading the FreeCad Macro gemini interface directly from: [GeminiFreeCad Webpage](%s)." %url)
    
    
    st.title("FreeCAD Code Generator with Gemini")

    available_models = [
            "gemini-2.0-pro-exp-02-05",
            "gemini-1.5-pro-002",
            "gemini-1.0-pro",
            "gemini-1.0-pro-001"
    ]
    selected_model_name = st.selectbox("Select a Generative Model:", available_models)

    # Gemini API Key (Use st.secrets, with fallback)
    api_key = st.text_input("OPTIONAL: Enter your Google AI Studio API Key", type="password")
    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("Google AI Studio API Key is not provided, no output will be generated.")

    
    # --- MongoDB Connection ---
    client = get_mongo_client(MONGO_URI)
    if client is None:
        return  # Stop if no connection

    db = client[DB_NAME]

    # --- Session Handling ---
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(ObjectId())  # Generate unique session ID
        st.session_state.code = ""
        st.session_state.original_description = ""
        st.session_state.feedback = ""

    with st.form(key='my_form'):
        description = st.text_area("Enter a description of the CAD model:", height=150)
        submit_button = st.form_submit_button(label='Generate Code')

    if submit_button:
        # Log the initial prompt
        generated_code = get_code_from_gemini(description, api_key, selected_model_name)
        st.session_state.code = generated_code
        st.session_state.original_description = description
        log_interaction(db, st.session_state.session_id, description) # Log initial interaction

    if st.session_state.code:
        feedback = st.text_area("Enter suggestions to improve the code:", height=100, key="feedback_input")
        improve_button = st.button("Improve Code")

        if improve_button:
            if feedback:  # Check for empty feedback
                st.session_state.feedback += f"\n---TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}---\n {feedback}"
                # Log interaction with feedback and updated code.
                improved_code = apply_suggestions(st.session_state.code, feedback, st.session_state.original_description, selected_model_name, api_key)
                st.session_state.code = improved_code
                log_interaction(db, st.session_state.session_id, st.session_state.original_description, feedback, improved_code)
            else:
                st.warning("Please enter suggestions to improve the code.")

        st.subheader("Generated Code:")
        st.code(st.session_state.code, language="python")  # Display the current code

    if st.session_state.feedback:
        with st.expander("Feedback History"):
            st.text(st.session_state.feedback)


if __name__ == "__main__":
    main()