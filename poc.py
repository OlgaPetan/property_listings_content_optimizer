import streamlit as st
import os
import re

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except ImportError:
    pass  # Not in Streamlit

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key not found. Set it in Streamlit secrets or as an environment variable.")

#os.environ["OPENAI_API_KEY"] = ''
### STREAMLIT FORMS CODE STARTS
st.set_page_config(layout="wide", page_title="Airbnb Content Automation")

# Equal width layout
col1, col2, col3 = st.columns([1, 1, 1])

# Title Section 
with col1:
    with st.expander("Title Settings", expanded=True):
        st.caption("Customize the AutoRank AI updates for the listing title.")
        title_enabled = st.checkbox("Enable Title Automation", value=True)
        title_additional = st.text_area("Additional", key="title_additional", placeholder="Optional context or guidance")
        title_include = st.text_area("Always Include", key="title_include", placeholder="Comma-separated or multi-line items")
        title_exclude = st.text_area("Always Exclude", key="title_exclude", placeholder="Comma-separated or multi-line items")

title_settings = {
    "enabled": title_enabled,
    "additional": title_additional,
    "always_include": [x.strip() for x in title_include.splitlines() if x.strip()],
    "always_exclude": [x.strip() for x in title_exclude.splitlines() if x.strip()],
}

# Summary Section 
with col2:
    with st.expander("Summary Settings", expanded=True):
        st.caption("Customize the AutoRank AI updates for the listing summary.")
        summary_enabled = st.checkbox("Enable Summary Automation", value=True)
        summary_additional = st.text_area("Additional", key="summary_additional", placeholder="Optional context or guidance")
        summary_include = st.text_area("Always Include", key="summary_include", placeholder="Comma-separated or multi-line items")
        summary_exclude = st.text_area("Always Exclude", key="summary_exclude", placeholder="Comma-separated or multi-line items")

summary_settings = {
    "enabled": summary_enabled,
    "additional": summary_additional,
    "always_include": [x.strip() for x in summary_include.splitlines() if x.strip()],
    "always_exclude": [x.strip() for x in summary_exclude.splitlines() if x.strip()],
}

# Space Section 
with col3:
    with st.expander("Space Settings", expanded=True):
        st.caption("Customize the AutoRank AI updates for the listing space description.")
        space_enabled = st.checkbox("Enable Space Automation", value=True)
        space_additional = st.text_area("Additional", key="space_additional", placeholder="Optional context or guidance")
        space_include = st.text_area("Always Include", key="space_include", placeholder="Comma-separated or multi-line items")
        space_exclude = st.text_area("Always Exclude", key="space_exclude", placeholder="Comma-separated or multi-line items")

space_settings = {
    "enabled": space_enabled,
    "additional": space_additional,
    "always_include": [x.strip() for x in space_include.splitlines() if x.strip()],
    "always_exclude": [x.strip() for x in space_exclude.splitlines() if x.strip()],
}

### STREAMLIT FORMS CODE ENDS

### CODE FOR SECTIONS TO BE EXCLUDED FROM GENERATED CONTENT STARTS



# Define the folder and file
exclude_folder = "sections_to_exclude"
exclude_file = "1.txt"  #

# Full path to the file
exclude_path = os.path.join(exclude_folder, exclude_file)

# Load content into variable
excluded_sections = ""
if os.path.exists(exclude_path):
    with open(exclude_path, "r", encoding="utf-8") as f:
        excluded_sections = f.read().strip()
else:
    print(f"⚠File not found: {exclude_path}")

### CODE FOR SECTIONS TO BE EXCLUDED FROM GENERATED CONTENT ENDS

### CODE FOR THE PROMPT THAT GENERATES CONTENT STARTS

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Load embeddings
embedder = OpenAIEmbeddings()
image_vs = FAISS.load_local("image_vectorstore", embedder, allow_dangerous_deserialization=True)
review_vs = FAISS.load_local("review_vectorstore", embedder, allow_dangerous_deserialization=True)

# Search for relevant context (can adjust query later if needed)
image_results = image_vs.similarity_search("Airbnb listing visuals", k=3)
review_results = review_vs.similarity_search("Airbnb guest impressions", k=3)

image_context = "\n".join([doc.page_content for doc in image_results])
review_context = "\n".join([doc.page_content for doc in review_results])

# Build prompt using user settings
def build_generation_prompt(title_settings, summary_settings, space_settings, image_context, review_context):
    parts = []
    parts.append("You are helping improve an Airbnb listing by generating a new title, summary, and space description.")

    if image_context:
        parts.append(f"\nHere are some descriptions of the photos from the listing:\n{image_context}")

    if review_context:
        parts.append(f"\nHere are highlights from guest reviews:\n{review_context}")

    # Section-specific guidance — simplified to remove "For the XYZ section"
    def section_block(label, settings):
        block = []
        if settings.get("enabled"):
            block.append(f"\n{label}")  # Just "Title", "Summary", or "Space"
            if settings.get("additional"):
                block.append(f"- Additional guidance: {settings['additional']}")
            if settings.get("always_include"):
                block.append(f"- Always include: {', '.join(settings['always_include'])}")
            if settings.get("always_exclude"):
                block.append(f"- Avoid: {', '.join(settings['always_exclude'])}")
        return "\n".join(block)

    parts.append(section_block("Title", title_settings))
    parts.append(section_block("Summary", summary_settings))
    parts.append(section_block("Space", space_settings))

    return "\n\n".join(parts).strip()


# Generate the prompt
prompt_text = build_generation_prompt(
    title_settings,
    summary_settings,
    space_settings,
    image_context,
    review_context
)

# Read excluded content
excluded_sections_path = "sections_to_exclude/1.txt"
excluded_sections = ""
if os.path.exists(excluded_sections_path):
    with open(excluded_sections_path, "r", encoding="utf-8") as f:
        excluded_sections = f.read().strip()

# Call OpenAI to generate content
openai_client = OpenAI()
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert in writing effective, clear, natural Airbnb listings."},
        {"role": "user", "content": prompt_text}
    ],
    temperature=0.7
)

generated_output = response.choices[0].message.content.strip()

# Final result with excluded section attached
final_output = f"{generated_output}\n\n---\n{excluded_sections}"

### THE GENERATED CONTENT FORM STARTS

# Clear instructional banner (no emojis)
st.markdown("""
<div style="
    background-color: #f0f4f8;
    padding: 16px;
    border-left: 4px solid #1a73e8;
    border-radius: 6px;
    font-weight: 500;
    font-size: 15px;
    color: #1a1a1a;
">
You can make changes to the content below.
</div>
""", unsafe_allow_html=True)

# Editable output field in bordered form-style box
edited_output = st.text_area(
    label="",
    value=final_output,  # ← your LLM output
    height=400,
    key="generated_editable_output"
)


import os
from datetime import datetime

# Make sure the folder exists
os.makedirs("changed_content", exist_ok=True)

# Create a timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"changed_content/changed_content_{timestamp}.txt"

# Save the edited content to file
with open(filename, "w", encoding="utf-8") as f:
    f.write(edited_output)

# Optional: Show confirmation
st.success(f"Edited content saved as `{filename}`.")


### THE GENERATED CONTENT FORM ENDS

### THE CODE TO ANALYZE CHANGES IN CONTENT OUTPUT STARTS

# Only continue if the user made changes
if edited_output != final_output:
    # Save edited content to a timestamped file
    os.makedirs("changed_content", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"changed_content/changed_content_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(edited_output)

    st.success(f"Edited content saved as `{filename}`.")

    # Compare original vs edited using GPT
    openai_client = OpenAI()

    comparison_prompt = f"""
You are a helpful assistant.

Below is the original content:

\"\"\"{final_output}\"\"\"

Below is the revised content after a human edited it:

\"\"\"{edited_output}\"\"\"

Please summarize the key differences in a clear and natural way. Focus on content, structure, and tone changes.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You explain how written content has been edited."},
            {"role": "user", "content": comparison_prompt}
        ],
        temperature=0.3
    )

    diff_summary = response.choices[0].message.content.strip()

    # Show edit summary
    st.markdown("### Summary of Edits")
    st.markdown(f"""
    <div style="
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 20px;
        background-color: #fefefe;
        white-space: pre-wrap;
        font-size: 15px;
    ">
    {diff_summary}
    </div>
    """, unsafe_allow_html=True)

### THE CODE TO ANALYZE CHANGES IN CONTENT OUTPUT ENDS

### THE CODE THAT CREATES A MAXIMUM CHANGES NUMBER AND NOTIFIES USER STARTS
import streamlit as st
import os
from datetime import datetime
from openai import OpenAI

# Initialize session state variables
if "original_content" not in st.session_state:
    st.session_state.original_content = final_output
if "generated_editable_output" not in st.session_state:
    st.session_state.generated_editable_output = final_output
if "edit_count" not in st.session_state:
    st.session_state.edit_count = 0

# Editable text area
edited_output = st.text_area(
    label="",
    height=400,
    key="generated_editable_output_"
)

# Detect change
if st.session_state.generated_editable_output != st.session_state.original_content:
    if st.session_state.edit_count < 2:
        st.session_state.edit_count += 1
        st.session_state.original_content = st.session_state.generated_editable_output

        # --- Save the edited content ---
        os.makedirs("changed_content", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"changed_content/changed_content_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(st.session_state.generated_editable_output)

        # --- Display floating notification ---
        remaining = 2 - st.session_state.edit_count
        st.markdown(f"""
        <div style="
            position: fixed;
            top: 40%;
            right: -100px;
            background-color: #1a73e8;
            color: white;
            padding: 12px 20px;
            border-radius: 8px 0 0 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            font-weight: 500;
            font-size: 15px;
            transform: translateX(-100%);
            animation: slide-in 0.5s ease-out forwards;
            z-index: 9999;
        ">
        You've made a change. {remaining} edit{'' if remaining == 1 else 's'} remaining.
        </div>

        <style>
        @keyframes slide-in {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to   {{ transform: translateX(0); opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True)

    else:
        st.warning("You have reached the maximum number of edits (2). Please add more specific instructions.")

# Show the summary of edits if applicable
if st.session_state.generated_editable_output != final_output and st.session_state.edit_count > 0:
    
    pass


### THE CODE TO DISPLAY THE CAPTIONS FOR EACH ROOM STARTS

st.markdown("## Room Image Captions")

with st.expander("Captions by Room", expanded=False):

    # Retrieve all captions
    captions = []
    for doc_id in image_vs.index_to_docstore_id.values():
        doc = image_vs.docstore.search(doc_id)
        if doc:
            captions.append((doc.metadata.get("filename", "unknown"), doc.page_content or ""))

    # Keyword rules
    ROOM_RULES = [
        ("Living Room",  [r"\bliving room\b", r"\blounge\b", r"\bsofa\b", r"\bcouch\b", r"\bsectional\b", r"\btv\b", r"\bfireplace\b"]),
        ("Bedroom",      [r"\bbedroom\b", r"\bmaster bedroom\b", r"\bguest bedroom\b", r"\bking\b", r"\bqueen\b", r"\bbunk\b", r"\bcrib\b", r"\bbed\b"]),
        ("Kitchen",      [r"\bkitchen\b", r"\bopen[- ]plan kitchen\b", r"\bkitchenette\b", r"\bstove\b", r"\boven\b", r"\bfridge\b", r"\bkitchen island\b"]),
        ("Dining",       [r"\bdining\b", r"\bdining table\b", r"\bdining area\b"]),
        ("Bathroom",     [r"\bbathroom\b", r"\bshower\b", r"\bbathtub\b", r"\bensuite\b", r"\bwashroom\b", r"\btoilet\b"]),
        ("Outdoor",      [r"\bpatio\b", r"\bgarden\b", r"\byard\b", r"\bdeck\b", r"\bterrace\b", r"\bbalcony\b", r"\bhot tub\b", r"\bfire pit\b", r"\bgrill\b", r"\bbarbecue\b", r"\bpotted plants\b"]),
        ("Entry / Hall", [r"\bentry\b", r"\bfoyer\b", r"\bhallway\b", r"\bstairs?\b", r"\bstaircase\b"]),
        ("Office",       [r"\boffice\b", r"\bstudy\b", r"\bworkspace\b", r"\bdesk\b"]),
        ("Laundry",      [r"\blaundry\b", r"\bwasher\b", r"\bdryer\b", r"\butility\b"]),
        ("Other",        [])  # fallback
    ]

    def classify_room(filename: str, caption: str) -> str:
        text = f"{filename} {caption}".lower()
        for room, patterns in ROOM_RULES:
            for pat in patterns:
                if re.search(pat, text):
                    return room
        return "Other"

    grouped = {room: [] for room, _ in ROOM_RULES}

    for filename, caption in captions:
        room = classify_room(filename, caption)
        grouped[room].append(f"{filename}: {caption}")

    # Display grouped captions
    for room, items in grouped.items():
        if items:
            st.subheader(room)
            for c in items:
                st.markdown(f"- **{c}**")

    # Quick counts
    counts = {room: len(items) for room, items in grouped.items()}
    st.caption(f"Counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

### THE CODE TO DISPLAY THE CAPTIONS FOR EACH ROOM ENDS
