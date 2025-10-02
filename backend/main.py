import streamlit as st
import tempfile
from backend.transcription_local import transcribe_audio
from backend.summarization_local import summarize_text
from backend.faiss_utils import store_in_faiss, semantic_search

# --- Streamlit App ---
st.set_page_config(page_title="MinuteGenie", page_icon="📝", layout="wide")
st.title("📝 MinuteGenie - Meeting Notes AI")

# ---------------------------
# Audio Upload and Processing
# ---------------------------
st.header("Upload Meeting Audio")
uploaded_file = st.file_uploader("Upload audio file (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_file_path = tmp.name

    st.audio(tmp_file_path, format="audio/wav")
    st.info("Transcribing audio...")

    transcript = transcribe_audio(tmp_file_path)
    st.success("✅ Transcription complete!")
    st.text_area("Transcript", transcript, height=200)

    st.info("Summarizing transcript...")
    summary_json = summarize_text(transcript)

    # Safely extract summary text
    if isinstance(summary_json, dict) and "summary_text" in summary_json:
        summary_text = summary_json["summary_text"]
    else:
        summary_text = str(summary_json)  # fallback

    st.success("✅ Summarization complete!")
    st.text_area("Summary", summary_text, height=150)

    # Store summary in FAISS (pass string only)
    store_in_faiss(summary_text)
    st.success("Summary stored in semantic search index.")


# --- Section 1: Store New Summary ---
st.subheader("Store a New Summary")
summary_text = st.text_area("Enter meeting summary here:")

if st.button("Store Summary"):
    summary_text = summary_text.strip()
    if summary_text:
        store_in_faiss(summary_text)
        st.success("✅ Summary stored successfully!")
    else:
        st.warning("⚠️ Please enter a summary before storing.")

st.markdown("---")

# --- Section 2: Semantic Search ---
st.subheader("Search Previous Summaries")
query = st.text_input("Enter your search query:")

if query.strip():
    results = semantic_search(query.strip())
    if results:
        st.success(f"Top {len(results)} results:")
        combined_text = ""
        for i, res in enumerate(results, 1):
            st.markdown(f"**Result {i}:** {res['summary']}")
            st.write(f"Similarity Score: {res['score']:.3f}")
            st.markdown("---")
            combined_text += f"Result {i}:\n{res['summary']}\nSimilarity Score: {res['score']:.3f}\n\n"

        # Download button for search results
        st.download_button(
            label="📥 Download Results",
            data=combined_text,
            file_name="semantic_search_results.txt",
            mime="text/plain"
        )
    else:
        st.info("No matching summaries found in the index.")

