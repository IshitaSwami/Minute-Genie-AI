A local personal assistant that transcribes meetings, generate concise summaries, and enable semantic for quick refrence.

Features:-
1. Audio to Text - Local Transcription using faster-whisper (CPU friendly - tiny/small)
2. Automated summaries - Pretrained facebook/bart-large-cnn model.
3. Semantic search - FAISS based search over recent summaries or meetings
4. Interative frontend: - Streamlit app for uploading audio, viewing transcripts, summaries and search results.
5. Deployment: - Docker ready for local or cloud use.

Setup:-
1. Clone and install-
   
   git clone <repo_url>
   
   cd <repo_name>
   
   python -m venv venv
   
   source venv/bin/activate   # Linux
   
   venv\Scripts\activate      # Windows
   
   pip install -r requirements.txt
   


   
   
2. Run the app-
   streamlit run app.py


   
   
4. Docker deployment-
   docker build -t meeting-assistant .
   docker run -p 8501:8501 meeting-assistant

