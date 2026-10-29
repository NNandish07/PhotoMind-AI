PhotoMind AI — Single-file Streamlit deployment

Files:
- app.py
- requirements.txt

Deploy on Streamlit Community Cloud:
1. Create a new public GitHub repository.
2. Upload app.py and requirements.txt to the repository root.
3. Go to https://share.streamlit.io/
4. Connect GitHub and select the repository.
5. Set the main file path to app.py and deploy.

Notes:
- The CLIP model downloads from Hugging Face when AI is first used; the first analysis can take time.
- Streamlit Community Cloud resources are limited. PyTorch + CLIP may hit memory or startup limits on some free deployments.
- Photos are held in the app session, not permanently stored. They may disappear when the session resets.
- This single-file version does not include persistent cloud storage or identity-based face recognition.
- Review AI classifications before relying on them.
