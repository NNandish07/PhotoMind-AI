# PhotoMind AI — Hybrid Dynamic Organizer

A Streamlit photo organizer using CLIP embeddings, a hybrid broad-category/fine-label taxonomy, event/activity suggestions, semantic text-to-image search, editable folders and ZIP export.

## Run locally
1. Use Python 3.11 or 3.12.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Run: `python -m streamlit run app.py`.

## Features
- Upload individual images, folder uploads where supported, or ZIP archives.
- Hybrid organization: familiar broad folders plus selected subcategories from a richer candidate vocabulary. Subcategory labels can be edited or added by the user.
- Event/activity suggestions are a separate visual label and are not verified EXIF or event metadata.
- Natural-language search ranks uploaded photos against a user query using CLIP embeddings.
- Optional same-person grouping uses FaceNet-PyTorch face embeddings. It is experimental, uses the most prominent detected face per image, does not identify people by name, and can make false matches. Review the groups carefully and use consent-aware handling of personal photos.
- Review category, subcategory and event labels before downloading a ZIP with nested folders.

## Deployment
Push `app.py`, `requirements.txt`, and `README.md` to GitHub and deploy `app.py` through Streamlit Community Cloud. First startup downloads model weights and can take several minutes. The optional face feature downloads an additional model on first use. If cloud resource limits are hit, remove `facenet-pytorch` from requirements and leave face grouping disabled, or deploy to a host with more memory.

## Limitations
- CLIP selects from an expandable candidate taxonomy; it does not invent unrestricted labels from scratch. The hierarchy can be extended by adding candidate prompts.
- Event/activity labels are visual guesses, not proof of an event.
- Face clustering is experimental, not identity verification. It should not be used for high-stakes identification.
- CLIP similarity scores are not calibrated probabilities or accuracy estimates.
- Test with a labeled sample before large-scale use. Avoid uploading private photos to a public app without appropriate consent.
