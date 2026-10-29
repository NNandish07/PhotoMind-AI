import io
import os
import re
import json
import zipfile
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFile
import imagehash

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ----------------------------- App configuration -----------------------------
st.set_page_config(
    page_title="PhotoMind AI | Intelligent Photo Library",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0b0d12; --panel:#131720; --line:#252b38; --muted:#9aa4b5; --text:#f3f5fa; --accent:#a78bfa; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% 0%, rgba(124,88,255,.12), transparent 32%), #0b0d12; color:var(--text); }
[data-testid="stSidebar"] { background:#10131b; border-right:1px solid #252b38; }
[data-testid="stHeader"] { background:rgba(11,13,18,.88); }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
.hero { padding: 1.6rem 1.7rem; border:1px solid #292d3c; border-radius:24px;
background:linear-gradient(120deg,rgba(124,88,255,.17),rgba(37,99,235,.08) 55%,rgba(16,185,129,.05));
margin-bottom:1.2rem; }
.eyebrow { color:#c4b5fd; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero h1 { font-size:clamp(2rem,4vw,3.1rem); line-height:1.1; margin:.4rem 0 .65rem; }
.hero p { color:#aeb7c7; font-size:1.02rem; max-width:760px; margin-bottom:0; }
.metric { border:1px solid #282e3b; background:linear-gradient(145deg,#171b26,#11141c); border-radius:18px; padding:1rem 1.1rem; }
.metric-label { color:#9aa4b5; font-size:.82rem; }
.metric-value { font-family:'Manrope',sans-serif; font-size:1.75rem; font-weight:800; color:#f5f3ff; margin-top:.15rem; }
.metric-note { color:#a78bfa; font-size:.75rem; margin-top:.1rem; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.35rem; font-weight:800; margin:.5rem 0 .25rem; }
.subtle { color:#9aa4b5; }
div.stButton > button, div.stDownloadButton > button { border-radius:12px; border:1px solid #353b4b; min-height:2.7rem; font-weight:650; }
div.stButton > button[kind="primary"] { background:linear-gradient(120deg,#8b5cf6,#6366f1); border:0; color:white; }
div[data-testid="stFileUploader"] { border:1px dashed #42475a; border-radius:16px; background:rgba(255,255,255,.015); padding:.4rem; }
.photo-card { background:#141822; border:1px solid #282e3b; border-radius:15px; padding:.75rem; }
div[data-testid="stExpander"] { border:1px solid #282e3b; border-radius:12px; }
.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea { border-radius:10px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

CATEGORIES = {
    "People": ["Portrait", "Family", "Friends", "Selfie", "Group photo", "Baby / Kids", "Wedding"],
    "Places": ["Travel", "Beach", "Mountains", "City", "Nature", "Landmark", "Architecture"],
    "Events": ["Birthday", "Wedding", "Graduation", "Festival", "Concert", "Party", "Ceremony"],
    "Food": ["Food", "Dessert", "Drinks", "Restaurant"],
    "Animals": ["Pets", "Dogs", "Cats", "Wildlife", "Birds"],
    "Everyday": ["Documents", "Screenshots", "Receipts", "Products", "Workspace", "Home"],
    "Nature": ["Flowers", "Sunset", "Sky", "Plants", "Waterfall", "Forest"],
    "Other": ["Other"],
}
CLIP_LABELS = [
    "a portrait photograph of a person", "a group photo of friends or family",
    "a selfie", "a wedding or ceremony photograph", "a birthday party photograph",
    "a travel photograph of a landmark", "a beach photograph", "a mountain landscape",
    "a city street or building", "a nature landscape with trees", "a sunset or sunrise",
    "a photograph of food or a meal", "a dessert or cake", "a photograph of a dog",
    "a photograph of a cat", "a wildlife or bird photograph", "a flower or plant photograph",
    "a screenshot from a phone or computer", "a document or scanned page",
    "a product photograph", "a room interior or home photograph",
    "a car or motorcycle photograph", "a sports activity photograph",
    "a concert or live performance photograph", "a graduation photograph",
]
LABEL_MAP = {
    "a portrait photograph of a person": ("People", "Portrait"),
    "a group photo of friends or family": ("People", "Group photo"),
    "a selfie": ("People", "Selfie"),
    "a wedding or ceremony photograph": ("Events", "Wedding"),
    "a birthday party photograph": ("Events", "Birthday"),
    "a travel photograph of a landmark": ("Places", "Landmark"),
    "a beach photograph": ("Places", "Beach"),
    "a mountain landscape": ("Places", "Mountains"),
    "a city street or building": ("Places", "Architecture"),
    "a nature landscape with trees": ("Nature", "Forest"),
    "a sunset or sunrise": ("Nature", "Sunset"),
    "a photograph of food or a meal": ("Food", "Food"),
    "a dessert or cake": ("Food", "Dessert"),
    "a photograph of a dog": ("Animals", "Dogs"),
    "a photograph of a cat": ("Animals", "Cats"),
    "a wildlife or bird photograph": ("Animals", "Wildlife"),
    "a flower or plant photograph": ("Nature", "Flowers"),
    "a screenshot from a phone or computer": ("Everyday", "Screenshots"),
    "a document or scanned page": ("Everyday", "Documents"),
    "a product photograph": ("Everyday", "Products"),
    "a room interior or home photograph": ("Everyday", "Home"),
    "a car or motorcycle photograph": ("Everyday", "Other"),
    "a sports activity photograph": ("Events", "Other"),
    "a concert or live performance photograph": ("Events", "Concert"),
    "a graduation photograph": ("Events", "Graduation"),
}

@st.cache_resource(show_spinner=False)
def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    import torch
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return model, processor, torch

def clip_classify(img):
    model, processor, torch = load_clip()
    inputs = processor(text=CLIP_LABELS, images=img, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    label = CLIP_LABELS[idx]
    cat, sub = LABEL_MAP.get(label, ("Other", "Other"))
    return cat, sub, float(probs[idx]), probs

def clip_image_embedding(img):
    model, processor, torch = load_clip()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype(np.float32)

def clip_text_embedding(query):
    model, processor, torch = load_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype(np.float32)

def get_bytes(upload):
    upload.seek(0)
    return upload.read()

def open_image(data):
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((1200, 1200))
    return im

def digest(data):
    return hashlib.sha256(data).hexdigest()

def p_hash(data):
    return str(imagehash.phash(open_image(data)))

def event_hint(category, subcategory, name):
    text = (category + " " + subcategory + " " + name).lower()
    rules = [
        (["birthday", "cake", "balloon"], "Birthday / celebration"),
        (["wedding", "bride", "groom", "ceremony"], "Wedding / ceremony"),
        (["beach", "mountain", "travel", "landmark"], "Travel memories"),
        (["graduation", "convocation"], "Graduation"),
        (["concert", "stage", "performance"], "Live event"),
        (["food", "dessert", "restaurant"], "Food moments"),
        (["sunset", "flower", "forest", "nature"], "Nature collection"),
    ]
    for words, event in rules:
        if any(w in text for w in words):
            return event
    return "Everyday moments"

def unpack_zip(upload):
    files = []
    try:
        with zipfile.ZipFile(upload) as z:
            for info in z.infolist():
                if info.is_dir() or info.file_size > 30 * 1024 * 1024:
                    continue
                name = info.filename
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif")):
                    files.append((Path(name).name, z.read(info)))
    except Exception as e:
        st.error(f"Could not read ZIP: {e}")
    return files

def init_state():
    if "photos" not in st.session_state:
        st.session_state.photos = []
    if "view" not in st.session_state:
        st.session_state.view = "Overview"
    if "processed_hashes" not in st.session_state:
        st.session_state.processed_hashes = set()

init_state()

with st.sidebar:
    st.markdown("## ✦ PhotoMind AI")
    st.caption("Your intelligent photo library")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Import photos", "All photos", "Smart search", "Events", "Duplicates", "Export library"], label_visibility="visible")
    st.divider()
    st.markdown("**AI engine**")
    st.caption("CLIP vision-language model")
    st.caption("Model downloads when AI features are first used.")
    st.divider()
    st.markdown("<div class='subtle' style='font-size:.78rem'>Private session workspace · Photos are processed in this app session.</div>", unsafe_allow_html=True)

photos = st.session_state.photos

def metric(label, value, note):
    st.markdown(f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)

def photo_grid(items, editable=False, key_prefix="photo"):
    if not items:
        st.info("No photos here yet. Import a few images to get started.")
        return
    cols = st.columns(4, gap="medium")
    for i, p in enumerate(items):
        with cols[i % 4]:
            st.image(p["image"], use_container_width=True)
            st.markdown(f"**{p['name'][:38]}**")
            st.caption(f"{p['category']} · {p['subcategory']}")
            st.caption(f"AI confidence: {p['confidence']:.0%} · {p['event']}")
            if editable:
                with st.expander("Edit details", expanded=False):
                    cat = st.selectbox("Category", list(CATEGORIES), index=list(CATEGORIES).index(p["category"]) if p["category"] in CATEGORIES else 0, key=f"{key_prefix}_cat_{p['id']}")
                    subs = CATEGORIES[cat]
                    sub = st.selectbox("Subcategory", subs, index=subs.index(p["subcategory"]) if p["subcategory"] in subs else 0, key=f"{key_prefix}_sub_{p['id']}")
                    tags = st.text_input("Tags (comma separated)", value=p.get("tags", ""), key=f"{key_prefix}_tags_{p['id']}")
                    if st.button("Save changes", key=f"{key_prefix}_save_{p['id']}"):
                        p["category"], p["subcategory"], p["tags"] = cat, sub, tags
                        p["event"] = event_hint(cat, sub, p["name"])
                        st.success("Saved.")
                        st.rerun()

# ----------------------------- Overview -----------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">A calmer way to find your memories</div>
      <h1>Your photos, <span style="color:#c4b5fd">understood.</span></h1>
      <p>Turn a messy camera roll into a searchable, thoughtfully organized photo library — powered by vision AI.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric("TOTAL PHOTOS", f"{len(photos):,}", "In this session")
    with c2: metric("CATEGORIES", len(set(p["category"] for p in photos)), "AI-organized")
    with c3: metric("EVENT COLLECTIONS", len(set(p["event"] for p in photos)), "Suggested moments")
    with c4: metric("DUPLICATE CANDIDATES", sum(p.get("duplicate_candidate", False) for p in photos), "Review before deleting")
    st.markdown("<div class='section-title'>A little more order. A lot more memory.</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtle'>Import photos to unlock smart categorization, natural-language search, event suggestions and duplicate review.</p>", unsafe_allow_html=True)
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("### ✨ Built for your photo library")
        st.markdown("- **Vision AI** suggests a category and subcategory for each photo.\n- **Semantic search** finds photos from natural-language descriptions.\n- **Event suggestions** collect related moments.\n- **Duplicate review** flags exact matches and visually similar images.\n- **Editable metadata** lets you stay in control.")
        if st.button("＋ Import your photos", type="primary", use_container_width=True):
            st.session_state["jump_import"] = True
            st.rerun()
    with b:
        if photos:
            st.markdown("### Recently added")
            photo_grid(photos[-4:][::-1], key_prefix="recent")
        else:
            st.markdown("<div class='photo-card' style='padding:2rem;text-align:center'><div style='font-size:2.5rem'>▧</div><h3>Your next favourite memory</h3><p class='subtle'>Start with a few photos. PhotoMind will do the sorting.</p></div>", unsafe_allow_html=True)

# ----------------------------- Import -----------------------------
elif page == "Import photos":
    st.markdown("<div class='hero'><div class='eyebrow'>Library intake</div><h1>Bring your memories in.</h1><p>Choose images or a ZIP archive. PhotoMind will analyze each photo and suggest an organization.</p></div>", unsafe_allow_html=True)
    st.warning("AI analysis downloads the CLIP model the first time it runs. On free hosting, model startup and memory limits may affect performance.")
    uploads = st.file_uploader("Choose photos", type=["jpg","jpeg","png","webp","bmp","tif","tiff","heic","heif"], accept_multiple_files=True, help="You can select multiple files at once.")
    zip_upload = st.file_uploader("Or import a ZIP archive", type=["zip"], accept_multiple_files=False)
    if st.button("✨ Analyze and organize", type="primary", use_container_width=True):
        batch = []
        if uploads:
            batch.extend([(u.name, get_bytes(u)) for u in uploads])
        if zip_upload:
            batch.extend(unpack_zip(zip_upload))
        if not batch:
            st.error("Choose at least one image or ZIP archive first.")
        else:
            progress = st.progress(0, text="Preparing your photos…")
            status = st.empty()
            added, skipped, failed = 0, 0, 0
            for idx, (name, data) in enumerate(batch):
                try:
                    sha = digest(data)
                    if sha in st.session_state.processed_hashes:
                        skipped += 1
                        progress.progress((idx+1)/len(batch), text=f"Skipping duplicate: {name}")
                        continue
                    img = open_image(data)
                    status.write(f"Analyzing **{name}** ({idx+1} of {len(batch)})")
                    cat, sub, conf, probs = clip_classify(img)
                    emb = clip_image_embedding(img)
                    ph = p_hash(data)
                    candidate = False
                    for old in st.session_state.photos:
                        try:
                            if imagehash.hex_to_hash(ph) - imagehash.hex_to_hash(old["phash"]) <= 5:
                                candidate = True
                                break
                        except Exception:
                            pass
                    item = {
                        "id": sha[:12], "name": name, "data": data, "image": img,
                        "category": cat, "subcategory": sub, "confidence": conf,
                        "event": event_hint(cat, sub, name), "sha256": sha, "phash": ph,
                        "embedding": emb, "tags": "", "duplicate_candidate": candidate,
                        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    st.session_state.photos.append(item)
                    st.session_state.processed_hashes.add(sha)
                    added += 1
                except Exception as e:
                    failed += 1
                    st.warning(f"Couldn't analyze {name}: {e}")
                progress.progress((idx+1)/len(batch), text=f"Processed {idx+1} of {len(batch)}")
            status.empty()
            st.success(f"Finished. Added {added} photo(s), skipped {skipped} exact duplicate(s), failed {failed}.")
            st.info("This demo keeps photos in the current app session. For durable cloud storage across sessions, connect persistent object storage and a database.")
            if added:
                st.balloons()
                st.rerun()

# ----------------------------- All photos -----------------------------
elif page == "All photos":
    st.markdown("<div class='hero'><div class='eyebrow'>Your collection</div><h1>Every photo, in its place.</h1><p>Review AI suggestions, filter the collection, and edit details whenever you like.</p></div>", unsafe_allow_html=True)
    if photos:
        f1, f2, f3 = st.columns(3)
        with f1: category_filter = st.selectbox("Category", ["All categories"] + list(CATEGORIES))
        with f2: event_filter = st.selectbox("Event", ["All events"] + sorted(set(p["event"] for p in photos)))
        with f3: name_filter = st.text_input("Filename contains", placeholder="e.g. IMG_")
        filtered = [p for p in photos if (category_filter == "All categories" or p["category"] == category_filter) and (event_filter == "All events" or p["event"] == event_filter) and name_filter.lower() in p["name"].lower()]
        st.caption(f"Showing {len(filtered)} of {len(photos)} photos")
        photo_grid(filtered, editable=True, key_prefix="all")
    else:
        st.info("Your library is empty. Open **Import photos** to begin.")

# ----------------------------- Semantic search -----------------------------
elif page == "Smart search":
    st.markdown("<div class='hero'><div class='eyebrow'>Search by meaning</div><h1>Describe the moment.</h1><p>Search by what's in the photo, not just by its filename.</p></div>", unsafe_allow_html=True)
    query = st.text_input("What are you looking for?", placeholder="e.g. a golden sunset at the beach, my dog in the garden, birthday cake")
    top_n = st.slider("Number of results", 1, 24, 8)
    if st.button("Search my library", type="primary"):
        if not photos:
            st.info("Import photos before searching.")
        elif not query.strip():
            st.info("Enter a description to search.")
        else:
            with st.spinner("Matching your description to your photos…"):
                try:
                    q_emb = clip_text_embedding(query.strip())
                    ranked = sorted(photos, key=lambda p: float(np.dot(p["embedding"], q_emb)), reverse=True)[:top_n]
                    st.session_state["search_results"] = [(p, float(np.dot(p["embedding"], q_emb))) for p in ranked]
                except Exception as e:
                    st.error(f"Search couldn't run: {e}")
    if st.session_state.get("search_results"):
        results = st.session_state.search_results
        cols = st.columns(4, gap="medium")
        for i, (p, score) in enumerate(results):
            with cols[i % 4]:
                st.image(p["image"], use_container_width=True)
                st.markdown(f"**{p['name'][:36]}**")
                st.caption(f"Match score · {score:.0%}")
                st.caption(f"{p['category']} / {p['subcategory']}")

# ----------------------------- Events -----------------------------
elif page == "Events":
    st.markdown("<div class='hero'><div class='eyebrow'>Memory lanes</div><h1>Moments, brought together.</h1><p>Suggested collections based on AI categories and filename clues.</p></div>", unsafe_allow_html=True)
    if not photos:
        st.info("Import photos to generate event suggestions.")
    else:
        groups = {}
        for p in photos:
            groups.setdefault(p["event"], []).append(p)
        for event, items in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            with st.expander(f"{event}  ·  {len(items)} photo(s)", expanded=(len(groups) <= 3)):
                photo_grid(items, key_prefix="event_" + re.sub(r"\W+", "", event))

# ----------------------------- Duplicates -----------------------------
elif page == "Duplicates":
    st.markdown("<div class='hero'><div class='eyebrow'>Library cleanup</div><h1>Spot the lookalikes.</h1><p>Review exact duplicates and visually similar photos before you export your collection.</p></div>", unsafe_allow_html=True)
    if not photos:
        st.info("Import photos to check for duplicates.")
    else:
        candidates = [p for p in photos if p.get("duplicate_candidate")]
        exact_groups = {}
        for p in photos:
            exact_groups.setdefault(p["sha256"], []).append(p)
        exact = [group for group in exact_groups.values() if len(group) > 1]
        st.caption("Exact duplicates are skipped during import. Similarity candidates use perceptual hashes and should be reviewed manually.")
        if candidates:
            st.markdown("### Similarity candidates")
            photo_grid(candidates, key_prefix="dupe")
        else:
            st.success("No visually similar candidates flagged so far.")

# ----------------------------- Export -----------------------------
elif page == "Export library":
    st.markdown("<div class='hero'><div class='eyebrow'>Take your memories with you</div><h1>Export your library.</h1><p>Download organized photos and a manifest so your classifications travel with your files.</p></div>", unsafe_allow_html=True)
    if not photos:
        st.info("Import photos before exporting.")
    else:
        st.markdown(f"**{len(photos)} photos** ready to export.")
        if st.button("Build organized ZIP", type="primary"):
            with st.spinner("Preparing your organized archive…"):
                buf = io.BytesIO()
                rows = []
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in photos:
                        safe_name = re.sub(r'[<>:"/\\\\|?*]', "_", p["name"])
                        folder = re.sub(r'[<>:"/\\\\|?*]', "_", p["category"])
                        sub = re.sub(r'[<>:"/\\\\|?*]', "_", p["subcategory"])
                        zf.writestr(f"{folder}/{sub}/{safe_name}", p["data"])
                        rows.append({
                            "filename": p["name"], "category": p["category"],
                            "subcategory": p["subcategory"], "event_suggestion": p["event"],
                            "ai_confidence": round(p["confidence"], 4), "tags": p.get("tags", ""),
                            "sha256": p["sha256"],
                        })
                    zf.writestr("manifest.csv", pd.DataFrame(rows).to_csv(index=False))
                    zf.writestr("README.txt", "Organized by PhotoMind AI. Categories are AI suggestions and may be edited.\n")
                st.session_state["export_zip"] = buf.getvalue()
        if st.session_state.get("export_zip"):
            st.download_button("⬇ Download organized ZIP", data=st.session_state.export_zip, file_name="PhotoMind_Organized_Library.zip", mime="application/zip", type="primary", use_container_width=True)

st.markdown("<div style='text-align:center;color:#646d7e;font-size:.78rem;padding:2rem 0 1rem'>PHOTOMIND AI · A smarter home for your memories</div>", unsafe_allow_html=True)
