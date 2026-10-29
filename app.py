
import io
import re
import json
import zipfile
import hashlib
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
except ImportError:
    pass

# ---------------- APP CONFIGURATION ----------------
st.set_page_config(
    page_title="PhotoMind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0b0d12; --panel:#131720; --line:#252b38; --muted:#9aa4b5; --text:#f3f5fa; --accent:#a78bfa; }
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 75% 0%,rgba(124,88,255,.12),transparent 32%),#0b0d12;color:var(--text);}
[data-testid="stSidebar"] {background:#10131b;border-right:1px solid #252b38;}
[data-testid="stHeader"] {background:rgba(11,13,18,.88);}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;}
.hero {padding:1.6rem 1.7rem;border:1px solid #292d3c;border-radius:24px;background:linear-gradient(120deg,rgba(124,88,255,.17),rgba(37,99,235,.08) 55%,rgba(16,185,129,.05));margin-bottom:1.2rem;}
.eyebrow {color:#c4b5fd;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700;}
.hero h1 {font-size:clamp(2rem,4vw,3.1rem);line-height:1.1;margin:.4rem 0 .65rem;}
.hero p {color:#aeb7c7;font-size:1.02rem;max-width:760px;margin-bottom:0;}
.metric {border:1px solid #282e3b;background:linear-gradient(145deg,#171b26,#11141c);border-radius:18px;padding:1rem 1.1rem;min-height:115px;}
.metric-label {color:#9aa4b5;font-size:.82rem;}
.metric-value {font-family:'Manrope',sans-serif;font-size:1.75rem;font-weight:800;color:#f5f3ff;margin-top:.15rem;}
.metric-note {color:#a78bfa;font-size:.75rem;margin-top:.1rem;}
.section-title {font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;margin:.5rem 0 .25rem;}
.subtle {color:#9aa4b5;}
div.stButton>button,div.stDownloadButton>button {border-radius:12px;border:1px solid #353b4b;min-height:2.7rem;font-weight:650;}
div.stButton>button[kind="primary"] {background:linear-gradient(120deg,#8b5cf6,#6366f1);border:0;color:white;}
div[data-testid="stFileUploader"] {border:1px dashed #42475a;border-radius:16px;background:rgba(255,255,255,.015);padding:.4rem;}
.photo-card {background:#141822;border:1px solid #282e3b;border-radius:15px;padding:.75rem;}
div[data-testid="stExpander"] {border:1px solid #282e3b;border-radius:12px;}
.stTextInput input,.stSelectbox div[data-baseweb="select"],.stTextArea textarea {border-radius:10px;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- CATEGORIES ----------------
CATEGORIES = {
    "People": ["Portrait", "Family", "Friends", "Selfie", "Group photo", "Baby / Kids", "Wedding"],
    "Places": ["Travel", "Beach", "Mountains", "City", "Nature", "Landmark", "Architecture"],
    "Events": ["Birthday", "Wedding", "Graduation", "Festival", "Concert", "Party", "Ceremony"],
    "Food": ["Food", "Dessert", "Drinks", "Restaurant"],
    "Animals": ["Pets", "Dogs", "Cats", "Wildlife", "Birds"],
    "Everyday": ["Documents", "Screenshots", "Receipts", "Products", "Workspace", "Home", "Other"],
    "Nature": ["Flowers", "Sunset", "Sky", "Plants", "Waterfall", "Forest"],
    "Other": ["Other"],
}

CLIP_LABELS = [
    "a portrait photograph of a person",
    "a group photo of friends or family",
    "a selfie",
    "a wedding or ceremony photograph",
    "a birthday party photograph",
    "a travel photograph of a landmark",
    "a beach photograph",
    "a mountain landscape",
    "a city street or building",
    "a nature landscape with trees",
    "a sunset or sunrise",
    "a photograph of food or a meal",
    "a dessert or cake",
    "a photograph of a dog",
    "a photograph of a cat",
    "a wildlife or bird photograph",
    "a flower or plant photograph",
    "a screenshot from a phone or computer",
    "a document or scanned page",
    "a product photograph",
    "a room interior or home photograph",
    "a car or motorcycle photograph",
    "a sports activity photograph",
    "a concert or live performance photograph",
    "a graduation photograph",
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

SUPPORTED = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif")


# ---------------- AI MODEL ----------------
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
    inputs = processor(
        text=CLIP_LABELS,
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        output = model(**inputs)
        probs = output.logits_per_image.softmax(dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    label = CLIP_LABELS[idx]
    category, subcategory = LABEL_MAP.get(label, ("Other", "Other"))
    return category, subcategory, float(probs[idx])


def clip_image_embedding(img):
    model, processor, torch = load_clip()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy().astype(np.float32)


def clip_text_embedding(query):
    model, processor, torch = load_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy().astype(np.float32)


# ---------------- UTILITIES ----------------
def get_bytes(upload):
    upload.seek(0)
    return upload.read()


def open_image(data):
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1200, 1200))
    return image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def p_hash(data):
    return str(imagehash.phash(open_image(data)))


def event_hint(category, subcategory, name):
    text = f"{category} {subcategory} {name}".lower()
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
        if any(word in text for word in words):
            return event
    return "Everyday moments"


def unpack_zip(upload):
    files = []
    try:
        with zipfile.ZipFile(upload) as archive:
            for item in archive.infolist():
                if item.is_dir() or item.file_size > 30 * 1024 * 1024:
                    continue
                if item.filename.lower().endswith(SUPPORTED):
                    files.append((Path(item.filename).name, archive.read(item)))
    except Exception as error:
        st.error(f"Could not read ZIP: {error}")
    return files


def init_state():
    if "photos" not in st.session_state:
        st.session_state.photos = []
    if "processed_hashes" not in st.session_state:
        st.session_state.processed_hashes = set()
    if "search_results" not in st.session_state:
        st.session_state.search_results = []


init_state()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ✦ PhotoMind AI")
    st.caption("Your intelligent photo library")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "Import photos", "All photos", "Smart search", "Export library"],
    )

    st.divider()
    st.markdown("**AI engine**")
    st.caption("CLIP vision-language model")
    st.divider()
    st.markdown(
        "<div style='color:#9aa4b5;font-size:.78rem'>Private session workspace. "
        "Photos remain available during the current app session.</div>",
        unsafe_allow_html=True,
    )


photos = st.session_state.photos


def metric(label, value, note):
    st.markdown(
        f"""<div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def photo_grid(items, editable=False, key_prefix="photo"):
    if not items:
        st.info("No photos here yet. Import some images to get started.")
        return

    cols = st.columns(4, gap="medium")
    for index, photo in enumerate(items):
        with cols[index % 4]:
            st.image(photo["image"], use_container_width=True)
            st.markdown(f"**{photo['name'][:38]}**")
            st.caption(f"{photo['category']} · {photo['subcategory']}")
            st.caption(f"AI confidence: {photo['confidence']:.0%}")

            if editable:
                with st.expander("Edit details"):
                    category = st.selectbox(
                        "Category",
                        list(CATEGORIES),
                        index=list(CATEGORIES).index(photo["category"])
                        if photo["category"] in CATEGORIES else 0,
                        key=f"{key_prefix}_cat_{photo['id']}",
                    )
                    subcategories = CATEGORIES[category]
                    subcategory = st.selectbox(
                        "Subcategory",
                        subcategories,
                        index=subcategories.index(photo["subcategory"])
                        if photo["subcategory"] in subcategories else 0,
                        key=f"{key_prefix}_sub_{photo['id']}",
                    )
                    tags = st.text_input(
                        "Tags (comma separated)",
                        value=photo.get("tags", ""),
                        key=f"{key_prefix}_tags_{photo['id']}",
                    )

                    if st.button("Save changes", key=f"{key_prefix}_save_{photo['id']}"):
                        photo["category"] = category
                        photo["subcategory"] = subcategory
                        photo["tags"] = tags
                        photo["event"] = event_hint(category, subcategory, photo["name"])
                        st.success("Changes saved.")
                        st.rerun()


# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">A calmer way to find your memories</div>
      <h1>Your photos, <span style="color:#c4b5fd">understood.</span></h1>
      <p>Turn a messy camera roll into a searchable, thoughtfully organized photo library, powered by vision AI.</p>
    </div>
    """, unsafe_allow_html=True)

    image_formats = {
        Path(photo["name"]).suffix.lower().lstrip(".").upper()
        for photo in photos if Path(photo["name"]).suffix
    }
    tagged_count = sum(bool(photo.get("tags", "").strip()) for photo in photos)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric("TOTAL PHOTOS", f"{len(photos):,}", "In your current session")
    with c2:
        category_count = len(set(photo["category"] for photo in photos))
        metric("CATEGORIES", category_count, "Organized by AI")
    with c3:
        metric("IMAGE FORMATS", len(image_formats), "Different file formats")
    with c4:
        metric("PHOTOS TAGGED", f"{tagged_count:,}", "With custom tags")

    st.markdown(
        "<div class='section-title'>A little more order. A lot more memory.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtle'>Import your photos to unlock smart categorization, natural-language search, "
        "and organized downloads.</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### ✨ Built for your photo library")
        st.markdown(
            "- **Vision AI:** suggests categories and subcategories.\n"
            "- **Smart search:** find images using natural-language descriptions.\n"
            "- **Flexible imports:** choose photos, a folder, or a ZIP archive.\n"
            "- **Editable details:** customize categories and tags.\n"
            "- **Organized export:** download categorized folders in a ZIP file."
        )
        if st.button("＋ Import your photos", type="primary", use_container_width=True):
            st.session_state["open_import"] = True
            st.rerun()

    with right:
        if photos:
            st.markdown("### Recently added")
            photo_grid(photos[-4:][::-1], key_prefix="recent")
        else:
            st.markdown(
                """<div class="photo-card" style="padding:2rem;text-align:center">
                <div style="font-size:2.5rem">▧</div>
                <h3>Your next favourite memory</h3>
                <p class="subtle">Start with a few photos. PhotoMind will do the sorting.</p>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------- IMPORT PHOTOS ----------------
elif page == "Import photos":
    st.markdown(
        """<div class="hero"><div class="eyebrow">Library intake</div>
        <h1>Bring your memories in.</h1>
        <p>Choose individual photos, an entire folder, or a ZIP archive.</p></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 1. Upload individual photos")
    uploads = st.file_uploader(
        "Choose image files",
        type=[ext.lstrip(".") for ext in SUPPORTED],
        accept_multiple_files=True,
        key="individual_uploads",
    )

    st.markdown("#### 2. Upload an entire folder")
    folder_uploads = st.file_uploader(
        "Choose a folder containing photos",
        type=[ext.lstrip(".") for ext in SUPPORTED],
        accept_multiple_files="directory",
        key="folder_uploads",
        help="Folder selection depends on browser support.",
    )

    st.markdown("#### 3. Import a ZIP archive")
    zip_upload = st.file_uploader(
        "Choose a ZIP file containing photos",
        type=["zip"],
        key="zip_upload",
    )

    if st.button("✨ Analyze and organize", type="primary", use_container_width=True):
        batch = []

        if uploads:
            batch.extend(
                (Path(upload.name).name, get_bytes(upload))
                for upload in uploads
                if Path(upload.name).suffix.lower() in SUPPORTED
            )

        if folder_uploads:
            batch.extend(
                (Path(upload.name).name, get_bytes(upload))
                for upload in folder_uploads
                if Path(upload.name).suffix.lower() in SUPPORTED
            )

        if zip_upload:
            batch.extend(unpack_zip(zip_upload))

        if not batch:
            st.error("Select photos, a folder, or a ZIP archive first.")
        else:
            progress = st.progress(0, text="Preparing photos...")
            status = st.empty()
            added = skipped = failed = 0

            for index, (name, data) in enumerate(batch):
                try:
                    sha = digest(data)
                    if sha in st.session_state.processed_hashes:
                        skipped += 1
                        progress.progress((index + 1) / len(batch))
                        continue

                    image = open_image(data)
                    status.write(f"Analyzing {name} ({index + 1}/{len(batch)})")

                    category, subcategory, confidence = clip_classify(image)
                    embedding = clip_image_embedding(image)
                    phash = p_hash(data)

                    similar = False
                    for old in st.session_state.photos:
                        try:
                            distance = (
                                imagehash.hex_to_hash(phash)
                                - imagehash.hex_to_hash(old["phash"])
                            )
                            if distance <= 5:
                                similar = True
                                break
                        except Exception:
                            continue

                    item = {
                        "id": sha[:12],
                        "name": name,
                        "data": data,
                        "image": image,
                        "category": category,
                        "subcategory": subcategory,
                        "confidence": confidence,
                        "event": event_hint(category, subcategory, name),
                        "sha256": sha,
                        "phash": phash,
                        "embedding": embedding,
                        "tags": "",
                        "duplicate_candidate": similar,
                        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }

                    st.session_state.photos.append(item)
                    st.session_state.processed_hashes.add(sha)
                    added += 1

                except Exception as error:
                    failed += 1
                    st.warning(f"Couldn't analyze {name}: {error}")

                progress.progress((index + 1) / len(batch))

            status.empty()
            st.success(
                f"Finished. Added {added}, skipped {skipped} exact duplicates, "
                f"failed {failed}."
            )
            if added:
                st.rerun()


# ---------------- ALL PHOTOS ----------------
elif page == "All photos":
    st.markdown(
        """<div class="hero"><div class="eyebrow">Your collection</div>
        <h1>Every photo, in its place.</h1>
        <p>Review AI suggestions and edit photo details.</p></div>""",
        unsafe_allow_html=True,
    )

    if photos:
        c1, c2, c3 = st.columns(3)
        with c1:
            category_filter = st.selectbox(
                "Category", ["All categories"] + list(CATEGORIES)
            )
        with c2:
            event_filter = st.selectbox(
                "Suggested group",
                ["All groups"] + sorted({photo["event"] for photo in photos}),
            )
        with c3:
            name_filter = st.text_input("Filename contains", placeholder="e.g. IMG_")

        filtered = [
            photo for photo in photos
            if (category_filter == "All categories" or photo["category"] == category_filter)
            and (event_filter == "All groups" or photo["event"] == event_filter)
            and name_filter.lower() in photo["name"].lower()
        ]
        st.caption(f"Showing {len(filtered)} of {len(photos)} photos")
        photo_grid(filtered, editable=True, key_prefix="all")
    else:
        st.info("Your library is empty. Open Import photos to begin.")


# ---------------- SMART SEARCH ----------------
elif page == "Smart search":
    st.markdown(
        """<div class="hero"><div class="eyebrow">Search by meaning</div>
        <h1>Describe the moment.</h1>
        <p>Search by what's in the photo, not just its filename.</p></div>""",
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "What are you looking for?",
        placeholder="A sunset at the beach, my dog in a garden...",
    )
    top_n = st.slider("Number of results", 1, 24, 8)

    if st.button("Search my library", type="primary"):
        if not photos:
            st.info("Import photos before searching.")
        elif not query.strip():
            st.info("Enter a description to search.")
        else:
            try:
                with st.spinner("Searching your photos..."):
                    query_embedding = clip_text_embedding(query.strip())
                    ranked = sorted(
                        photos,
                        key=lambda photo: float(
                            np.dot(photo["embedding"], query_embedding)
                        ),
                        reverse=True,
                    )[:top_n]
                    st.session_state.search_results = [
                        (photo, float(np.dot(photo["embedding"], query_embedding)))
                        for photo in ranked
                    ]
            except Exception as error:
                st.error(f"Search couldn't run: {error}")

    if st.session_state.search_results:
        cols = st.columns(4, gap="medium")
        for index, (photo, score) in enumerate(st.session_state.search_results):
            with cols[index % 4]:
                st.image(photo["image"], use_container_width=True)
                st.markdown(f"**{photo['name'][:36]}**")
                st.caption(f"Match score: {score:.0%}")
                st.caption(f"{photo['category']} / {photo['subcategory']}")


# ---------------- EXPORT LIBRARY ----------------
elif page == "Export library":
    st.markdown(
        """<div class="hero"><div class="eyebrow">Take your memories with you</div>
        <h1>Export your library.</h1>
        <p>Download organized photos and a CSV manifest.</p></div>""",
        unsafe_allow_html=True,
    )

    if not photos:
        st.info("Import photos before exporting.")
    else:
        st.markdown(f"**{len(photos)} photos** ready to export.")

        if st.button("Build organized ZIP", type="primary"):
            buffer = io.BytesIO()
            rows = []

            with zipfile.ZipFile(
                buffer, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for photo in photos:
                    safe_name = re.sub(r'[<>:"/\\\\|?*]', "_", photo["name"])
                    category = re.sub(
                        r'[<>:"/\\\\|?*]', "_", photo["category"]
                    )
                    subcategory = re.sub(
                        r'[<>:"/\\\\|?*]', "_", photo["subcategory"]
                    )

                    archive.writestr(
                        f"{category}/{subcategory}/{safe_name}",
                        photo["data"],
                    )

                    rows.append({
                        "filename": photo["name"],
                        "category": photo["category"],
                        "subcategory": photo["subcategory"],
                        "confidence": round(photo["confidence"], 4),
                        "tags": photo.get("tags", ""),
                        "sha256": photo["sha256"],
                    })

                archive.writestr(
                    "manifest.csv",
                    pd.DataFrame(rows).to_csv(index=False),
                )
                archive.writestr(
                    "README.txt",
                    "Organized by PhotoMind AI. Review AI suggestions before relying on them.\n",
                )

            st.session_state.export_zip = buffer.getvalue()

        if st.session_state.get("export_zip"):
            st.download_button(
                "⬇ Download organized ZIP",
                data=st.session_state.export_zip,
                file_name="PhotoMind_Organized_Library.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )


st.markdown(
    "<div style='text-align:center;color:#646d7e;font-size:.78rem;padding:2rem 0 1rem'>"
    "PHOTOMIND AI · A smarter home for your memories</div>",
    unsafe_allow_html=True,
)
