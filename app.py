from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="PhotoMind AI", layout="wide", initial_sidebar_state="collapsed")

# Hybrid taxonomy: broad folders remain familiar, while CLIP selects finer labels
# from an expandable vocabulary. These are candidate labels, not a closed universe.
TAXONOMY = {
    "People": {
        "Portraits & Selfies": ["a portrait of one person", "a selfie of a person", "a close-up portrait photograph", "a person posing for a photograph"],
        "Groups & Friends": ["a group photograph of friends", "a family group photograph", "people standing together for a group photo", "a candid photo of people socializing"],
        "Work & Study": ["people studying together", "people collaborating around laptops", "students working on computers in a classroom", "people working together at a desk", "a person using a laptop in a real room"],
        "Daily Life": ["people cooking", "people shopping", "people walking outdoors", "people playing with children", "people talking together"],
        "Performance & Ceremony": ["people giving a presentation", "people at a wedding ceremony", "people on a stage performing", "people at a graduation ceremony"],
    },
    "Animals": {
        "Dogs": ["a dog", "a puppy", "a dog playing outdoors", "a close-up photograph of a dog"],
        "Cats": ["a cat", "a kitten", "a cat sitting indoors", "a close-up photograph of a cat"],
        "Birds": ["a bird", "a parrot", "a bird perched on a branch", "birds flying in the sky"],
        "Wildlife": ["a wild animal in nature", "a lion", "an elephant", "a deer in a forest", "a monkey outdoors"],
        "Other Animals": ["a horse", "cattle on a farm", "a fish underwater", "a rabbit", "a reptile"],
    },
    "Nature": {
        "Flowers & Plants": ["flowers in a garden", "a close-up of a flower", "green plants and leaves", "a tree"],
        "Forests & Mountains": ["a forest landscape", "mountains in nature", "a wooded trail", "a waterfall in a forest"],
        "Sky & Weather": ["clouds in the sky", "a sunset", "a sunrise", "a stormy sky", "a rainbow"],
        "Water & Beaches": ["a lake in nature", "a river", "ocean waves", "a sandy beach", "a waterfall"],
    },
    "Food": {
        "Meals & Dishes": ["a cooked meal on a plate", "a bowl of soup", "a restaurant dish", "a plate of rice and curry"],
        "Fruit & Vegetables": ["fresh fruit", "vegetables in a basket", "a sliced watermelon", "fresh berries"],
        "Drinks & Desserts": ["a cup of coffee", "a glass of juice", "a cake", "ice cream", "a dessert"],
    },
    "Vehicles": {
        "Cars & SUVs": ["a car", "a blue SUV", "a sport utility vehicle", "a side view of a car", "a professional car product photograph on a plain background", "a parked car in a catalogue photo"],
        "Motorcycles": ["a motorcycle", "a sports motorcycle", "a motorbike with two wheels and an engine", "a motorcycle photographed outdoors", "a motorcycle product photograph", "a close-up of a motorcycle"],
        "Commercial Vehicles": ["a bus", "a truck", "a delivery van", "a tractor"],
        "Other Transport": ["a bicycle", "a train", "an airplane", "a boat", "a helicopter"],
    },
    "Documents & Screenshots": {
        "Screenshots & Interfaces": ["a direct screenshot of a computer application interface", "a phone screenshot showing app icons and text", "a screenshot of a webpage", "a software interface captured as a screenshot"],
        "Documents & Receipts": ["a scanned document page", "a receipt with printed text", "a paper form", "a page of a book", "a printed invoice"],
        "Charts & Slides": ["a presentation slide", "a chart or graph", "a diagram with labels", "a spreadsheet screenshot"],
    },
    "Architecture & Places": {
        "Buildings & Landmarks": ["a building exterior", "a temple", "a church", "a monument", "a city landmark"],
        "Streets & Cities": ["a city street", "a road in a town", "a city skyline", "a market street"],
        "Interiors": ["the interior of a room", "a living room", "a classroom interior", "an office interior"],
    },
    "Art & Objects": {
        "Art & Illustrations": ["a drawing or illustration", "a painting", "digital artwork", "a cartoon illustration"],
        "Electronics & Gadgets": ["a smartphone product photo", "a laptop product photo", "electronic gadgets on a desk", "a camera"],
        "Fashion & Accessories": ["clothing displayed in a photograph", "shoes", "a handbag", "jewelry"],
        "Sports & Fitness": ["a football match", "a basketball", "a person exercising in a gym", "sports equipment", "a bicycle race"],
        "Other Objects": ["a household object", "furniture", "a tool", "a product photographed on a plain background"],
    },
}

EVENT_LABELS = [
    "a birthday party", "a wedding", "a graduation", "a family reunion", "a festival celebration",
    "a religious ceremony", "a holiday or vacation", "a sightseeing trip", "a beach holiday",
    "a hiking or camping trip", "a picnic", "a sports event", "a concert or live performance",
    "a school or college activity", "a classroom lesson", "a work meeting", "a conference or presentation",
    "a cooking activity", "a shopping trip", "a pet playing", "a casual everyday moment",
    "a vehicle product photo shoot", "a landscape photography session", "a meal at a restaurant",
]

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
MAX_FILES = 1500
MAX_ZIP_MB = 500

st.markdown("""
<style>
.block-container{max-width:1240px;padding:1.6rem 2rem 3rem}#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.pm-hero{position:relative;overflow:hidden;border-radius:24px;padding:2.25rem 2.35rem;background:linear-gradient(118deg,#101a2c 0%,#182c4c 53%,#315b91 100%);color:#fff;margin-bottom:1.65rem}
.pm-hero:after{content:"";position:absolute;width:300px;height:300px;border:1px solid #ffffff1c;border-radius:50%;right:-80px;top:-115px;box-shadow:0 0 0 35px #ffffff08,0 0 0 75px #ffffff05}
.pm-eyebrow{font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;font-weight:750;color:#b8d2ff;margin-bottom:.8rem}
.pm-title{font-size:3rem;line-height:1.05;letter-spacing:-.055em;font-weight:800;color:#fff;margin:0 0 .8rem}
.pm-lead{font-size:1.12rem;line-height:1.65;color:#e3ecfa;max-width:720px;margin:0}
.pm-tags{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.25rem}.pm-tag{font-size:.82rem;border:1px solid #ffffff38;background:#ffffff0b;color:#f1f6ff;border-radius:999px;padding:.42rem .72rem}
.pm-section-title{font-size:1.55rem;letter-spacing:-.03em;font-weight:760;margin:.2rem 0 .35rem}.pm-section-copy{font-size:1rem;opacity:.72;margin-bottom:1rem}
.pm-step-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:.85rem 0 1.8rem}.pm-step{border:1px solid rgba(130,145,165,.28);background:rgba(130,145,165,.045);border-radius:17px;padding:1.15rem 1.2rem;min-height:145px}
.pm-step-num{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:10px;background:#dce9ff;color:#1b3d70;font-weight:800;margin-bottom:.8rem}.pm-step h3{font-size:1.03rem;margin:0 0 .35rem;font-weight:750}.pm-step p{font-size:.91rem;line-height:1.5;opacity:.72;margin:0}
.pm-built{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:.8rem 0 1.6rem}.pm-built-item{border:1px solid rgba(130,145,165,.25);border-radius:14px;padding:.9rem 1rem}.pm-built-item b{display:block;font-size:.93rem;margin-bottom:.25rem}.pm-built-item span{font-size:.82rem;line-height:1.45;opacity:.7}
.pm-divider{height:1px;background:rgba(130,145,165,.25);margin:1.5rem 0}.pm-upload-title{font-size:1.55rem;font-weight:760;letter-spacing:-.03em;margin:0 0 .3rem}.pm-upload-copy{font-size:1rem;opacity:.72;margin-bottom:1rem}
div[data-testid="stFileUploader"]{border:1px solid rgba(130,145,165,.35);border-radius:16px;padding:.55rem;background:rgba(130,145,165,.035)}div[data-testid="stMetric"]{border:1px solid rgba(130,145,165,.25);border-radius:14px;padding:.8rem 1rem}.stButton>button,.stDownloadButton>button{border-radius:12px;min-height:2.9rem;font-weight:700}
@media(max-width:850px){.block-container{padding:1rem .9rem 2rem}.pm-hero{padding:1.6rem 1.3rem;border-radius:18px}.pm-title{font-size:2.35rem}.pm-lead{font-size:1rem}.pm-step-grid{grid-template-columns:1fr;gap:.7rem}.pm-step{min-height:0}.pm-built{grid-template-columns:repeat(2,minmax(0,1fr)}}
@media(max-width:420px){.pm-built{grid-template-columns:1fr}.pm-title{font-size:2rem}}
</style>
<div class="pm-hero"><div class="pm-eyebrow">AI-powered photo organization</div><div class="pm-title">PhotoMind AI</div>
<p class="pm-lead">Organize a growing photo library with flexible categories, meaningful subfolders, event suggestions and natural-language search.</p>
<div class="pm-tags"><span class="pm-tag">Hybrid dynamic taxonomy</span><span class="pm-tag">Event & activity suggestions</span><span class="pm-tag">Semantic photo search</span></div></div>
<div class="pm-section-title">From scattered photos to organized folders</div><div class="pm-section-copy">Upload, let AI suggest a structure, review it, then export.</div>
<div class="pm-step-grid"><div class="pm-step"><div class="pm-step-num">01</div><h3>Upload your collection</h3><p>Select images, a folder where supported, or ZIP archives.</p></div><div class="pm-step"><div class="pm-step-num">02</div><h3>Discover categories</h3><p>CLIP matches broad categories and finer labels; event themes are suggested separately.</p></div><div class="pm-step"><div class="pm-step-num">03</div><h3>Review and download</h3><p>Edit category and subcategory labels, search visually, and export organized folders as ZIP.</p></div></div>
<div class="pm-section-title">Designed to grow with your collection</div><div class="pm-built"><div class="pm-built-item"><b>Flexible taxonomy</b><span>Broad folders plus automatically selected subcategories.</span></div><div class="pm-built-item"><b>Event suggestions</b><span>Activities and occasions are inferred as reviewable labels.</span></div><div class="pm-built-item"><b>Semantic search</b><span>Search photos with phrases such as “blue car in a forest”.</span></div><div class="pm-built-item"><b>Optional face grouping</b><span>Experimental same-person clustering using face embeddings.</span></div></div><div class="pm-divider"></div>
""", unsafe_allow_html=True)

# ---- Model and text vocabulary ------------------------------------------------
def build_prompts():
    prompts, meta = [], []
    for category, subcategories in TAXONOMY.items():
        for subcategory, descriptions in subcategories.items():
            for description in descriptions:
                prompts.append(f"a photo of {description}")
                meta.append((category, subcategory, description))
    return prompts, meta

@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name)
    model.eval()
    prompts, meta = build_prompts()
    event_prompts = [f"a photograph showing {label}" for label in EVENT_LABELS]
    all_text = prompts + event_prompts
    inputs = processor(text=all_text, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, processor, torch, text_features, meta, len(prompts)

def encode_images(images, model, processor, torch):
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

def classify_batch(images, model, processor, torch, text_features, meta, n_category_prompts):
    image_features = encode_images(images, model, processor, torch)
    similarities = image_features @ text_features.T
    outputs = []
    for row in similarities:
        broad_scores = defaultdict(list)
        sub_scores = defaultdict(list)
        prompt_scores = []
        for idx, (category, subcategory, description) in enumerate(meta):
            score = float(row[idx])
            broad_scores[category].append(score)
            sub_scores[(category, subcategory)].append(score)
            prompt_scores.append((score, category, subcategory, description))
        category_scores = {}
        for category, vals in broad_scores.items():
            vals.sort(reverse=True)
            # Strongest match plus top descriptions: avoid a distracting object dominating.
            category_scores[category] = .70 * vals[0] + .30 * sum(vals[:3]) / min(3, len(vals))
        category = max(category_scores, key=category_scores.get)
        subcategory_scores = {
            sub: (.70 * max(vals) + .30 * sum(sorted(vals, reverse=True)[:2]) / min(2, len(vals)))
            for (cat, sub), vals in sub_scores.items() if cat == category
        }
        subcategory = max(subcategory_scores, key=subcategory_scores.get)
        # Independent event/activity prediction. This is a suggestion, not verified metadata.
        event_slice = row[n_category_prompts:]
        event_idx = int(event_slice.argmax().item())
        event_score = float(event_slice[event_idx])
        event_label = EVENT_LABELS[event_idx]
        ranked = sorted(category_scores.items(), key=lambda pair: pair[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
        outputs.append({
            "category": category, "subcategory": subcategory,
            "event": event_label, "event_score": event_score,
            "category_scores": category_scores, "margin": margin,
            "embedding": image_features[len(outputs)].detach().cpu().tolist(),
        })
    return outputs

# ---- File helpers -------------------------------------------------------------
def safe_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return (name or "photo")[:160]

def decode_image(data):
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1600, 1600))
        return im.copy()

def image_bytes_from_uploads(uploads):
    found = []
    seen_sources = set()
    for upload in uploads:
        raw = upload.getvalue()
        filename = getattr(upload, "name", "photo")
        suffix = Path(filename).suffix.lower()
        if suffix == ".zip":
            if len(raw) > MAX_ZIP_MB * 1024 * 1024:
                raise ValueError(f"{filename} exceeds {MAX_ZIP_MB} MB. Split it into smaller ZIP archives.")
            if not zipfile.is_zipfile(io.BytesIO(raw)):
                st.warning(f"Skipped {filename}: invalid ZIP archive.")
                continue
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for member in zf.infolist():
                    p = Path(member.filename)
                    if member.is_dir() or p.suffix.lower() not in EXTS:
                        continue
                    if p.is_absolute() or ".." in p.parts or member.file_size > 35 * 1024 * 1024:
                        continue
                    try:
                        data = zf.read(member)
                        image = decode_image(data)
                        key = (member.filename.lower(), len(data))
                        if key in seen_sources:
                            continue
                        seen_sources.add(key)
                        found.append({"name": safe_name(p.name), "relative": member.filename, "bytes": data, "image": image})
                    except (UnidentifiedImageError, OSError, ValueError, RuntimeError, zipfile.BadZipFile):
                        continue
        elif suffix in EXTS:
            try:
                image = decode_image(raw)
                key = (filename.lower(), len(raw))
                if key in seen_sources:
                    continue
                seen_sources.add(key)
                found.append({"name": safe_name(Path(filename).name), "relative": filename, "bytes": raw, "image": image})
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
                st.warning(f"Skipped {filename}: image format could not be opened.")
        else:
            st.warning(f"Skipped {filename}: upload images or ZIP archives.")
    if len(found) > MAX_FILES:
        raise ValueError(f"Found {len(found)} images; the limit is {MAX_FILES} per run.")
    return found

def unique_name(name, used):
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    candidate, n = f"{stem}{suffix}", 1
    while candidate.lower() in used:
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    used.add(candidate.lower())
    return candidate

def build_download(results):
    buf = io.BytesIO()
    used = defaultdict(set)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        manifest = ["PhotoMind AI - Organization manifest", "", "AI category, subcategory and event labels are suggestions. Review before relying on them.", "Same-person grouping is experimental and may make mistakes.", ""]
        for item in results:
            if not item.get("ok"):
                continue
            category = safe_name(item.get("final_category", item["category"]))
            subcategory = safe_name(item.get("final_subcategory", item["subcategory"]))
            event = safe_name(item.get("final_event", item["event"]))
            # Hybrid hierarchy: broad category / subcategory / optional event.
            folders = [category, subcategory]
            if item.get("include_event_folder") and event:
                folders.append(event)
            person_group = item.get("person_group", "Unassigned")
            if item.get("include_person_folder") and person_group != "Unassigned":
                folders.append("People Groups/" + safe_name(person_group))
            folder = "/".join(folders)
            name = unique_name(item["name"], used[folder])
            out.writestr(f"PhotoMind_Organized/{folder}/{name}", item["bytes"])
            manifest.append(f"{item['name']} -> {folder}/{name}")
        out.writestr("PhotoMind_Organized/organization_manifest.txt", "\n".join(manifest))
    buf.seek(0)
    return buf.getvalue()

# ---- Optional face identity grouping -----------------------------------------
@st.cache_resource(show_spinner=False)
def load_face_models():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    detector = MTCNN(image_size=160, margin=14, keep_all=True, post_process=True, device="cpu")
    encoder = InceptionResnetV1(pretrained="vggface2").eval()
    return detector, encoder

def group_same_people(results, threshold=0.72):
    """Create conservative face-embedding clusters; no face means ungrouped."""
    import torch
    detector, encoder = load_face_models()
    faces = []
    for idx, result in enumerate(results):
        detected = detector(result["image"])
        if detected is None:
            continue
        if detected.ndim == 3:
            detected = detected.unsqueeze(0)
        with torch.inference_mode():
            embeddings = encoder(detected)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Assign the most prominent detected face to the image to avoid grouping
        # a crowd photo with every person in it.
        face_embedding = embeddings[0].cpu()
        faces.append((idx, face_embedding))
    clusters = []
    for idx, embedding in faces:
        assigned = False
        for cluster in clusters:
            similarities = [float(torch.dot(embedding, member)) for member in cluster["embeddings"]]
            if similarities and max(similarities) >= threshold:
                cluster["indices"].append(idx)
                cluster["embeddings"].append(embedding)
                assigned = True
                break
        if not assigned:
            clusters.append({"indices": [idx], "embeddings": [embedding]})
    person_num = 0
    for cluster in clusters:
        if len(cluster["indices"]) < 2:
            continue
        person_num += 1
        for idx in cluster["indices"]:
            results[idx]["person_group"] = f"Person {person_num:02d}"
    return results

# ---- App state and upload UI --------------------------------------------------
if "pm_results" not in st.session_state:
    st.session_state.pm_results = None
if "pm_archive_name" not in st.session_state:
    st.session_state.pm_archive_name = "uploaded_photos"

st.markdown('<div class="pm-upload-title">Start organizing</div>', unsafe_allow_html=True)
st.markdown('<div class="pm-upload-copy">Choose how you want to add your photos.</div>', unsafe_allow_html=True)
tab_images, tab_folder, tab_zip = st.tabs(["Images", "Folder", "ZIP archive"])
with tab_images:
    image_uploads = st.file_uploader("Select image files", type=["jpg","jpeg","png","webp","bmp","tif","tiff","heic","heif"], accept_multiple_files=True, key="pm_images")
with tab_folder:
    folder_uploads = st.file_uploader("Select a photo folder", type=["jpg","jpeg","png","webp","bmp","tif","tiff","heic","heif"], accept_multiple_files="directory", key="pm_folder", help="Folder selection depends on browser support. ZIP works on most devices.")
with tab_zip:
    zip_uploads = st.file_uploader("Select ZIP archives", type=["zip"], accept_multiple_files=True, key="pm_zip")

uploads = list(image_uploads or []) + list(folder_uploads or []) + list(zip_uploads or [])
photos = []
if uploads:
    try:
        photos = image_bytes_from_uploads(uploads)
        c1, c2 = st.columns(2)
        c1.metric("Photos ready", len(photos))
        c2.metric("Items selected", len(uploads))
        if photos:
            with st.expander("Preview photos", expanded=True):
                cols = st.columns(5)
                for i, item in enumerate(photos[:10]):
                    with cols[i % 5]:
                        st.image(item["image"], caption=item["name"], use_container_width=True)
    except Exception as error:
        st.error(str(error))
        photos = []

with st.expander("Advanced organization options", expanded=False):
    include_event_folder = st.checkbox("Create event/activity subfolders in the ZIP", value=False, help="Event labels are visual suggestions, not reliable event metadata.")
    enable_face_grouping = st.checkbox("Experimental: group photos by same-person face embeddings", value=False, help="Requires downloading an additional face model on first use. Face matching may be wrong; use only with consent and review the output.")
    include_person_folder = st.checkbox("Include detected person groups as a folder level in the ZIP", value=False, help="Only applies when face grouping is enabled and a group is found.")
    st.caption("Face grouping uses an optional separate model. Photos without a detected face remain ungrouped. The app does not identify anyone by name.")

start = st.button("Analyze and organize photos", type="primary", use_container_width=True, disabled=not bool(photos))
if start:
    progress = st.progress(0)
    status = st.empty()
    try:
        with st.spinner("Loading CLIP. The first launch may take a few minutes..."):
            model, processor, torch, text_features, meta, n_prompts = load_model()
        results = []
        t0 = time.perf_counter()
        batch_size = 8
        total_batches = (len(photos) + batch_size - 1) // batch_size
        for batch_idx, offset in enumerate(range(0, len(photos), batch_size), start=1):
            batch = photos[offset:offset + batch_size]
            status.write(f"Analyzing batch {batch_idx} of {total_batches} · {min(offset + len(batch), len(photos))} of {len(photos)} photos")
            try:
                predictions = classify_batch([item["image"] for item in batch], model, processor, torch, text_features, meta, n_prompts)
                for item, prediction in zip(batch, predictions):
                    results.append({**item, **prediction, "ok": True, "review": "Review suggested" if prediction["margin"] < 0.025 else "Looks clear", "include_event_folder": include_event_folder, "include_person_folder": include_person_folder, "person_group": "Unassigned"})
            except Exception:
                for item in batch:
                    try:
                        prediction = classify_batch([item["image"]], model, processor, torch, text_features, meta, n_prompts)[0]
                        results.append({**item, **prediction, "ok": True, "review": "Review suggested" if prediction["margin"] < 0.025 else "Looks clear", "include_event_folder": include_event_folder, "include_person_folder": include_person_folder, "person_group": "Unassigned"})
                    except Exception as error:
                        results.append({**item, "category": "Needs Review", "subcategory": "Needs Review", "event": "Uncertain", "embedding": [], "margin": 0.0, "ok": False, "error": str(error), "review": "Could not analyze", "include_event_folder": include_event_folder, "include_person_folder": include_person_folder, "person_group": "Unassigned"})
            progress.progress(min(1.0, (offset + len(batch)) / len(photos)))
        if enable_face_grouping:
            try:
                with st.spinner("Detecting faces and grouping likely matches. This may take time..."):
                    results = group_same_people(results)
            except Exception as error:
                st.warning(f"Face grouping could not run: {error}. The other organization features are still available.")
        st.session_state.pm_results = results
        st.session_state.pm_elapsed = time.perf_counter() - t0
        status.success(f"Analysis complete · {sum(r['ok'] for r in results)} of {len(results)} photos ready.")
    except Exception as error:
        st.error("The AI model could not load. Check the deployment logs and dependencies.")
        st.exception(error)

# ---- Results, category editing and semantic search ---------------------------
if st.session_state.pm_results:
    results = st.session_state.pm_results
    st.divider()
    st.markdown("### Review your dynamic folders")
    st.write("Review and edit the broad category, subcategory and event/activity suggestion before export.")
    choices = []
    for r in results:
        choices.append({
            "Photo": r["name"],
            "AI category": r["category"],
            "Final category": r.get("final_category", r["category"]),
            "AI subcategory": r.get("subcategory", "Other"),
            "Final subcategory": r.get("final_subcategory", r.get("subcategory", "Other")),
            "Event/activity": r.get("final_event", r.get("event", "Uncertain")),
            "Person group": r.get("person_group", "Unassigned"),
            "Review cue": r.get("review", "Review suggested"),
        })
    df = pd.DataFrame(choices)
    edited = st.data_editor(
        df, use_container_width=True, hide_index=True, key="pm_dynamic_editor",
        disabled=["Photo", "AI category", "AI subcategory", "Person group", "Review cue"],
        column_config={
            "Final category": st.column_config.SelectboxColumn("Final category", options=list(TAXONOMY.keys()), required=True),
            "Final subcategory": st.column_config.TextColumn("Final subcategory", help="Editable: create your own subcategory label."),
            "Event/activity": st.column_config.TextColumn("Event/activity", help="Editable AI suggestion."),
        },
    )
    for i, r in enumerate(results):
        r["final_category"] = str(edited.iloc[i]["Final category"]).strip() or r["category"]
        r["final_subcategory"] = safe_name(str(edited.iloc[i]["Final subcategory"]).strip() or r["subcategory"])
        r["final_event"] = safe_name(str(edited.iloc[i]["Event/activity"]).strip() or r["event"])

    st.markdown("### Search your photos with a phrase")
    query = st.text_input("Describe a photo", placeholder="e.g. blue car in a forest, dog playing, people using laptops", key="pm_search_query")
    if query.strip():
        try:
            model, processor, torch, text_features, meta, n_prompts = load_model()
            text_inputs = processor(text=[query.strip()], return_tensors="pt", padding=True, truncation=True)
            with torch.inference_mode():
                query_embedding = model.get_text_features(**text_inputs)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            vectors = [r.get("embedding", []) for r in results]
            scored = []
            for idx, vector in enumerate(vectors):
                if vector:
                    score = float(query_embedding[0].cpu().numpy() @ __import__("numpy").array(vector))
                    scored.append((score, idx))
            scored.sort(reverse=True)
            if scored:
                st.caption("Semantic matches are ranked by CLIP similarity, not calibrated probabilities.")
                matches = [results[idx] for score, idx in scored[:min(12, len(scored))]]
                cols = st.columns(min(4, len(matches)))
                for i, item in enumerate(matches):
                    with cols[i % len(cols)]:
                        st.image(item["image"], caption=f"{item['name']} · {item['category']}", use_container_width=True)
        except Exception as error:
            st.warning(f"Search could not run: {error}")

    st.markdown("### Folder summary")
    counts = Counter(f"{r['final_category']} / {r['final_subcategory']}" for r in results if r.get("ok"))
    cols = st.columns(min(4, max(1, len(counts))))
    for i, (label, count) in enumerate(counts.most_common()):
        with cols[i % len(cols)]:
            st.metric(label, count)

    valid = []
    for r in results:
        item = r.copy()
        item["category"] = item.get("final_category", item["category"])
        item["subcategory"] = item.get("final_subcategory", item.get("subcategory", "Other"))
        item["event"] = item.get("final_event", item.get("event", "Uncertain"))
        if item.get("ok") and item["category"] and item["subcategory"]:
            valid.append(item)
    if valid:
        archive = build_download(valid)
        st.download_button("Download organized photo library (.zip)", data=archive, file_name="PhotoMind_Organized_Photos.zip", mime="application/zip", type="primary", use_container_width=True)
        st.success(f"Ready to download {len(valid)} photo(s) across {len(set((r['category'], r['subcategory']) for r in valid))} category/subcategory folders.")
        st.caption(f"Analysis time: {st.session_state.get('pm_elapsed', 0):.1f} seconds. CLIP similarities and event suggestions are not guaranteed labels.")

st.markdown('<div class="pm-divider"></div>', unsafe_allow_html=True)
st.caption("Powered by Streamlit · PyTorch · Hugging Face Transformers · OpenAI CLIP · Optional FaceNet-PyTorch")
