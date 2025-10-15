import os
import json
import faiss
import torch
import numpy as np
import requests
import io
import pdfplumber
from PIL import Image
import pytesseract
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template, session

# ===== CONFIG =====
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

FOLDER_LAW = ["data", "family"]
FOLDER_LAWYER = ["lawyer"]

INDEX_LAW = "embeddings/faiss_law.index"
INDEX_LAWYER = "embeddings/faiss_lawyer.index"
TEXTS_LAW = "cache/law.json"
TEXTS_LAWYER = "cache/lawyer.json"

# ===== INIT =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ ใช้งานบน: {device.upper()}")

app = Flask(__name__)
app.secret_key = "supersecretkey"

embedder = SentenceTransformer("intfloat/multilingual-e5-base")
embedder.to(device)

# ===== GLOBAL =====
law_texts, lawyer_texts = [], []
law_index, lawyer_index = None, None
lawyer_data = []


# ===== UTILITIES =====
def load_json_files(folder_paths: List[str]):
    texts = []
    for folder in folder_paths:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".json"):
                with open(os.path.join(folder, f), "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        texts.extend(data)
                    else:
                        texts.append(data)
    return texts


def embed_texts(texts: List[str]):
    return embedder.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )


def build_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_similar(query: str, texts: List[str], index, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True, device=device)
    scores, ids = index.search(q_emb, top_k)
    return [(texts[i], float(scores[0][idx])) for idx, i in enumerate(ids[0])]


def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


def extract_text_from_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img, lang="tha+eng").strip()


def is_positive_reply(message: str) -> bool:
    text = message.strip().lower()
    positive = ["ตกลง", "โอเค", "ได้", "ครับ", "ค่ะ", "ยินยอม", "yes", "ok", "agree"]
    negative = ["ไม่", "no", "ยัง", "ไม่ตกลง", "ไม่ครับ", "ไม่ค่ะ"]

    if any(n in text for n in negative):
        return False
    if any(p in text for p in positive):
        return True

    prompt = f"""
วิเคราะห์ว่า ข้อความนี้แปลว่า 'ตกลง' หรือ 'ยินยอม' หรือไม่
ข้อความ: "{message}"
ตอบ TRUE หรือ FALSE เท่านั้น
"""
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        reply = res.json().get("response", "").strip().lower()
        return "true" in reply
    except:
        return False


# ===== BUILD DATASET =====
def build_law_dataset():
    data = load_json_files(FOLDER_LAW)
    texts = [f"{d['law_name']} มาตรา {d['section_num']}: {d['section_content']}"
             for d in data if "law_name" in d and "section_content" in d]
    emb = embed_texts(texts)
    return texts, build_index(emb)


def build_lawyer_dataset():
    data = load_json_files(FOLDER_LAWYER)
    texts = [f"{l['name']} - ความเชี่ยวชาญ: {', '.join(l['expertise'])}"
             for l in data if "name" in l and "expertise" in l]
    emb = embed_texts(texts)
    return data, texts, build_index(emb)


# ===== ROUTES =====
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    แชทรวม:
    - ถ้ามีไฟล์ → สรุป + เก็บใน session
    - ถ้าเป็นข้อความ → ใช้ context + RAG + AI ตอบ
    """
    # ==== อัปโหลดไฟล์ ====
    if "file" in request.files:
        f = request.files["file"]
        name = f.filename.lower()
        fb = f.read()

        if name.endswith(".pdf"):
            text = extract_text_from_pdf(fb)
        elif name.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(fb)
        else:
            return jsonify({"error": "⚠️ รองรับเฉพาะ PDF หรือรูปภาพเท่านั้น"}), 400

        if not text:
            return jsonify({"error": "⚠️ ไม่พบข้อความในไฟล์"}), 400

        prompt = f"สรุปเอกสารต่อไปนี้ให้เข้าใจง่ายและแก้คำให้ถูกต้องสมเหตุสมผล:\n{text[:5000]}"
        res = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        summary = res.json().get("response", "").strip()
        session["chat_context"] = summary

        return jsonify({"type": "summary", "message": summary})

    # ==== ข้อความ ====
    data = request.get_json()
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "❌ โปรดพิมพ์คำถาม"}), 400

    # ตรวจว่าเป็นคำตอบยืนยันไหม
    if session.get("awaiting_confirm"):
        session["awaiting_confirm"] = False
        if is_positive_reply(msg):
            if "last_lawyer" in session:
                return jsonify({
                    "type": "lawyer",
                    "message": "📞 ข้อมูลติดต่อทนาย",
                    "lawyer": session["last_lawyer"]
                })
            else:
                return jsonify({"type": "text", "message": "❌ ไม่มีข้อมูลทนายล่าสุด"})
        else:
            return jsonify({"type": "text", "message": "โอเค หวังว่าคำตอบจะช่วยให้การตัดสินใจได้ง่ายขึ้น"})

    # ===== ค้นหากฎหมาย =====
    rag_contexts = search_similar(msg, law_texts, law_index)
    law_context = "\n\n".join([t for t, _ in rag_contexts])

    # ===== ค้นหาทนายที่เกี่ยวข้อง =====
    lawyer_matches = search_similar(msg, lawyer_texts, lawyer_index, top_k=3)
    related_lawyers = []
    for text, _ in lawyer_matches:
        for l in lawyer_data:
            if l["name"] in text:
                related_lawyers.append(l)
    if related_lawyers:
        session["last_lawyer"] = related_lawyers[0]

    # ===== รวม context ทั้งหมด =====
    user_context = session.get("chat_context", "")
    full_context = f"{user_context}\n\n{law_context}" if user_context else law_context

    # ===== ให้ AI ตอบ =====
    lawyer_suggestions = "\n".join([f"- {l['name']} ({', '.join(l['expertise'])})" for l in related_lawyers])
    prompt = f"""
คุณคือผู้เชี่ยวชาญด้านกฎหมายที่สามารถอธิบายข้อกฎหมายให้เข้าใจง่าย
คำถาม: {msg}

บริบท:
{full_context}

ทนายที่เกี่ยวข้อง:
{lawyer_suggestions}

กรุณาตอบให้เข้าใจง่าย และอ้างอิงมาตราที่เกี่ยวข้อง พร้อมยกตัวอย่างประกอบ (ถ้ามี) 
และลงท้ายด้วยคำว่า
"คุณต้องการติดต่อทนายไหม?"
"""
    res = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
    answer = res.json().get("response", "").strip()
    session["awaiting_confirm"] = True

    return jsonify({
        "type": "answer",
        "message": answer,
        "lawyers": related_lawyers
    })


# ===== MAIN =====
if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    if os.path.exists(TEXTS_LAW):
        with open(TEXTS_LAW, "r", encoding="utf-8") as f:
            law_texts = json.load(f)
        law_index = faiss.read_index(INDEX_LAW)
    else:
        law_texts, law_index = build_law_dataset()
        faiss.write_index(law_index, INDEX_LAW)
        with open(TEXTS_LAW, "w", encoding="utf-8") as f:
            json.dump(law_texts, f, ensure_ascii=False)

    if os.path.exists(TEXTS_LAWYER):
        with open(TEXTS_LAWYER, "r", encoding="utf-8") as f:
            lawyer_data = json.load(f)
        lawyer_texts = [f"{d['name']} - ความเชี่ยวชาญ: {', '.join(d['expertise'])}" for d in lawyer_data]
        lawyer_index = faiss.read_index(INDEX_LAWYER)
    else:
        lawyer_data, lawyer_texts, lawyer_index = build_lawyer_dataset()
        faiss.write_index(lawyer_index, INDEX_LAWYER)
        with open(TEXTS_LAWYER, "w", encoding="utf-8") as f:
            json.dump(lawyer_data, f, ensure_ascii=False)

    print("🎯 ระบบแชทกฎหมายพร้อมใช้งานแล้ว")
    app.run(host="0.0.0.0", port=8000, debug=True)
