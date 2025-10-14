import os
import json
import faiss
import torch
import numpy as np
import requests
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

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
embedder = SentenceTransformer("intfloat/multilingual-e5-base")
embedder.to(device)

# ===== GLOBAL =====
law_texts, lawyer_texts = [], []
law_index, lawyer_index = None, None
lawyer_data = []

# ===== LOAD FUNCTIONS =====
def load_json_files(folder_paths: List[str]):
    """โหลดไฟล์ JSON ทั้งหมดจากหลายโฟลเดอร์ และรวม list ข้างในให้อยู่ในลิสต์เดียว"""
    texts = []
    for folder in folder_paths:
        if not os.path.exists(folder):
            continue
        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
        for filename in tqdm(json_files, desc=f"📚 โหลด {folder}"):
            path = os.path.join(folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts.extend(data)
                    else:
                        texts.append(data)
            except Exception as e:
                print(f"⚠️ อ่านไฟล์ {filename} ไม่ได้: {e}")
    return texts


def embed_texts(texts: List[str], batch_size: int = 16):
    """สร้าง embedding จากข้อความทั้งหมด"""
    if not texts:
        raise ValueError("❌ ไม่มีข้อความสำหรับสร้าง embeddings")
    return embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )


def build_index(embeddings: np.ndarray):
    """สร้างดัชนี FAISS รองรับทั้ง GPU และ CPU"""
    print("📏 Embedding shape:", np.shape(embeddings))
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("❌ ไม่มี embeddings สำหรับสร้างดัชนี (index)")

    # สร้าง index แบบ Inner Product (เหมาะกับ cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # ✅ ตรวจว่ามี GPU ของ FAISS ให้ใช้หรือไม่ (Windows ไม่มี)
    if device == "cuda" and hasattr(faiss, "StandardGpuResources"):
        try:
            print("⚡ ใช้งาน FAISS GPU acceleration")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"⚠️ ไม่สามารถใช้ GPU ได้: {e} → ใช้ CPU แทน")
    else:
        print("🧠 ใช้งาน FAISS CPU")

    index.add(embeddings)
    return index


def search_similar(query: str, texts: List[str], index, top_k=5, threshold=0.5):
    """ค้นหาข้อความที่ใกล้เคียงที่สุด"""
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True, device=device)
    scores, indices = index.search(query_emb, top_k)
    results = []
    for idx, i in enumerate(indices[0]):
        score = scores[0][idx]
        if score >= threshold:
            results.append((texts[i], float(score)))
    return results if results else [(texts[i], float(scores[0][idx])) for idx, i in enumerate(indices[0])]


# ===== BUILD KNOWLEDGE =====
def build_law_dataset():
    """โหลดและสร้าง embedding สำหรับกฎหมาย"""
    law_data = load_json_files(FOLDER_LAW)
    law_texts = []
    for d in law_data:
        if isinstance(d, dict) and "section_content" in d and "law_name" in d:
            law_texts.append(f"{d['law_name']} มาตรา {d['section_num']}:\n{d['section_content']}")
    if not law_texts:
        raise ValueError("❌ ไม่มีข้อมูลกฎหมายในไฟล์ (law_texts ว่าง)")
    print(f"✅ พบ {len(law_texts)} มาตราในการสร้าง embedding")
    emb = embed_texts(law_texts)
    index = build_index(emb)
    return law_texts, index


def build_lawyer_dataset():
    """โหลดและสร้าง embedding สำหรับข้อมูลทนาย"""
    lawyer_data = load_json_files(FOLDER_LAWYER)
    lawyer_texts = []
    for l in lawyer_data:
        if "name" in l and "expertise" in l:
            lawyer_texts.append(f"{l['name']} - ความเชี่ยวชาญ: {', '.join(l['expertise'])}")
    if not lawyer_texts:
        raise ValueError("❌ ไม่มีข้อมูลทนายในไฟล์ (lawyer_texts ว่าง)")
    print(f"✅ พบ {len(lawyer_texts)} ทนายในการสร้าง embedding")
    emb = embed_texts(lawyer_texts)
    index = build_index(emb)
    return lawyer_data, lawyer_texts, index


# ===== AI UTILITIES =====
def generate_answer(context: str, lawyers: List[dict], query: str):
    """เรียก Ollama เพื่อสร้างคำตอบ"""
    lawyer_suggestions = "\n".join([
        f"- {l['name']} ({', '.join(l['expertise'])})" for l in lawyers
    ])
    prompt = f"""
คุณคือผู้เชี่ยวชาญด้านกฎหมายที่สามารถอธิบายข้อกฎหมายให้เข้าใจง่าย

คำถาม: {query}

บริบทกฎหมายที่เกี่ยวข้อง:
{context}

ทนายที่เชี่ยวชาญในเรื่องนี้:
{lawyer_suggestions}

กรุณาตอบโดย:
1. อธิบายให้เข้าใจง่าย
2. อ้างอิงมาตราที่เกี่ยวข้อง
3. ยกตัวอย่างประกอบ (ถ้ามี)
4. ปิดท้ายด้วยข้อความ “คุณต้องการติดต่อทนายไหม”
"""
    res = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return res.json().get("response", "").strip()


def is_positive_reply(message: str) -> bool:
    """วิเคราะห์ว่าผู้ใช้ยินยอมติดต่อทนายหรือไม่"""
    text = message.strip().lower()

    positive_keywords = ["ตกลง", "โอเค", "ได้", "ครับ", "ค่ะ", "เห็นด้วย", "ยินยอม", "yes", "ok", "agree"]
    negative_keywords = ["ไม่", "no", "ยัง", "ปฏิเสธ", "ไม่ตกลง", "ไม่ครับ", "ไม่ค่ะ"]

    # ถ้ามีคำว่า "ไม่" อยู่ก่อนคำยินยอม เช่น "ไม่ตกลง" หรือ "ยังไม่"
    if any(neg in text for neg in negative_keywords):
        return False

    if any(pos in text for pos in positive_keywords):
        return True

    # ✅ 2. ถ้าไม่ชัดเจน → ใช้ Ollama วิเคราะห์ต่อ
    prompt = f"""
จงวิเคราะห์ว่า ข้อความต่อไปนี้สื่อถึงการ 'ตกลง' หรือ 'ยินยอม' หรือไม่

ข้อความ: "{message}"

ให้ตอบเพียงคำเดียว:
- ถ้าใช่ ตอบว่า TRUE
- ถ้าไม่ใช่ ตอบว่า FALSE
"""
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        reply = res.json().get("response", "").strip().lower()
        print(f"🤖 วิเคราะห์โดย AI: {reply}")
        return "true" in reply
    except Exception as e:
        print("❌ ตรวจวิเคราะห์ไม่ได้:", e)
        return False


# ===== ROUTES =====
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    global lawyer_data
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "❌ โปรดระบุคำถาม"}), 400

    # 1️⃣ หา context กฎหมาย
    contexts = search_similar(question, law_texts, law_index)
    context_text = "\n\n".join([t for t, _ in contexts])

    # 2️⃣ หา lawyer ที่เกี่ยวข้อง
    lawyer_matches = search_similar(question, [t for t in lawyer_texts], lawyer_index, top_k=3)
    related_lawyers = []
    for text, _ in lawyer_matches:
        for l in lawyer_data:
            if l["name"] in text:
                related_lawyers.append(l)

    # 3️⃣ จำทนายล่าสุดไว้
    if related_lawyers:
        app.config["last_lawyer"] = related_lawyers[0]

    # 4️⃣ ให้ AI ตอบ
    answer = generate_answer(context_text, related_lawyers, question)

    return jsonify({
        "answer": answer,
        "context": context_text,
        "lawyers": related_lawyers
    })


@app.route("/confirm", methods=["POST"])
def confirm_contact():
    """Route สำหรับยืนยันการติดต่อทนาย (AI วิเคราะห์เอง)"""
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "❌ โปรดพิมพ์ข้อความ"}), 400
    if is_positive_reply(message):
        if "last_lawyer" not in app.config:
            return jsonify({"error": "❌ ไม่มีข้อมูลทนายล่าสุด"}), 400
        return jsonify({
            "message": "📞 ข้อมูลติดต่อทนาย",
            "lawyer": app.config["last_lawyer"]
        })
    else:
        return jsonify({"message": "โอเค หวังว่าคำตอบจะช่วยให้การตัดสินใจได้ง่ายมากขึ้น"}), 200


# ===== MAIN =====
if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    # โหลดหรือสร้างข้อมูลกฎหมาย
    if os.path.exists(TEXTS_LAW) and os.path.exists(INDEX_LAW):
        print("📂 โหลดกฎหมายจากแคช")
        with open(TEXTS_LAW, "r", encoding="utf-8") as f:
            law_texts = json.load(f)
        law_index = faiss.read_index(INDEX_LAW)
    else:
        law_texts, law_index = build_law_dataset()
        faiss.write_index(law_index, INDEX_LAW)
        with open(TEXTS_LAW, "w", encoding="utf-8") as f:
            json.dump(law_texts, f, ensure_ascii=False)

    # โหลดหรือสร้างข้อมูลทนาย
    if os.path.exists(TEXTS_LAWYER) and os.path.exists(INDEX_LAWYER):
        print("📂 โหลดทนายจากแคช")
        with open(TEXTS_LAWYER, "r", encoding="utf-8") as f:
            lawyer_data = json.load(f)
        lawyer_texts = [f"{d['name']} - ความเชี่ยวชาญ: {', '.join(d['expertise'])}" for d in lawyer_data]
        lawyer_index = faiss.read_index(INDEX_LAWYER)
    else:
        lawyer_data, lawyer_texts, lawyer_index = build_lawyer_dataset()
        faiss.write_index(lawyer_index, INDEX_LAWYER)
        with open(TEXTS_LAWYER, "w", encoding="utf-8") as f:
            json.dump(lawyer_data, f, ensure_ascii=False)

    print("\n🎯 ระบบ RAG กฎหมาย + ทนาย (AI ยืนยันการติดต่อ) พร้อมใช้งาน!")
    app.run(host="0.0.0.0", port=8000, debug=True)
