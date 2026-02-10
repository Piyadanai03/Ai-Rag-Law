# ระบบถาม-ตอบกฎหมายไทย (Thai Law QA System)

ระบบตอบคำถามเกี่ยวกับกฎหมายไทยโดยใช้ RAG (Retrieval-Augmented Generation) และ Ollama

## การเปิดใช้งานใน LocalHost

1. ติดตั้ง Python 3.8 หรือใหม่กว่า
2. ติดตั้ง Ollama จาก https://ollama.ai
3. ติดตั้ง Ollama Model:
```bash
   ollama pull llama3.2
```

4. สร้าง Virtual Environment (ถ้ายังไม่สร้าง)
```bash
   python -m venv venv
```

5. เข้าใช้งาน venv
```bash
   venv\Scripts\activate
```   

6. ติดตั้ง Dependencies:
```bash
   pip install -r requirements.txt
```

## การรันระบบ

1. รันแอปพลิเคชัน:
```bash
python app.py
```
2. เปิดเว็บบราวเซอร์ไปที่ http://127.0.0.1:8000


## การเปิดใช้งานใน DOCKER ใช้ GPU ใช้การ์ดจอ Nvidia

Ubuntu 20.04/22.04  

1. ตรวจ GPU บนเครื่อง
```bash
   nvidia-smi
```  

2. ตรวจ Docker เห็น NVIDIA runtime
```bash
docker info | grep -i nvidia 
```

# ( 3-7 ติดตั้ง NVIDIA Container Toolkit)

3. 
```bash 
sudo apt update 
```

4. 
```bash
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

5. 
```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

6. 
```bash
sudo apt update
```

7. 
```bash
sudo apt install -y nvidia-container-toolkit
```

8. ทำครั้งเดียวพอ bind NVIDIA เข้ากับ Docker
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

9. restart docker
```bash
sudo service docker restart 
```

10. 
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## การเตรียมข้อมูล

1. สร้างโฟลเดอร์ `data` ในโปรเจค
2. นำไฟล์ JSON ที่มีข้อมูลกฎหมายไปไว้ในโฟลเดอร์ `data`
   - รูปแบบไฟล์ JSON ต้องประกอบด้วย: law_name, section_num, section_content


## วิธีใช้งาน

1. พิมพ์คำถามเกี่ยวกับกฎหมายในช่องคำถาม
2. กดปุ่ม "ส่งคำถาม" หรือกด Enter
3. รอระบบประมวลผลและแสดงคำตอบ
4. สามารถเลือกคำถามตัวอย่างได้จากปุ่มด้านล่าง

## กระบวนการทำงาน

1. **การเตรียมข้อมูล**
   - โหลดข้อมูลกฎหมายจากไฟล์ JSON
   - สร้าง Embeddings ด้วย SentenceTransformer
   - สร้าง FAISS Index สำหรับการค้นหา

2. **การประมวลผลคำถาม**
   - รับคำถามจากผู้ใช้
   - ค้นหาบริบทกฎหมายที่เกี่ยวข้องด้วย FAISS
   - ส่งคำถามและบริบทไปยัง Ollama
   - แสดงคำตอบที่ได้แก่ผู้ใช้

3. **การแคช**
   - ระบบจะสร้างไฟล์แคช (faiss.index และ texts.json)
   - ใช้ไฟล์แคชเพื่อลดเวลาในการโหลดครั้งต่อไป


# ==== PyTorch + CUDA ====
torch==2.8.0+cu126
torchvision==0.23.0+cu126
torchaudio==2.8.0+cu126