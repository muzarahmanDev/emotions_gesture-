# emotions_gesture

Aplikasi desktop berbasis Python untuk mendeteksi ekspresi wajah dan gestur tangan secara real-time melalui webcam, lalu menampilkan overlay stiker lucu di atas video (dengan dukungan output ke Virtual Camera).

## Fitur Utama (Rencana Pengembangan)
- Preview webcam real-time dengan OpenCV
- Deteksi wajah dan gestur tangan secara lokal (tanpa upload ke cloud)
- Overlay stiker transparan dengan cooldown anti-spam
- Toggle enable/disable
- Output ke Virtual Camera untuk Zoom / Google Meet / WhatsApp Desktop

## Privasi
Seluruh pemrosesan video dilakukan **100% secara lokal** di perangkat pengguna. Aplikasi ini tidak merekam, menyimpan, atau mengirim frame video ke server eksternal mana pun.

## Setup Awal
```bash
# 1. Buat virtual environment (Windows)
py -3.11 -m venv .venv

# 2. Aktifkan virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt 