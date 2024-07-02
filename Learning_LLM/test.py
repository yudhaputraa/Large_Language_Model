from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Large Language Model (LLM) Explanation', 0, 1, 'C')

    def chapter_title(self, chapter_title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, chapter_title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, chapter_title, body):
        self.add_page()
        self.chapter_title(chapter_title)
        self.chapter_body(body)

pdf = PDF()

# Adding title and sections to the PDF
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Large Language Model (LLM) Explanation', 0, 1, 'C')

sections = [
    ("Data Preprocessing (Pra-pemrosesan Data)", 
    "Tokenization (Tokenisasi)\nMemecah teks menjadi token-token (kata atau sub-kata).\n"
    "tokens = tokenize(text)\n\n"
    "Embedding\nMengonversi token-token ke dalam representasi vektor berdimensi tinggi.\n"
    "embeddings = EmbeddingLayer(tokens)"),

    ("Model Architecture (Arsitektur Model)",
    "Positional Encoding\nMenambahkan informasi posisi ke embedding input.\n"
    "PE(pos, 2i) = sin(pos / 10000^(2i/d_model))\n"
    "PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\n\n"
    "Transformer Encoder Layer\nMengandung dua sub-lapisan: Self-Attention dan Feed-Forward Neural Network.\n\n"
    "Self-Attention\nMenghitung perhatian diri untuk setiap posisi dalam input sequence.\n"
    "Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V\n"
    "di mana Q adalah query, K adalah key, V adalah value, dan d_k adalah dimensi key.\n\n"
    "Feed-Forward Neural Network\nLapisan jaringan saraf yang diterapkan setelah mekanisme perhatian.\n"
    "FFN(x) = max(0, xW_1 + b_1)W_2 + b_2\n\n"
    "Layer Normalization\nNormalisasi di setiap sub-lapisan.\n"
    "LayerNorm(x) = (x - mu) / sqrt(sigma^2 + epsilon)"),

    ("Training (Pelatihan)",
    "Loss Function (Fungsi Kehilangan)\nMengukur seberapa baik model membuat prediksi.\n"
    "Cross-Entropy Loss (Untuk klasifikasi)\n"
    "L(y, hat{y}) = -sum(y_i log(hat{y}_i))\n"
    "di mana y adalah label sebenarnya dan hat{y} adalah prediksi model.\n\n"
    "Optimization (Optimasi)\nMemperbarui bobot model untuk meminimalkan fungsi kehilangan.\n"
    "Gradient Descent\n"
    "theta := theta - eta nabla_theta J(theta)\n"
    "di mana theta adalah parameter model, eta adalah learning rate, dan J(theta) adalah fungsi kehilangan.\n\n"
    "Adam (Adaptive Moment Estimation)\n"
    "m_t = beta_1 m_{t-1} + (1 - beta_1) g_t\n"
    "v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2\n"
    "hat{m}_t = m_t / (1 - beta_1^t)\n"
    "hat{v}_t = v_t / (1 - beta_2^t)\n"
    "theta_t := theta_{t-1} - eta hat{m}_t / (sqrt(hat{v}_t) + epsilon)"),

    ("Evaluation (Evaluasi)",
    "Metrics (Metrik)\nMengukur kinerja model.\n\n"
    "Accuracy (Akurasi)\nPersentase prediksi benar dari total prediksi.\n"
    "Accuracy = Correct Predictions / Total Predictions\n\n"
    "Precision, Recall, F1-Score\nDigunakan untuk evaluasi tugas klasifikasi.\n"
    "Precision = True Positives / (True Positives + False Positives)\n"
    "Recall = True Positives / (True Positives + False Negatives)\n"
    "F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"),

    ("Alur Keseluruhan",
    "1. Data Preprocessing: Tokenisasi -> Embedding\n"
    "2. Model Architecture: Positional Encoding -> Transformer Encoder Layer (Self-Attention -> Feed-Forward Neural Network -> Layer Normalization)\n"
    "3. Training: Hitung Loss (Cross-Entropy) -> Optimasi (Gradient Descent/Adam)\n"
    "4. Evaluation: Hitung Metrics (Accuracy, Precision, Recall, F1-Score)")
]

for title, body in sections:
    pdf.add_chapter(title, body)

# Save the PDF to a file
output_path = "LLM_Explanation.pdf"
pdf.output(output_path)

output_path
