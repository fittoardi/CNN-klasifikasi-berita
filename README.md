# Klasifikasi Berita CNN: Deteksi Berita Hoax vs Berita Nyata
Proyek ini mengimplementasikan Convolutional Neural Network (CNN) untuk klasifikasi berita hoaks dan asli dalam bahasa Indonesia. Model ini dirancang untuk membantu dalam deteksi misinformasi dan fake news yang beredar di media sosial dan platform berita online.
### 1. **Metode CNN (Convolutional Neural Network)**
- Menggunakan arsitektur CNN 1D untuk pemrosesan teks
- Layer Embedding untuk representasi kata
- Multiple Conv1D layers dengan MaxPooling
- GlobalMaxPooling dan Dense layers untuk klasifikasi

### 2. **Program Python Lengkap**
- **Preprocessing**: Lowercase, remove special characters, tokenization, stopwords removal
- **Data Generation**: Membuat 30 sampel data (15 hoax, 15 real)
- **dataset berita**:**hoaks**= "Vaksin COVID-19 mengandung microchip untuk mengontrol manusia",
                                "Pemerintah menyembunyikan fakta bahwa bumi ini datar",
                                "Minum air putih hangat bisa menyembuhkan kanker dalam seminggu",
                                "Gelombang 5G menyebabkan virus corona menyebar lebih cepat",
                                "Obat tradisional X bisa menyembuhkan diabetes tanpa efek samping",
                                "Alien sudah mendarat di bumi dan bekerja sama dengan pemerintah",
                                "Makan nanas setiap hari bisa membuat awet muda 20 tahun",
                                "Pemerintah meracuni air minum untuk mengurangi populasi",
                                "Teknologi HAARP digunakan untuk mengontrol cuaca dan gempa",
                                "Vitamin C dosis tinggi bisa mencegah semua jenis penyakit",
                                "Ilmuwan menemukan cara hidup abadi dengan teknologi rahasia",
                                "Ponsel pintar bisa membaca pikiran pengguna melalui radiasi",
                                "Makanan organik 100% bisa menyembuhkan autisme pada anak",
                                "Chemtrail dari pesawat adalah upaya pemerintah meracuni udara",
                                "Manusia bisa hidup tanpa makan selama berbulan-bulan dengan meditasi"
                    **real** =  "Menteri Kesehatan mengumumkan program vaksinasi nasional tahap kedua",
                                "Badan Meteorologi melaporkan cuaca ekstrem di berbagai daerah",
                                "Penelitian menunjukkan efektivitas masker dalam mencegah penularan virus",
                                "Pemerintah alokasikan dana bantuan untuk UMKM terdampak pandemi",
                                "Teknologi AI terbaru membantu diagnosis penyakit lebih akurat",
                                "Universitas terkemuka meluncurkan program beasiswa untuk mahasiswa berprestasi",
                                "Perusahaan teknologi investasi besar untuk pengembangan energi terbarukan",
                                "Dokter menekankan pentingnya pola hidup sehat dan olahraga teratur",
                                "Startup lokal berhasil mengembangkan aplikasi edukasi inovatif",
                                "Bank sentral mempertahankan suku bunga untuk stabilitas ekonomi",
                                "Ilmuwan berhasil mengembangkan terapi gen untuk penyakit langka",
                                "Kementerian Pendidikan luncurkan kurikulum baru berbasis teknologi",
                                "Perusahaan farmasi uji klinis obat baru untuk pengobatan kanker",
                                "Pemerintah bangun infrastruktur digital untuk daerah terpencil",
                                "Peneliti temukan spesies baru di kedalaman laut Indonesia"
- **Model Architecture**: CNN dengan Embedding → Conv1D → MaxPooling → Dense → Output
- **Training**: Dengan early stopping dan learning rate reduction
- **Evaluation**: Accuracy, classification report, confusion matrix

COODE ATAU PROGRAM PYTHON
```# CNN untuk Klasifikasi Berita Hoaks atau Asli
# Implementasi lengkap dengan preprocessing, training, validasi, dan testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Download punkt_tab if not found (required by word_tokenize)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


print("="*60)
print("IMPLEMENTASI CNN UNTUK KLASIFIKASI BERITA HOAKS/ASLI")
print("="*60)

# 1. GENERATE DATA DUMMY (karena data minimal 20 sampel)
def generate_dummy_data():
    """Generate data dummy untuk klasifikasi berita"""

    # Contoh berita hoaks
    hoax_news = [
        "Vaksin COVID-19 mengandung microchip untuk mengontrol manusia",
        "Pemerintah menyembunyikan fakta bahwa bumi ini datar",
        "Minum air putih hangat bisa menyembuhkan kanker dalam seminggu",
        "Gelombang 5G menyebabkan virus corona menyebar lebih cepat",
        "Obat tradisional X bisa menyembuhkan diabetes tanpa efek samping",
        "Alien sudah mendarat di bumi dan bekerja sama dengan pemerintah",
        "Makan nanas setiap hari bisa membuat awet muda 20 tahun",
        "Pemerintah meracuni air minum untuk mengurangi populasi",
        "Teknologi HAARP digunakan untuk mengontrol cuaca dan gempa",
        "Vitamin C dosis tinggi bisa mencegah semua jenis penyakit",
        "Ilmuwan menemukan cara hidup abadi dengan teknologi rahasia",
        "Ponsel pintar bisa membaca pikiran pengguna melalui radiasi",
        "Makanan organik 100% bisa menyembuhkan autisme pada anak",
        "Chemtrail dari pesawat adalah upaya pemerintah meracuni udara",
        "Manusia bisa hidup tanpa makan selama berbulan-bulan dengan meditasi"
    ]

    # Contoh berita asli
    real_news = [
        "Menteri Kesehatan mengumumkan program vaksinasi nasional tahap kedua",
        "Badan Meteorologi melaporkan cuaca ekstrem di berbagai daerah",
        "Penelitian menunjukkan efektivitas masker dalam mencegah penularan virus",
        "Pemerintah alokasikan dana bantuan untuk UMKM terdampak pandemi",
        "Teknologi AI terbaru membantu diagnosis penyakit lebih akurat",
        "Universitas terkemuka meluncurkan program beasiswa untuk mahasiswa berprestasi",
        "Perusahaan teknologi investasi besar untuk pengembangan energi terbarukan",
        "Dokter menekankan pentingnya pola hidup sehat dan olahraga teratur",
        "Startup lokal berhasil mengembangkan aplikasi edukasi inovatif",
        "Bank sentral mempertahankan suku bunga untuk stabilitas ekonomi",
        "Ilmuwan berhasil mengembangkan terapi gen untuk penyakit langka",
        "Kementerian Pendidikan luncurkan kurikulum baru berbasis teknologi",
        "Perusahaan farmasi uji klinis obat baru untuk pengobatan kanker",
        "Pemerintah bangun infrastruktur digital untuk daerah terpencil",
        "Peneliti temukan spesies baru di kedalaman laut Indonesia"
    ]

    # Buat DataFrame
    news_data = hoax_news + real_news
    labels = ['hoax'] * len(hoax_news) + ['real'] * len(real_news)

    df = pd.DataFrame({
        'text': news_data,
        'label': labels
    })

    return df

# 2. PREPROCESSING TEXT
def preprocess_text(text):
    """Preprocessing text untuk CNN"""
    # Lowercase
    text = text.lower()

    # Hapus karakter khusus
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisasi
    tokens = word_tokenize(text)

    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

# 3. LOAD DAN PREPROCESS DATA
print("1. Loading dan Preprocessing Data...")
df = generate_dummy_data()
print(f"Total data: {len(df)}")
print(f"Distribusi label:\n{df['label'].value_counts()}")

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

print("\nContoh data setelah preprocessing:")
print(df[['text', 'processed_text', 'label']].head(3))

# 4. TOKENIZATION DAN PADDING
print("\n2. Tokenization dan Padding...")
MAX_WORDS = 1000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['processed_text'])

sequences = tokenizer.texts_to_sequences(df['processed_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label_encoded'].values

print(f"Shape data X: {X.shape}")
print(f"Shape data y: {y.shape}")

# 5. SPLIT DATA
print("\n3. Splitting Data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} sampel")
print(f"Validation set: {X_val.shape[0]} sampel")
print(f"Test set: {X_test.shape[0]} sampel")

# 6. MEMBANGUN MODEL CNN
print("\n4. Membangun Model CNN...")
def create_cnn_model():
    """Membuat model CNN untuk klasifikasi teks"""
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),

        # Convolutional layers
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=16, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),

        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    return model

model = create_cnn_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Arsitektur Model CNN:")
model.summary()

# 7. TRAINING MODEL
print("\n5. Training Model...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
]

history = model.fit(
    X_train, y_train,
    batch_size=8,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# 8. EVALUASI MODEL
print("\n6. Evaluasi Model...")

# Prediksi
y_pred_train = (model.predict(X_train) > 0.5).astype(int)
y_pred_val = (model.predict(X_val) > 0.5).astype(int)
y_pred_test = (model.predict(X_test) > 0.5).astype(int)

# Akurasi
train_accuracy = accuracy_score(y_train, y_pred_train)
val_accuracy = accuracy_score(y_val, y_pred_val)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Hoax', 'Real']))

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# 9. VISUALISASI HASIL
print("\n7. Visualisasi Hasil...")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hoax', 'Real'],
            yticklabels=['Hoax', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 10. FUNGSI PREDIKSI UNTUK TEKS BARU
def predict_news(text, model, tokenizer, le):
    """Prediksi apakah berita hoax atau asli"""
    # Preprocess
    processed = preprocess_text(text)

    # Tokenize dan padding
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Prediksi
    prediction = model.predict(padded)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    predicted_label = le.inverse_transform([predicted_class])[0]

    return predicted_label, prediction

# 11. CONTOH PENGGUNAAN
print("\n8. Contoh Prediksi untuk Berita Baru:")
test_texts = [
    "Pemerintah mengumumkan program bantuan sosial untuk rakyat miskin",
    "Minum air kelapa bisa menyembuhkan semua jenis kanker dalam 3 hari",
    "Universitas terkemuka meluncurkan program penelitian teknologi AI"
]

for i, text in enumerate(test_texts, 1):
    predicted_label, confidence = predict_news(text, model, tokenizer, le)
    print(f"\nTeks {i}: {text}")
    print(f"Prediksi: {predicted_label.upper()}")
    print(f"Confidence: {confidence:.4f}")

# 12. KESIMPULAN
print("\n" + "="*60)
print("KESIMPULAN")
print("="*60)
print(f"✓ Model CNN berhasil dibangun dengan akurasi test: {test_accuracy:.4f}")
print(f"✓ Data minimal 20 sampel telah dipenuhi ({len(df)} sampel)")
print(f"✓ Pembagian data: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} testing")
print(f"✓ Model menggunakan arsitektur CNN dengan Embedding layer")
print(f"✓ Preprocessing meliputi: lowercase, remove special chars, tokenization, stopwords removal")
print(f"✓ Model dapat memprediksi berita hoax/asli dengan tingkat akurasi yang baik")
print("="*60)

# 13. SAVE MODEL (OPTIONAL)
print("\n9. Menyimpan Model...")
model.save('cnn_news_classifier.h5')
print("✓ Model tersimpan sebagai 'cnn_news_classifier.h5'")

# Save tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✓ Tokenizer dan Label Encoder tersimpan")
print("\nProgram selesai! Model siap digunakan untuk klasifikasi berita.")
```

### 3. **Data 30 Sampel**
- Program menggunakan 30 sampel berita (15 hoax, 15 real)
- Data dibagi menjadi: Training (60%), Validation (20%), Testing (20%)

