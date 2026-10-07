# Model Gradient Boosting untuk Mengidentifikasi Kesenjangan Upah Berdasarkan Gender

Model klasifikasi biner yang memprediksi apakah seorang tenaga kerja berpenghasilan **upah tinggi** (`1`) atau **upah rendah** (`0`), memakai algoritma **Gradient Boosting**. Proyek ini mendukung **SDG 5 (Kesetaraan Gender)**.

- **Penulis:** Dino Fahri, Teknik Informatika, Universitas Halu Oleo
- **Algoritma:** `GradientBoostingClassifier` (scikit-learn), satu-satunya algoritma yang dipakai
- **File model:** `model_upah_gb.joblib`

## Data

Dataset: [Gender Pay Gap Dataset (fedesoriano, Kaggle)](https://www.kaggle.com/datasets/fedesoriano/gender-pay-gap-dataset), gabungan dua survei Amerika Serikat.

| Sumber | Baris |
|---|---|
| CPS (Current Population Survey) | 344.287 |
| PSID (Panel Study of Income Dynamics) | 33.398 |
| **Gabungan** | **377.685** (24 kolom) |

**Target `upah_tinggi`:** bernilai 1 bila upah per jam riil lebih dari **US$32,79** (dolar 2010). Ambang ini adalah ambang Kohavi (1996), yaitu penghasilan lebih dari US$50 ribu per tahun, dikonversi ke upah per jam. Kelas upah tinggi hanya sekitar 15% data.

## Metodologi singkat

1. Pembersihan data dan penyeragaman kolom CPS dan PSID.
2. Encoding `metro`, `industri`, dan `pekerjaan` dengan `get_dummies` (59 kolom), lalu fitur lemah dibuang menjadi **55 fitur**. Kolom `female` dan `sumber_psid` dipertahankan.
3. Pembagian data latih dan uji 80:20 per orang (`GroupShuffleSplit` pada `id_orang`), sehingga satu orang tidak muncul di kedua bagian. Latih 302.122 baris, uji 75.563 baris.
4. Ketidakseimbangan kelas ditangani dengan `compute_sample_weight('balanced')`.
5. Nilai kosong diisi median data latih.
6. Pelatihan: `n_estimators=200`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.8`, `random_state=42`.

Kolom upah (`realhrwage`) tidak dipakai sebagai fitur untuk mencegah kebocoran data (*data leakage*).

## Hasil pada data uji

| Metrik | Nilai |
|---|---|
| Akurasi | 0,7878 |
| Precision (upah tinggi) | 0,3976 |
| Recall (upah tinggi) | 0,7888 |
| F1-score | 0,5287 |
| ROC-AUC | 0,8691 |
| Balanced accuracy | 0,7882 |

Akurasi latih 0,7899 dan akurasi uji 0,7878, jadi tidak ada tanda overfitting. Precision kelas upah tinggi rendah (0,40) karena model sengaja diseimbangkan demi recall.

## Isi file model

`model_upah_gb.joblib` adalah sebuah dictionary dengan empat kunci:

| Kunci | Isi |
|---|---|
| `model` | objek `GradientBoostingClassifier` yang sudah dilatih |
| `kolom_fitur` | daftar 55 nama kolom, dengan urutan yang harus diikuti |
| `median_train` | median tiap fitur dari data latih, untuk mengisi nilai kosong |
| `ambang_per_jam` | ambang upah tinggi, 32,79 (dolar 2010 per jam) |

## Cara menggunakan model

### 1. Instal dan unduh model

```bash
pip install scikit-learn pandas joblib
```

```python
import joblib, urllib.request

URL = "https://github.com/difar-nim/model_upah_gb/raw/main/model_upah_gb.joblib"
urllib.request.urlretrieve(URL, "model_upah_gb.joblib")

paket = joblib.load("model_upah_gb.joblib")
model        = paket["model"]
kolom_fitur  = paket["kolom_fitur"]
median_train = paket["median_train"]
```

> Gunakan versi scikit-learn yang sama dengan saat pelatihan agar file dapat dimuat tanpa peringatan.

### 2. Siapkan data masukan

Data masukan harus memiliki **kolom yang sama persis dan urutan yang sama** dengan `kolom_fitur`. Kolom dummy (`metro_*`, `industri_*`, `pekerjaan_*`) diisi 1 untuk kategori yang sesuai dan 0 untuk yang lain.

```python
import pandas as pd

def prediksi(age, female, sch, jam_mingguan, ft, region, metro=True,
             ba=0, adv=0, year=2011, ras="white", industri=None, pekerjaan=None):
    """female: 1 = perempuan, 0 = laki-laki. ft: 1 = full-time, 0 = part-time.
    region: northeast | northcentral | south | west."""
    x = pd.DataFrame(0, index=[0], columns=kolom_fitur, dtype=float)
    x["year"], x["age"], x["female"] = year, age, female
    x["sch"], x["ba"], x["adv"] = sch, ba, adv
    x["potexp"] = max(0, age - sch - 6)
    x["jam_mingguan"], x["ft"] = jam_mingguan, ft
    x[ras] = 1                      # white | black | hisp | othrace
    x[region] = 1
    x["metro_1" if metro else "metro_0"] = 1
    if industri is not None and f"industri_{industri}" in x.columns:
        x[f"industri_{industri}"] = 1
    if pekerjaan is not None and f"pekerjaan_{pekerjaan}" in x.columns:
        x[f"pekerjaan_{pekerjaan}"] = 1

    p = model.predict_proba(x[kolom_fitur])[0, 1]
    return p, int(p >= 0.5)
```

### 3. Contoh pemakaian

```python
p, kelas = prediksi(age=35, female=1, sch=16, jam_mingguan=40, ft=1,
                    region="northeast", ba=1)
print(f"Probabilitas upah tinggi: {p*100:.2f}%")
print("Klasifikasi:", "UPAH TINGGI" if kelas == 1 else "UPAH RENDAH")

# Bandingkan dengan profil yang sama tetapi laki-laki
p_l, _ = prediksi(age=35, female=0, sch=16, jam_mingguan=40, ft=1,
                  region="northeast", ba=1)
print(f"Selisih probabilitas (laki-laki - perempuan): {(p_l - p)*100:.2f} poin")
```

### 4. Prediksi untuk banyak data sekaligus

Bila data sudah berbentuk tabel dengan 55 kolom yang sama (misalnya `cps_bersih.csv` setelah di-encode):

```python
X = data[kolom_fitur].fillna(median_train)
proba = model.predict_proba(X)[:, 1]
kelas = (proba >= 0.5).astype(int)
```

### Keterangan fitur

| Kelompok | Kolom |
|---|---|
| Demografi | `age`, `female`, `white`, `black`, `hisp`, `othrace` |
| Pendidikan dan pengalaman | `sch` (tahun sekolah), `ba` (sarjana), `adv` (pascasarjana), `potexp` (potensi pengalaman) |
| Pekerjaan | `jam_mingguan`, `ft` (full-time), `industri_*`, `pekerjaan_*` |
| Wilayah | `northeast`, `northcentral`, `south`, `west`, `metro_0`, `metro_1`, `metro_nan` |
| Waktu dan sumber | `year`, `sumber_psid` (0 = CPS, 1 = PSID) |

`potexp` dihitung sebagai `usia − tahun sekolah − 6`. Untuk `sumber_psid` gunakan 0 bila tidak ada alasan memakai 1.

## Keterbatasan

- Akurasi uji **78,78%**, belum mencapai target 90%. Kelas upah tinggi hanya 15% data sehingga precision-nya rendah (0,40).
- Bila `industri` dan `pekerjaan` tidak diisi, semua kolom dummy-nya bernilai 0, padahal `pekerjaan` adalah fitur dengan hubungan terkuat terhadap target. Prediksi menjadi kurang akurat.
- Data berasal dari Amerika Serikat (1981–2013) dengan upah dalam dolar 2010, sehingga model tidak mewakili pasar kerja Indonesia.
- Hasil menunjukkan **asosiasi**, bukan bukti sebab-akibat diskriminasi upah.

## Referensi

- Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.* KDD-96. (asal ambang US$50 ribu per tahun)
- Dataset: fedesoriano, *Gender Pay Gap Dataset*, Kaggle.
