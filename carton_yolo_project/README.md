# Carton YOLO Project (Emrah / `carton_pack`)

Bu proje, senin **sadece carton görevin** için hazırlanmış temiz, modüler ve genişletilebilir bir YOLO bilgisayarlı görü pipeline'ıdır.

Bu yapı özellikle şu hedeflere göre tasarlandı:

- **CVAT export hazır kabul edilir**
- **tek sınıf başlangıç modeli**: `carton_pack`
- **temiz kod / modüler yapı**
- **logging + error handling**
- **Ubuntu terminalden kolay kullanım**
- **ileride Raspberry Pi 5 / AI camera tarafına taşınabilir yapı**
- **image / video / camera için ayrı komut akışı**
- **eğitim, validasyon, export ve tahmin modülleri ayrı**

## Neden bu yapı?

Bu scaffold, sizin proje belgelerinizde geçen şu temel ilkelere göre kuruldu:

- Emrah için odak sınıf **`carton_pack`**
- gerçek demo koşullarına benzeyen webcam görüntüleri önemli
- klasör yapısı tutarlı olmalı
- CVAT'tan YOLO uyumlu export alınmalı
- erken test ve kontrollü genişleme yaklaşımı kullanılmalı

Bu noktalar yüklediğin proje metinlerinde açıkça geçiyor. fileciteturn0file0L21-L24 fileciteturn0file1L41-L47 fileciteturn0file2L67-L76

## Proje yapısı

```text
carton_yolo_project/
├── README.md
├── requirements.txt
├── .gitignore
├── bin/
│   ├── prepare_dataset
│   ├── train
│   ├── validate
│   ├── image
│   ├── video
│   ├── camera
│   └── export_model
├── configs/
│   ├── default.yaml
│   └── logging.yaml
├── data/
│   ├── external/
│   │   └── cvat_export/
│   │       ├── README.md
│   │       ├── images/
│   │       └── labels/
│   ├── dataset/
│   │   ├── README.md
│   │   ├── data_carton.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── demo/
│   │   ├── README.md
│   │   ├── unseen_images/
│   │   └── unseen_labels/
│   └── raw/
│       └── README.md
├── docs/
│   └── RESOURCES.md
├── logs/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── runs/
│   └── .gitkeep
└── src/
    └── carton_yolo/
        ├── __init__.py
        ├── main.py
        ├── config.py
        ├── constants.py
        ├── dataset.py
        ├── exceptions.py
        ├── model_export.py
        ├── predictor.py
        ├── trainer.py
        ├── validator.py
        └── utils/
            ├── __init__.py
            ├── io_utils.py
            ├── logging_utils.py
            └── paths.py
```

## Kurulum

### 1) Venv aktif et
```bash
source .venv/bin/activate
```

### 2) Paketleri kur
```bash
pip install -r requirements.txt
```

### 3) Çalıştırılabilir komutları aktif et
Proje klasöründe:

```bash
chmod +x bin/*
export PATH="$PWD/bin:$PATH"
```

İstersen bunu kalıcı yapmak için `~/.bashrc` içine şunu ekleyebilirsin:

```bash
export PATH="/tam/proje/yolu/carton_yolo_project/bin:$PATH"
```

Sonra:
```bash
source ~/.bashrc
```

## CVAT'tan gelen dosyalar nereye konulacak?

Şu klasöre:

```text
data/external/cvat_export/images/
data/external/cvat_export/labels/
```

Bu klasör **özellikle CVAT'tan export edilmiş hazır dosyalar** içindir.

Sizin belgelerde CVAT → YOLO export → dataset düzenleme akışı zaten öneriliyor. fileciteturn0file2L114-L123 fileciteturn0file2L139-L148

## Önerilen akış

### A. CVAT export dosyalarını koy
- `images/` içine resimler
- `labels/` içine aynı isimli `.txt` label dosyaları

### B. Dataset split yap
```bash
prepare_dataset
```

Bu komut:
- image/label eşleşmesini kontrol eder
- train/val split yapar
- `data/dataset/data_carton.yaml` üretir
- log yazar
- eksik dosya varsa hata verir

### C. Modeli eğit
```bash
train
```

### D. Validate et
```bash
validate
```

### E. Test et

#### image klasörü için
```bash
image
```

Varsayılan olarak:
```text
data/demo/unseen_images/
```
klasörünü okur.

Farklı klasör vermek istersen:
```bash
image --source /home/emrah/test_images
```

#### video için
```bash
video --source /home/emrah/test_videos/carton.mp4
```

#### webcam için
```bash
camera
```

## Model seçimi

Başlangıç için default model:

```text
yolo11n.pt
```

Sebep:
- hafif
- eğitim hızlı
- ilk baseline için uygun
- ileride edge/deployment tarafına geçişte daha mantıklı
- daha sonra ONNX / NCNN export akışına uygun temel oluşturur

Senin GPU ortamın güçlü olduğu için eğitim rahat olur; ama proje ileride Raspberry Pi 5 tarafına taşınabileceği için küçük model ile başlamak daha kontrollü olur.

## Unseen demo set neden ayrı?

Belgelerinde özellikle demo için gerçek webcam görüntülerinin önemli olduğu ve train/val dışında ayrıca küçük bir demo set tutmanın faydalı olduğu belirtiliyor. fileciteturn0file1L50-L58 fileciteturn0file2L28-L36

Bu nedenle bu yapıda ayrıca:

```text
data/demo/unseen_images/
data/demo/unseen_labels/
```

klasörleri var.

## Git yaklaşımı

Bu proje `.gitignore` ile büyük veri klasörlerini dışarıda bırakır. Böylece:
- kod repo'da temiz kalır
- dataset ve run çıktıları Git'i şişirmez
- GitHub tarafı daha güvenli olur

## Hızlı komut özeti

```bash
prepare_dataset
train
validate
image
video --source /path/to/file.mp4
camera
export_model
```

## Export

ONNX export örneği:
```bash
export_model
```

Bu, default olarak eğitilmiş en iyi ağırlığı export etmeye çalışır.

## Not

Bu yapı **tek sınıf carton baseline** içindir. Daha sonra grup integrasyonu için:
- `data_team.yaml`
- çok sınıflı ortak dataset
- ortak inference pipeline

kolayca eklenebilir.
