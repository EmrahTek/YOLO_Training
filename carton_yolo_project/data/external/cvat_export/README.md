# CVAT Export Klasörü

Buraya **CVAT'tan export edilmiş hazır dosyalar** gelecek.

Beklenen yapı:

```text
data/external/cvat_export/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/
    ├── image_001.txt
    ├── image_002.txt
    └── ...
```

## Önemli
- Resim ve label dosya isimleri birebir eşleşmeli.
- Örnek:
  - `image_001.jpg`
  - `image_001.txt`

## Not
Bu klasör, CVAT export sonrası manuel kopyalama/yerleştirme için ayrılmıştır.
Yani burası doğrudan:
- CVAT export ZIP açıldıktan sonra
- images ve labels klasörleri düzenlendikten sonra
- pipeline'ın okuyacağı giriş alanıdır
