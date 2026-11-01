# PIT WALL · F1 yarış sonucu tahmini

Formula 1 yarışlarının bitiş sıralamasını, yalnız yarış başlamadan önce bilinen
verilerle tahmin eden bir model ve yerel Streamlit paneli.

Veriler OpenF1, Jolpica, MultiViewer ve resmî FIA belgelerinden toplanır; hiçbiri
giriş gerektirmez ([veri kaynakları](#veri-kaynakları)). Tahmin, FIA final gridi yayımlandıktan sonra ve yarış başlamadan önce
üretilir; kaydedilen tahmin dosyası bir daha değiştirilmez.

## Model

Ana model **PenaltyAwareHybrid**:

- Grid cezası almayan pilotlar için CatBoost ile start sırasına göre beklenen
  sıra kazancı/kaybı tahmin edilir (grid-delta). Bu kazancın ağırlığı yalnız
  önceki yarışların örneklem dışı tahminlerinden seçilir.
- Grid cezası alan pilotlar, start sırasından bağımsız performansı ölçen
  CatBoostRanker sırasıyla yerleştirilir; böylece cezayla geriden başlayan güçlü
  bir pilotun ilerlemesi modellenebilir.

Karşılaştırma için Grid (start sırası), Ridge, CatBoostRanker ve
CatBoostGridDelta modelleri de aynı backtest'te çalışır.

### 2026 walk-forward backtest

Her round yalnız kendisinden önceki roundlarla eğitilir (round 5–18, 308 pilot
satırı). MAE tahmin edilen ve gerçek bitiş sırası arasındaki ortalama mutlak
farktır; CleanMAE yalnız yarışı tamamlayan pilotları sayar.

| Model              |  MAE | CleanMAE |
|--------------------|-----:|---------:|
| PenaltyAwareHybrid | 3.10 |     2.59 |
| CatBoostRanker     | 3.25 |     2.71 |
| CatBoostGridDelta  | 3.29 |     2.76 |
| Grid               | 3.29 |     2.75 |
| Ridge              | 3.44 |     2.89 |

Bu sonuçlar geriye dönük testtir, canlı tahmin kaydı değildir.

### Sızıntı korumaları

- Yarış sonrası sütunlar (sonuç, durum, tamamlanan tur vb.) model izin
  listesine giremez; yarış öncesi tabloda doluysa tahmin durur.
- Pilot ve takım form özellikleri yalnız önceki roundlardan hesaplanır.
- Tahmin, yarışın UTC başlangıç saatinden sonra kaydedilmez ve mevcut tahmin
  dosyasının üzerine yazılmaz.
- Geriye dönük denemelerde FIA final gridinin yarıştan önce yayımlandığı
  doğrulanır; bu çıktılar ayrı klasörde tutulur ve canlı tahmin gibi sunulmaz.

## Kurulum

Python 3.10+ ile:

```powershell
python -m pip install -r requirements.txt
```

FastF1 önbelleği (`data/cache/`) depoda yoktur; gereken oturumlar ilk
çalıştırmada indirilir. Veri toplamak için internet bağlantısı gerekir.

### Veri kaynakları

Formula 1, Ekim 2026'dan beri zamanlama arşivine giriş şartı koyuyor (FastF1'e
göre F1 TV Access veya üstü). Proje bu arşive bağlanmaz:

| Veri | Kaynak |
|------|--------|
| Antrenman ve sprint turları, lastik, stint | [OpenF1](https://openf1.org) |
| Sıralama, sprint ve yarış sonuçları | [Jolpica](https://api.jolpi.ca) (FastF1 üzerinden) |
| Final grid, grid cezaları, final yarış sınıflandırması | FIA belgeleri |
| Viraj sayısı ve yaklaşık pist uzunluğu | MultiViewer |
| Yarış takvimi | FastF1 |

OpenF1 ve MultiViewer gayriresmî servislerdir. OpenF1'e ulaşılamazsa tahmin durmaz:
antrenman ve sprint turu özellikleri `MISSING` işaretlenir, pist kimliği geçmiş
kayıtlardan alınır. Round 18 tahmini antrenman verisi olmadan üretilmiştir.

`backfill_track_history.py` hâlâ F1 arşivindeki pist kimliğine bağlıdır ve giriş
olmadan çalışmaz; 2022–2025 pist kayıtları depoda hazırdır.

## Kullanım

### Panel

```powershell
python -m streamlit run app.py
```

Windows'ta `PITWALL_Baslat.cmd` aynı komutu çalıştırır; kullanıcı klasöründe
Anaconda varsa onu, yoksa `PATH` üzerindeki Python'u kullanır. Panelin bölümleri
[docs/dashboard.md](docs/dashboard.md) dosyasında anlatılır.

### Yarış haftası akışı

```powershell
# FIA final gridi yayımlandıktan sonra, yarıştan önce: veriyi topla ve tahmin et
python scripts/predict_race.py 2026 18

# Yarış bittikten ve FIA final sınıflandırması yayımlandıktan sonra
python scripts/collect_result.py 2026 18

# Modelleri sezon geçmişi üzerinde yeniden karşılaştır
python scripts/backtest_models.py 2026
```

Diğer betikler:

| Betik | Ne yapar |
|-------|----------|
| `scripts/collect_weekend.py <sezon> <round>` | Yalnız yarış öncesi özellik tablosunu toplar |
| `scripts/backfill_training_history.py <sezon> <son_round>` | Sezonun tamamlanan roundlarını sırayla eğitim geçmişine ekler |
| `scripts/backfill_track_history.py <ilk_sezon> [son_sezon]` | Eski sezonlardan pist karakteri verisi toplar |

## Proje yapısı

```text
app.py                     PIT WALL Streamlit paneli
src/f1_prediction/
  collection.py            yarış öncesi özellik toplama
  openf1.py                OpenF1 turlarını FastF1 biçimine çevirme
  practice.py              antrenman temposu ve uzun stint özellikleri
  qualifying.py            sıralama farkları ve takım arkadaşı karşılaştırması
  sprint.py                sprint yarışı özellikleri
  track.py                 pist geometrisi ve tarihsel pozisyon değişimi
  fia_grid.py, grid.py     FIA final grid ve grid cezası ayrıştırma
  fia_race.py, race.py     FIA final yarış sınıflandırması
  history.py, backfill.py  eğitim geçmişi ve toplu veri toplama
  modeling.py              modeller ve walk-forward backtest
  prediction.py            yarış öncesi ve geriye dönük tahmin akışları
  dashboard.py             panelin veri okuma yardımcıları
scripts/                   komut satırı betikleri
tests/                     pytest testleri
data/
  processed/               özellik tabloları, eğitim geçmişi, backtest ve tahminler
  raw/fia/                 indirilen FIA PDF belgeleri
```

## Test

```powershell
python -m pytest -q
```
