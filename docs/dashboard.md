# PIT WALL arayüzü

Proje klasöründe şu komutu çalıştırın:

```powershell
python -m streamlit run app.py
```

Tarayıcıda açılan yerel adres genellikle `http://localhost:8501` olur. Streamlit
kurulu değilse önce `python -m pip install -r requirements.txt` çalıştırın.

Arayüzde dört bölüm bulunur:

- **Genel bakış:** Sızıntısız walk-forward backtest sonuçları ve model karşılaştırması.
- **Yarışlar:** Seçilen yarışın tahmini, resmî sonucu, MAE ve CleanMAE. Kaydedilmiş
  gerçek yarış öncesi tahmin varsa o gösterilir. Yoksa geriye dönük test tahmini
  açıkça bu adla gösterilir.
- **Yeni tahmin:** FIA final grid yayımlandıktan ve yarış başlamadan önce yeni
  tahmini üretip `data/processed/predictions/<sezon>/` altına kaydeder.
- **Geriye dönük:** Kaçırılan yarışın önceki roundlarını gerekirse indirir,
  yarış öncesi verileriyle ayrı bir tahmin oluşturur, resmî sonucu çekip MAE ve
  CleanMAE gösterir. Çıktı `data/processed/retrospectives/<sezon>/` altında
  ayrı tutulur ve yarış öncesi tahmin gibi sunulmaz. FIA final gridinin yarış
  başlamadan yayımlanmış ve final yarış sonucunun hazır olması gerekir.

Arayüz yerel çalışır. Sunucuya veya internete yayımlanmış bir site değildir.
Yeni tahmin üretirken OpenF1, Jolpica ve FIA verilerine erişmek için internet gerekir.
OpenF1'e ulaşılamazsa tahmin antrenman verisi olmadan üretilir ve arayüz bunu uyarır.
Kaydedilmiş bir yarış öncesi tahmin dosyası yeniden yazılmaz.
