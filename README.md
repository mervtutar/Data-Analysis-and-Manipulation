# Data-Analysis-and-Manipulation
Patika.dev &amp; New Mind AI Bootcamp Homework
# Veri Analizi ve Manipülasyonu

Bu proje, veri temizleme, analiz ve görselleştirme süreçlerini içeren bir veri analizi çalışmasını kapsamaktadır. Proje, satış ve müşteri verileri üzerinde çeşitli veri manipülasyonu ve analiz yöntemlerini kullanarak sonuçlar elde etmeyi amaçlamaktadır.

## 📚 İçerik

1. [Giriş](#giriş)
2. [Proje Hedefleri](#proje-hedefleri)
3. [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
4. [Veri Setleri](#veri-setleri)
5. [Görevler](#görevler)
6. [Sonuçlar](#sonuçlar)
---

### Giriş

Bu proje kapsamında, satış ve müşteri verileri üzerinde:
- **Veri temizleme**,
- **Zaman serisi analizi**,
- **Kategorik ve sayısal analiz**,
- **İleri düzey veri manipülasyonu** ve
- **Tahmin modelleri** uygulanmıştır.

Ayrıca, müşteri davranışlarını anlamak için Pareto ve cohort analizleri yapılmıştır.

---

### Proje Hedefleri

1. Karmaşık veri manipülasyonu ve analiz becerilerini geliştirmek.
2. Verilerden anlamlı sonuçlar çıkararak içgörü sağlamak.
3. Görselleştirme teknikleriyle sonuçları sunmak.

---

### Kullanılan Kütüphaneler

Projenin uygulanması için aşağıdaki Python kütüphaneleri kullanılmıştır:
- **Pandas** (Veri manipülasyonu)
- **NumPy** (Sayısal işlemler)
- **Matplotlib/Seaborn** (Görselleştirme)
- **scikit-learn** (Tahmin modelleri)

---

### Veri Setleri

Proje kapsamında iki veri seti kullanılmıştır:
1. **Satış Verisi**  
   - Kolonlar: `tarih`, `ürün_kodu`, `ürün_adı`, `kategori`, `fiyat`, `adet`, `toplam_satis`
2. **Müşteri Verisi**  
   - Kolonlar: `musteri_id`, `isim`, `cinsiyet`, `yas`, `sehir`, `harcama_miktari`

---

### Görevler

#### Görev 1: Veri Temizleme ve Manipülasyonu
- Eksik ve aykırı değerlerin analizi ve temizlenmesi.
- Satış ve müşteri verilerinin `musteri_id` üzerinden birleştirilmesi.

#### Görev 2: Zaman Serisi Analizi
- Haftalık ve aylık bazda satış ve ürün trendlerinin analizi.
- Tarih sütununu kullanarak, her ayın ilk ve son satış günlerinin bulunması,her hafta kaç ürün satıldığının hesaplanması.
- Zaman serisindeki dalgalanmaların grafiklerle görselleştirilmesi.

#### Görev 3: Kategorik ve Sayısal Analiz
- Ürün kategorilerine göre toplam satış miktarı ve her kategorinin tüm satışlar içindeki oranının hesaplanması.
- Müşterilerin yaş gruplarına göre satış eğilimlerinin analiz edilmesi. (Örnek yaş grupları: 18-25, 26-35, 36-50, 50+)
- Kadın ve erkek müşterilerin harcama davranışlarının karşılaştırılması.

#### Görev 4: İleri Düzey Veri Manipülasyonu
- Müşterilerin şehir bazında toplam harcama miktarının bulunması ve şehirlerin en çok harcama yapan müşterilere göre sıralanması.
- Ürün ve kategori bazında satış artış oranlarının hesaplanması.
- Her bir kategorinin aylık toplam satışlarının hesaplanması ve değişim oranlarının görselleştirilmesi.

#### Görev 5: Ekstra (Bonus)
- **Pareto Analizi**: Satışların %80’ini oluşturan ürünlerin belirlenmesi.
- **Cohort Analizi**: Müşteri davranışlarının incelenmesi.
- **Tahmin Modeli**: Aylık satış tahmini için bir regresyon modeli oluşturulması.

---

### Sonuçlar

- Zaman serisi analizi, belirli dönemlerdeki satış trendlerini ortaya koymuştur.
- Ürün ve kategori bazlı değişim oranları, hangi alanlarda daha fazla odaklanılması gerektiğini göstermiştir.
- Cinsiyetlere ve yaş gruplarına göre harcama miktarlarının analizi satış stratejileri belirlemede önem arz etmektedir.
- Şehir bazında harcama analizleri, bölgesel satış stratejilerinin geliştirilmesine yönelik içgörüler sunabilir.
- Müşterilerin ilk alışveriş aylarına göre tekrar alım oranları analiz edilmiştir. Cohort analizi, müşteri davranışlarının sürekliliği hakkında bilgi vermiştir.
- Tahmin modeli, aylık satış tahmininde **%87 doğruluk** sağlamıştır (R²: 0.87).



