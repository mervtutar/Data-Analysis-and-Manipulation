
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape) # satır sütun sayısı
    print("##################### Types #####################")
    print(dataframe.dtypes) # tip bilgileri
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    # sayısal sütunlar için temel istatistiksel ölçümleri (ortalama, standart sapma, minimum, maksimum gibi) belirli yüzdelik dilimlerle hesapla
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# müşteri verisi ve satış verisini inceleyelim
satis_df = pd.read_csv("satis_verisi_5000.csv")
musteri_df = pd.read_csv("musteri_verisi_5000_utf8.csv")

print("satış verisi bilgileri/n")
check_df(satis_df)

print("müşteri verisi bilgileri/n")
check_df(musteri_df)

####################################################################################
# Görev 1: Veri Temizleme ve Manipülasyonu (%25)
# 1.	Eksik verileri ve aykırı (outlier) verileri analiz edip temizleyin. Eksik verileri tamamlamak için çeşitli yöntemleri (ortalama, medyan gibi) kullanarak eksiklikleri doldurun.
# 2.	Fiyat ve harcama gibi değişkenler için aykırı değerleri tespit edip verisetinden çıkarın veya aykırı değerleri belirli bir aralık içine çekin.
# 3.	Müşteri verisi ile satış verisini musteri_id üzerinden birleştirerek geniş bir veri seti oluşturun.
####################################################################################

# eksik değerlere bakalım
satis_df.info() # veride eksik değer bulunmadı
musteri_df.info() # # veride eksik değer bulunmadı


# boxplot ile aykırı değerlere bakalım

# Müşteri verisindeki harcama miktarı
plt.subplot(1, 3, 1)
sns.boxplot(data=musteri_df, y='harcama_miktari', color='skyblue')
plt.title('Müşteri Verisi - Harcama Miktarı')
plt.ylabel('Harcama Miktarı')

# Satış verisindeki fiyat
plt.subplot(1, 3, 2)
sns.boxplot(data=satis_df, y='fiyat', color='salmon')
plt.title('Satış Verisi - Fiyat')
plt.ylabel('Fiyat')

plt.subplot(1, 3, 3)
sns.boxplot(data=satis_df, y='toplam_satis', color='lightgreen')
plt.title('Satış Verisi - Toplam Satış')
plt.ylabel('Fiyat')

plt.tight_layout()
plt.show()

# grafiklerde Toplam Satışta aykırı değer gözlemlendi

# aykırı değerleri IQR ile tespit edelim
def check_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Aykırı değer tespiti: müşteri verisindeki harcama miktarı
harcama_outliers, harcama_lower, harcama_upper = check_outliers(musteri_df, 'harcama_miktari')
print("harcama miktarı değişkenindeki aykırı değer sayısı: ",len(harcama_outliers))
# Aykırı değer tespiti: satış verisindeki fiyat ve toplam satış miktarı verisi
fiyat_outliers, fiyat_lower, fiyat_upper = check_outliers(satis_df, 'fiyat')
print("fiyat değişkenindeki aykırı değer sayısı: ",len(fiyat_outliers))
toplam_satis_outliers, toplam_satis_lower, toplam_satis_upper = check_outliers(satis_df, 'toplam_satis')
print("toplam satış değişkenindeki aykırı değer sayısı: ",len(toplam_satis_outliers))

# harcama_miktari ve fiyatta aykırı değer tespit edilmedi ancak toplam satış değişkeninde 35 tane aykırı değer var.

# aykırı değerler, alt ve üst sınırların dışına çıkmayacak şekilde sınır değerleriyle değiştirilsin (Winsorization)

def winsorize_df(df, column, lower_bound, upper_bound):
    df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df

# toplam_satis a Winsorization uygulayalım
satis_df = winsorize_df(satis_df, 'toplam_satis', toplam_satis_lower, toplam_satis_upper)

# aykırı değerleri tekrar kontrol edelim hem check fonksiyonumuzla hem boxplot ile
new_toplam_satis_outliers,toplam_satis_lower, toplam_satis_upper = check_outliers(satis_df, 'toplam_satis')
print("Winsorization sonrası toplam satış değişkenindeki aykırı değer sayısı:", len(new_toplam_satis_outliers))


sns.boxplot(data=satis_df, y='toplam_satis', color='lightgreen')
plt.title('Satış Verisi - Toplam Satış')
plt.ylabel('Toplam Satış')
plt.show()
# hem check_outliers fonksiyonu hem boxplot ile kontrol edildiğinde aykırı değer tespit edilmedi


# Müşteri ve Satış Verilerini Birleştirme
data_df = pd.merge(musteri_df, satis_df, on='musteri_id', how='inner')
data_df.head()

# Birleştirilmiş veri setinin genel bilgilerini kontrol edelim
check_df(data_df)

# Birleştirilmiş veri setini yeni bir CSV dosyasına kaydedelim
data_df.to_csv('merged_data.csv', index=False)

####################################################################################
# Görev 2: Zaman Serisi Analizi (%25)
# 1. Satış verisi üzerinde haftalık ve aylık bazda toplam satış ve ürün satış trendlerini analiz edin.
# 2. tarih sütununu kullanarak, her ayın ilk ve son satış günlerini bulun. Ayrıca, her hafta kaç ürün satıldığını hesaplayın.
# 3. Zaman serisindeki trendleri tespit etmek için grafikler çizdirin (örneğin: aylık satış artışı veya düşüşü).
####################################################################################

satis_df.head()

# Tarih sütununu datetime formatına çevirme
satis_df['tarih'] = pd.to_datetime(satis_df['tarih'])

# Veriyi tarih sırasına göre sıralayalım
satis_df = satis_df.sort_values(by='tarih')


# Haftalık bazda toplam satış ve adet analizi (Kategori bazında)
haftalik_kategori_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='W'), 'kategori']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(haftalik_kategori_trend.head(10))


# Haftalık bazda toplam satış ve adet analizi (Ürün bazında)
haftalik_urun_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='W'), 'ürün_adi']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(haftalik_urun_trend.head(15))


# Aylık bazda toplam satış ve adet analizi (Kategori bazında)
aylik_kategori_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='M'), 'kategori']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(aylik_kategori_trend.head(10))


# Aylık bazda toplam satış ve adet analizi (Ürün bazında)
aylik_urun_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='M'), 'ürün_adi']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(aylik_urun_trend.head(15))


# tarih sütunundan ay bazında gruplama
ilk_satis_gunleri = satis_df.groupby(satis_df['tarih'].dt.to_period('M')).first().reset_index(drop=True)
son_satis_gunleri = satis_df.groupby(satis_df['tarih'].dt.to_period('M')).last().reset_index(drop=True)

print("Her ayın ilk satış günleri:")
print(ilk_satis_gunleri[['tarih', 'kategori', 'ürün_adi', 'toplam_satis']].head())

print("\nHer ayın son satış günleri:")
print(son_satis_gunleri[['tarih', 'kategori', 'ürün_adi', 'toplam_satis']].head())

# her hafta kaç ürün satıldığına bakalım
haftalik_urun_satis = satis_df.groupby([pd.Grouper(key='tarih', freq='W')]).agg({'adet': 'sum'}).reset_index()
print("\nHaftalık Ürün Satışları:\n", haftalik_urun_satis.head())



# Zaman serisindeki trendleri tespit etmek için grafikler çizdirelim
# haftalık ve aylık bazda toplam satışları hesaplayalım
# haftalık toplam satışlar
haftalik_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='W')]).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()

# aylık toplam satışlar
aylik_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='M')]).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()

# haftalık trendin grafiğini çizelim
plt.figure(figsize=(10, 6))
sns.lineplot(data=haftalik_trend, x='tarih', y='toplam_satis', marker='o', label='Haftalık Toplam Satış')
plt.title('Haftalık Toplam Satış Trendleri')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# aylık trendin grafiğini çizelim
plt.figure(figsize=(10, 6))
sns.lineplot(data=aylik_trend, x='tarih', y='toplam_satis', marker='o', label='Aylık Toplam Satış')
plt.title('Aylık Toplam Satış ve Ürün Satış Trendleri')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


####################################################################################
# Görev 3: Kategorisel ve Sayısal Analiz (%25)
# 1.	Ürün kategorilerine göre toplam satış miktarını ve her kategorinin tüm satışlar içindeki oranını hesaplayın.
# 2.	Müşterilerin yaş gruplarına göre satış eğilimlerini analiz edin. (Örnek yaş grupları: 18-25, 26-35, 36-50, 50+)
# 3.	Kadın ve erkek müşterilerin harcama miktarlarını karşılaştırın ve harcama davranışları arasındaki farkı tespit edin.
####################################################################################


# kategorilere göre toplam satış miktarını hesaplama
kategori_satis = satis_df.groupby('kategori')['toplam_satis'].sum().reset_index()
toplam_satis = kategori_satis['toplam_satis'].sum() # Tüm satışların toplamı

# oran
kategori_satis['oran'] = kategori_satis['toplam_satis'] / toplam_satis * 100
print("Kategori Bazında Toplam Satış ve Oranlar:\n", kategori_satis)

# müşteri verisi ve satış verisindeki bilgileri beraber kullanmamız gerektiği için ilk adımda elde ettiğimiz birleştirilmiş veriyi(data_df) kullanalım
aralik = [0, 25, 35, 50, 100]
yas_gruplari = ['18-25', '26-35', '36-50', '50+']
data_df['yas_grubu'] = pd.cut(data_df['yas'], bins=aralik, labels=yas_gruplari, right=False)

# Yaş gruplarına göre toplam satış analizi
yas_grubu_satis = data_df.groupby('yas_grubu')['toplam_satis'].sum().reset_index()

# Toplam satışın yüzdesini hesaplama
total_sales = yas_grubu_satis['toplam_satis'].sum()
yas_grubu_satis['oran'] = (yas_grubu_satis['toplam_satis'] / total_sales) * 100
print("Yaş Gruplarına Göre Toplam Satış ve Oranlar:\n", yas_grubu_satis)

# Pasta Grafiği
plt.figure(figsize=(8, 8))
plt.pie(
    yas_grubu_satis['oran'],
    labels=yas_grubu_satis['yas_grubu'],
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("pastel"),
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Yaş Grubu Bazında Toplam Satış Oranları')
plt.show()


# cinsiyete göre toplam harcama miktarı
count_cinsiyet = data_df['cinsiyet'].value_counts()
print(count_cinsiyet) # Kadın    2531  Erkek    2469


# kadın ve erkeğin harcama davranışları arasındaki farkı gözlemlemek için cinsiyetlere göre ürün ve kategori bazında harcamaları analiz edelim
# Cinsiyetlere göre toplam harcama miktarını kategori bazında analiz etme
kategori_cinsiyet_harcama = data_df.groupby(['kategori', 'cinsiyet'])['harcama_miktari'].sum().unstack()

# Fark Analizi: Kadın ve erkeklerin kategori bazındaki harcama oranlarını hesaplama
kategori_cinsiyet_harcama['Kadın (%)'] = (kategori_cinsiyet_harcama['Kadın'] / kategori_cinsiyet_harcama.sum(axis=1)) * 100
kategori_cinsiyet_harcama['Erkek (%)'] = (kategori_cinsiyet_harcama['Erkek'] / kategori_cinsiyet_harcama.sum(axis=1)) * 100

# Ürün Bazında Harcama
urun_cinsiyet_harcama = data_df.groupby(['ürün_adi', 'cinsiyet'])['harcama_miktari'].sum().unstack()

# Cinsiyetlerin kategori ve ürünler üzerindeki harcamalarını görselleştirelim
# Kategori Bazında Cinsiyet Harcama Karşılaştırması
kategori_cinsiyet_harcama[['Kadın', 'Erkek']].plot(kind='bar', figsize=(12, 6), color=['skyblue', 'salmon'])
plt.title('Kategori Bazında Cinsiyetlere Göre Harcama Miktarı')
plt.ylabel('Toplam Harcama Miktarı')
plt.xlabel('Kategori')
plt.legend(title='Cinsiyet')
plt.tight_layout()
plt.show()

# Ürün Bazında Cinsiyet Harcama Karşılaştırması
urun_cinsiyet_harcama.head(len(data_df["ürün_adi"].unique())).plot(kind='bar', figsize=(14, 7), color=['skyblue', 'salmon'])
plt.title('Ürün Bazında Cinsiyetlere Göre Harcama Miktarı')
plt.ylabel('Toplam Harcama Miktarı')
plt.xlabel('Ürün Adı')
plt.legend(title='Cinsiyet')
plt.tight_layout()
plt.show()

print("Ürün Sayısı:", len(data_df["ürün_adi"].unique())) # 10
print("Kategori Bazında Cinsiyet Harcama Analizi:\n", kategori_cinsiyet_harcama)
print("\nÜrün Bazında Cinsiyet Harcama Analizi:\n", urun_cinsiyet_harcama.head(10))


####################################################################################
# Görev 4: İleri Düzey Veri Manipülasyonu (%25)
# 1.	Müşterilerin şehir bazında toplam harcama miktarını bulun ve şehirleri en çok harcama yapan müşterilere göre sıralayın.
# 2.	Satış verisinde her bir ürün için ortalama satış artışı oranı hesaplayın. Bu oranı hesaplamak için her bir üründe önceki aya göre satış değişim yüzdesini kullanın.
# 3.	Pandas groupby ile her bir kategorinin aylık toplam satışlarını hesaplayın ve değişim oranlarını grafikle gösterin.
####################################################################################

# şehir bazında toplam harcama miktarı
sehir_harcama = data_df.groupby('sehir')['harcama_miktari'].sum().sort_values(ascending=False).reset_index()
print("Şehir Bazında Toplam Harcama\n",sehir_harcama)

# görselleştirelim
plt.figure(figsize=(10, 6))
sns.barplot(x='harcama_miktari', y='sehir', data=sehir_harcama, palette='viridis')
plt.title('Şehir Bazında Toplam Harcama')
plt.xlabel('Toplam Harcama Miktarı')
plt.ylabel('Şehir')
plt.tight_layout()
plt.show()


# ürünlerin ortalama satış oranını hesaplamak için öncelikle tarih sütunundan yıl-ay bilgisi bulalım
data_df['tarih'] = pd.to_datetime(data_df['tarih'])
data_df['yil_ay'] = data_df['tarih'].dt.to_period('M')

# ürün bazında aylık toplam satış
urun_aylik_satis = data_df.groupby(['ürün_adi', 'yil_ay'])['toplam_satis'].sum().reset_index()

# satış artış oranı hesaplama
urun_aylik_satis['satis_degisim'] = urun_aylik_satis.groupby('ürün_adi')['toplam_satis'].pct_change() * 100

# ortalama satış artışı oranı hesaplama
urun_ortalama_artis = urun_aylik_satis.groupby('ürün_adi')['satis_degisim'].mean().reset_index()
urun_ortalama_artis.rename(columns={'satis_degisim': 'ortalama_artis_orani'}, inplace=True)
urun_ortalama_artis.sort_values('ortalama_artis_orani', ascending=False, inplace=True)

# sonuç
print("Ürünlerin Ortalama Satış Artış Oranı:\n", urun_ortalama_artis.head(10))

# aynı işlemi ürün için değil kategoriler için yapalım

kategori_aylik_satis = data_df.groupby(['kategori', 'yil_ay'])['toplam_satis'].sum().reset_index()

# aylık satış değişim oranı
kategori_aylik_satis['degisim_orani'] = kategori_aylik_satis.groupby('kategori')['toplam_satis'].pct_change() * 100

# her kategorinin aylık toplam satış değişim oranını görselleştirelim
plt.figure(figsize=(12, 8))
for kategori in kategori_aylik_satis['kategori'].unique():
    temp_df = kategori_aylik_satis[kategori_aylik_satis['kategori'] == kategori]
    plt.plot(temp_df['yil_ay'].astype(str), temp_df['degisim_orani'], marker='o', label=kategori)

plt.title('Kategori Bazında Aylık Toplam Satış Değişim Oranı')
plt.xlabel('Yıl-Ay')
plt.ylabel('Değişim Oranı (%)')
plt.legend(title='Kategori')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sonuç
print("Kategori Bazında Aylık Satışlar ve Değişim Oranları:\n", kategori_aylik_satis.head(10))


####################################################################################
# Görev 3: İleri Düzey Veri Manipülasyonu (%25)
# 1.	Müşterilerin şehir bazında toplam harcama miktarını bulun ve şehirleri en çok harcama yapan müşterilere göre sıralayın.
# 2.	Satış verisinde her bir ürün için ortalama satış artışı oranı hesaplayın. Bu oranı hesaplamak için her bir üründe önceki aya göre satış değişim yüzdesini kullanın.
# 3.	Pandas groupby ile her bir kategorinin aylık toplam satışlarını hesaplayın ve değişim oranlarını grafikle gösterin.
####################################################################################
