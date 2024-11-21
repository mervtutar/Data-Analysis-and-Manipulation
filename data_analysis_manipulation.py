
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
    print("################ NA #################")
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

# grafiklerde Toplam Satışta 35 aykırı değer gözlemlendi

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
plt.figure(figsize=(12, 6))
sns.lineplot(data=haftalik_kategori_trend, x="tarih", y="toplam_satis", hue="kategori", marker="o")
plt.title("Haftalık Kategori Bazında Toplam Satış Trendleri")
plt.xlabel("Tarih")
plt.ylabel("Toplam Satış")
plt.xticks(rotation=45)
plt.legend(title="Kategori")
plt.tight_layout()
plt.show()


# Haftalık bazda toplam satış ve adet analizi (Ürün bazında)
haftalik_urun_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='W'), 'ürün_adi']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(haftalik_urun_trend.head(15))
plt.figure(figsize=(12, 6))
sns.lineplot(data=haftalik_urun_trend, x="tarih", y="toplam_satis", hue="ürün_adi", marker="o")
plt.title("Haftalık Ürün Bazında Toplam Satış Trendleri")
plt.xlabel("Tarih")
plt.ylabel("Toplam Satış")
plt.xticks(rotation=45)
plt.legend(title="Ürün Adı")
plt.tight_layout()
plt.show()


# Aylık bazda toplam satış ve adet analizi (Kategori bazında)
aylik_kategori_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='M'), 'kategori']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(aylik_kategori_trend.head(10))
plt.figure(figsize=(12, 6))
sns.lineplot(data=aylik_kategori_trend, x="tarih", y="toplam_satis", hue="kategori", marker="o")
plt.title("Aylık Kategori Bazında Toplam Satış Trendleri")
plt.xlabel("Tarih")
plt.ylabel("Toplam Satış")
plt.xticks(rotation=45)
plt.legend(title="Kategori")
plt.tight_layout()
plt.show()


# Aylık bazda toplam satış ve adet analizi (Ürün bazında)
aylik_urun_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='M'), 'ürün_adi']).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()
print(aylik_urun_trend.head(15))
plt.figure(figsize=(12, 6))
sns.lineplot(data=aylik_urun_trend, x="tarih", y="toplam_satis", hue="ürün_adi", marker="o")
plt.title("Aylık Ürün Bazında Toplam Satış Trendleri")
plt.xlabel("Tarih")
plt.ylabel("Toplam Satış")
plt.xticks(rotation=45)
plt.legend(title="Ürün Adı")
plt.tight_layout()
plt.show()



# tarih sütunundan ay bazında gruplama
ilk_satis_gunleri = satis_df.groupby(satis_df['tarih'].dt.to_period('M')).first().reset_index(drop=True)
son_satis_gunleri = satis_df.groupby(satis_df['tarih'].dt.to_period('M')).last().reset_index(drop=True)

print("Her ayın ilk satış günleri:\n",ilk_satis_gunleri[['tarih', 'toplam_satis']].head())
print("Her ayın son satış günleri:\n",son_satis_gunleri[['tarih', 'toplam_satis']].head())

# her hafta kaç ürün satıldığına bakalım
haftalik_urun_satis = satis_df.groupby([pd.Grouper(key='tarih', freq='W')]).agg({'adet': 'sum'}).reset_index()
print("\nHaftalık Ürün Satışları:\n", haftalik_urun_satis.head())



# Zaman serisindeki trendleri tespit etmek için grafikler çizdirelim
# haftalık ve aylık bazda toplam satışları hesaplayalım
# haftalık toplam satışlar
haftalik_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='W')]).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()

# aylık toplam satışlar
aylik_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='M')]).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()

# Yıllık toplam satışları hesaplama
yillik_trend = satis_df.groupby([pd.Grouper(key='tarih', freq='Y')]).agg({'toplam_satis': 'sum', 'adet': 'sum'}).reset_index()


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
plt.title('Aylık Toplam Satış Trendleri')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# yıllık trendin çizgi grafiğini çizelim
plt.figure(figsize=(10, 6))
sns.lineplot(data=yillik_trend, x='tarih', y='toplam_satis', marker='o', label='Yıllık Toplam Satış', color='blue')
plt.title('Yıllık Toplam Satış Trendleri')
plt.xlabel('Yıl')
plt.ylabel('Toplam Satış Değeri')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# hangi ürünün adet olarak daha fazla satıldığını inceleyelim
urun_adet_toplam = satis_df.groupby('ürün_adi')['adet'].sum().reset_index()

# adet sayısına göre azalan şekilde sıralama
urun_adet_toplam = urun_adet_toplam.sort_values('adet', ascending=False)

# pasta grafiği ile görselleştirelim
plt.figure(figsize=(8, 8))
plt.pie(urun_adet_toplam['adet'], labels=urun_adet_toplam['ürün_adi'], autopct='%1.1f%%', startangle=140)
plt.title('Ürünlerin Satılma Oranları (Adet Sayısı Bazında) \n')
plt.axis('equal')
plt.show()

# aynı işlemi kategoriler için yapalım
# kategori bazında toplam adet hesaplama
kategori_adet_toplam = satis_df.groupby('kategori')['adet'].sum().reset_index()

# adet sayısına göre azalan şekilde sıralama
kategori_adet_toplam = kategori_adet_toplam.sort_values('adet', ascending=False)

# kategorilere göre Satış Oranları için pasta grafiği oluşturma
plt.figure(figsize=(8, 8))
plt.pie(kategori_adet_toplam['adet'], labels=kategori_adet_toplam['kategori'], autopct='%1.1f%%', startangle=140)
plt.title('Kategorilere göre Satış Oranları (Adet Sayısı Bazında)\n')
plt.axis('equal')  # Eşit oranlı daire
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
# Görev 5: Ekstra (BONUS)
# 1.	Pareto Analizi: Satışların %80’ini oluşturan ürünleri belirleyin (80/20 kuralını uygulayın). Bu ürünleri grafikte gösterin.
# 2.	Cohort Analizi: Müşterilerin satın alım alışkanlıklarını analiz etmek için Pandas ile cohort analizi yapın. Örneğin, ilk kez satın alan müşterilerin tekrar alım oranlarını inceleyin.
# 3.	Tahmin Modeli: Aylık veya haftalık satış miktarlarını tahmin etmek için basit bir regresyon modeli (örneğin Linear Regression) uygulayın. sklearn kullanarak train/test split işlemi ile modeli eğitin ve modelin doğruluğunu ölçün.
####################################################################################

# ürün bazında satışları büyükten küçüğe sıralayalım
urun_satislari = data_df.groupby('ürün_adi')['toplam_satis'].sum().sort_values(ascending=False).reset_index()
toplam_satis = urun_satislari['toplam_satis'].sum()

# Satışların %80’ini oluşturan ürünleri belirlemek için kümülatif yüzde kullanalım
urun_satislari['kümülatif_satis'] = urun_satislari['toplam_satis'].cumsum()
urun_satislari['kümülatif_yuzde'] = (urun_satislari['kümülatif_satis'] / toplam_satis) * 100
pareto_urunler = urun_satislari[urun_satislari['kümülatif_yuzde'] <= 80]
print("satışların %80’ini oluşturan ürünler:\n", pareto_urunler)

# görselleştirelim
plt.figure(figsize=(12, 8))
sns.barplot(x='ürün_adi', y='toplam_satis', data=pareto_urunler, palette='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.title('Pareto Analizi: Satışların %80\'ini Oluşturan Ürünler')
plt.xlabel('Ürün Adı')
plt.ylabel('Toplam Satış')
plt.tight_layout()
plt.show()


#Cohort analizi
# İlk satın alma tarihini hesaplama
ilk_satin_alma = data_df.groupby('musteri_id')['tarih'].min().reset_index()
ilk_satin_alma.rename(columns={'tarih': 'ilk_satin_alma_tarihi'}, inplace=True)

# Cohort başlangıç ayı
ilk_satin_alma['ilk_ay'] = ilk_satin_alma['ilk_satin_alma_tarihi'].dt.to_period('M')

# Veri setine ekleme
data_df = data_df.merge(ilk_satin_alma, on='musteri_id')

# Cohort ve satış ayı
data_df['satis_ayi'] = data_df['tarih'].dt.to_period('M')
data_df['cohort_ay'] = (data_df['satis_ayi'] - data_df['ilk_ay']).apply(lambda x: x.n)

# Cohort analizi pivot table
cohort_data = data_df.pivot_table(
    index='ilk_ay',
    columns='cohort_ay',
    values='musteri_id',
    aggfunc='nunique'
)

# Cohort yüzdeleri
cohort_percentage = cohort_data.divide(cohort_data.iloc[:, 0], axis=0) * 100

# Görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(cohort_percentage, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Cohort Analizi: Müşteri Tekrar Alım Oranları')
plt.xlabel('Cohort Ayı')
plt.ylabel('İlk Alım Ayı')
plt.show()

# Sonuç
print("Cohort Analizi Tablosu:\n", cohort_percentage)


# Tahmin Modeli - aylık toplam satışı tahmin edelim
data_df = pd.read_csv("merged_data.csv")
data_df.head()
data_df['tarih'] = pd.to_datetime(data_df['tarih'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aylık toplam satış
aylik_satis = data_df.groupby(data_df['tarih'].dt.to_period('M')).agg({'toplam_satis': 'sum'}).reset_index()
aylik_satis['tarih'] = aylik_satis['tarih'].astype(str)

# Tarihi sayısal değere dönüştürme
aylik_satis['tarih_num'] = range(1, len(aylik_satis) + 1)

# Model için özellikler ve hedef
X = aylik_satis[['tarih_num']]
y = aylik_satis['toplam_satis']

# Veriyi train/test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Model başarımı
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sonuç
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Gelecek aylar için tahmin
future = pd.DataFrame({'tarih_num': range(len(aylik_satis) + 1, len(aylik_satis) + 13)})
future['tahmin_satis'] = model.predict(future)

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(aylik_satis['tarih_num'], aylik_satis['toplam_satis'], label='Gerçek Veriler', marker='o')
plt.plot(future['tarih_num'], future['tahmin_satis'], label='Tahmin', linestyle='--', color='red')
plt.title('Aylık Satış Tahmini (Linear Regression)')
plt.xlabel('Tarih (Sayısal)')
plt.ylabel('Toplam Satış')
plt.legend()
plt.tight_layout()
plt.show()

#########################################################


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tarih sütunundan ay ve yıl bilgilerini çıkartma
data_df['ay'] = data_df['tarih'].dt.month
data_df['yil'] = data_df['tarih'].dt.year

# One-hot encoding kullanma (kategori, sehir, cinsiyet için)
data_df = pd.get_dummies(data_df, columns=['kategori', 'sehir', 'cinsiyet'], drop_first=True)

# Kontrol etme
print(data_df.head(10))

# Özellikler (Features)
X = data_df[['ay', 'yil', 'fiyat', 'adet', 'yas'] + [col for col in data_df.columns if col.startswith('kategori_') or col.startswith('sehir_') or col.startswith('cinsiyet_')]]

# Hedef değişken (Target)
y = data_df['toplam_satis']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitim
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sonuçları yazdırma
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

