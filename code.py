
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

##################################################################
# Görev 1: Veri Temizleme ve Manipülasyonu (%25)
# 1.	Eksik verileri ve aykırı (outlier) verileri analiz edip temizleyin. Eksik verileri tamamlamak için çeşitli yöntemleri (ortalama, medyan gibi) kullanarak eksiklikleri doldurun.
# 2.	Fiyat ve harcama gibi değişkenler için aykırı değerleri tespit edip verisetinden çıkarın veya aykırı değerleri belirli bir aralık içine çekin.
# 3.	Müşteri verisi ile satış verisini musteri_id üzerinden birleştirerek geniş bir veri seti oluşturun.
##################################################################

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



