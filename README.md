# Telco Customer Churn Prediction

## Project Description

* Pada dataset Telco Customer Churn, terdapat rincian data yaitu Customer ID, Gender, Payment Method, Monthly dan Total charge serta status apakah pelanggan tersebut churn atau tidak. 
* Tujuan dari project ini adalah untuk menciptakan model machine learning yang dapat melakukan prediksi pelanggan bagaimana yang akan churn dan tidak churn, sehingga dapat membantu perusahaan dalam mengambil tindakan dan keputusan selanjutnya untuk mempertahankan konsumen.

## Project Cycle
* Identifying activities
* Understanding the business & data
* Data preparation & pre-processing
* Exploratory data analysis
* Feature importance 
* Feature Engineering
* Modeling
* Evaluation
* Conclussion

## Identifying Activities
Pada project ini terdapat beberapa aktivitas yang akan dilakukan seperti memahami tujuan bisnis dari data, membersihkan data, melakukan analisis pada data, ekstraksi fitur, membangun model, melakukan evaluasi pada model dan yang terakhir menarik kesimpulan mengenai model yang mana yang akan digunakan dan mengapa.

## Understanding the business & data
### Understanding the business
Customer churn merupakan jumlah dari hilangnya pelanggan yang menggunakan layanan/jasa/produk dari perusahaan dengan berbagai alasan. Terdapat beberapa keuntungan apabila melakukan analisis mengenai customer churn seperti kesempatan untuk meningkatkan keuntungan, meningkatkan kepuasan pelanggan, mengetahui target pasar dan meningkatkan kualitas dari produk. 

### Understanding the data
* Memahami definisi dari setiap kolom pada dataset. Pada data ini terdapat beberapa kolom dengan informasi sebagai berikut : 
  * Gender : Informasi jenis kelamin pelanggan apakah pria atau wanita
  * Payment method : Informasi jenis pembayaran dari pelanggan
  * Monthly charges : Tagihan bulanan dari pelanggan
  * Total charges : Jumlah tagihan pelanggan selama berlangganan
* Memahami tipe data dari setiap kolom seperti kategorikal dan numerikal
* Memahami isi dari setiap kolom pada data
  * Gender : Female/Male
  * Payment Method : Bank Transfer, Credit Card, Electronic Check, Mailed Check
  * Monthly Charges : Berisi nilai pembayaran pelanggan setiap bulan
  * Total Charges : Berisi nilai total pembayaran pelanggan
  * Churn : Yes/No
  
## Data Preparation & Pre-Processing

* Import seluruh library yang akan digunakan, library dapat dilihat pada [file](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/2.%20Library/library.ipynb) ini.
 * Library yang digunakan adalah : 
   * [numpy](https://numpy.org/)
   * [pandas](https://pandas.pydata.org/docs/index.html)
   * [matplotlib](https://matplotlib.org/)
   * [seaborn](https://seaborn.pydata.org/)
   * [math](https://docs.python.org/3/library/math.html)
   * [sklearn](https://scikit-learn.org/stable/)
   
* [Data Reading](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/3.%20Data-Reading/data-reading.ipynb)

Data yang akan digunakan adalah dataset Telco Customer Churn, dapat diakses pada [file](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Dataset/Dataset10_Telco_Churn.csv) ini. Gunakan kode berikut untuk membaca data.

```ruby 
telco_data = pd.read_csv('https://raw.githubusercontent.com/naomiachoo/data-science-project-one-voice/main/Dataset/Dataset10_Telco_Churn.csv?token=GHSAT0AAAAAAB7V33L5YKO3DRSKO5FMSSXYZAHIOJQ')
telco_data.head()
```
```output```

![image](https://user-images.githubusercontent.com/70925629/225082263-26748564-071e-4e27-9ae6-828c9fc97b96.png)

* [Data Preparation](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/4.%20Data-Preparation/data_preparation.ipynb)

Pada bagian ini akan dilakukan pemeriksaan pada tipe data dari setiap kolom, apakah sesuai atau tidak dengan isi data nya. 

```ruby
#check columns data type
data_type = pd.DataFrame(telco_data.dtypes).T.rename(index={0:'Columns Type'})
data_type
```

```output```

![image](https://user-images.githubusercontent.com/70925629/225084520-fa3c4c1c-b72d-4e67-a395-27df26ff1119.png)

Pada data ini ditemukan bahwa kolom ```TotalCharges``` memiliki tipe data object padahal data nya berupa angka-angka desimal, oleh sebab itu perlu dilakukan konversi ke tipe data numerikal yaitu float.

Namun sebelum melakukan konversi, perlu dilakukan lagi pemeriksaan pada setiap data apakah setiap angka desimal memiliki koma atau titik. Karena apabila yang digunakan pada data adalah koma, maka akan sulit untuk melakukan konversi. Pemeriksaan dan pengubahan  dapat dilakukan dengan kode berikut menggunakan fungsi ```replacee```


```ruby
#function to replace number separator
def replacee(s):
    i=str(s).find(',')
    if(i>0):
        return s[:i] + '.' + s[i+1:]
    else :
        return s
```













