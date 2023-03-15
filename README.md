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

Gunakan fungsi ```replacee``` pada dataframe ```telco_data``` untuk dapat melakukan pemeriksaan dan penggantian. Kodenya adalah sebagai berikut : 

```ruby

#change the number separator
telco_data['TotalCharges'] = telco_data['TotalCharges'].apply(replacee)

```

Setelahnya lakukan lah konversi dari tipe data object ke float dengan kode sebagai berikut : 

```ruby

#convert TotalCharges dtype
telco_data['TotalCharges'] = pd.to_numeric(telco_data['TotalCharges'], errors = 'coerce')
print(telco_data['TotalCharges'].dtypes)

```

Selanjutnya adalah memeriksa nilai null pada data menggunakan kode berikut : 

``` ruby

#check null values
null_val = data_type.append(pd.DataFrame(telco_data.isnull().sum()).T.rename(index = {0:'Amount of Null Values'}))
null_val

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225224773-c5b9e3a2-e08f-4856-b362-41575ded3b24.png)

Untuk melihat persentasi dari nilai null pada data dapat menggunakan fungsi ```checking_null_values``` berikut : 

```ruby
def checking_null_values(dataset):
  """
  show null values and its percentage
  """
  print('Dimension of the dataset', dataset.shape)
  null_val = pd.DataFrame(dataset.dtypes).T.rename(index={0:'Columns Type'})
  null_val = null_val.append(pd.DataFrame(dataset.isnull().sum()).T.rename(index = {0:'Amount of Null Values'}))
  null_val = null_val.append(pd.DataFrame(round(dataset.isnull().sum()/dataset.shape[0]*100,2)).T.rename(index={0:'Percentage null values'}))
  return null_val.T

```

Kemudian fungsi ini dapat digunakan sebagai berikut : 

```ruby
#exclude the unnamed variable
data = telco_data.iloc[:, 1:]

#show null values and its percentage
checking_null_values(data)

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225225442-f2573c94-ac4e-48dc-b796-077b13ca6796.png)


Selanjutnya adalah memeriksa data pada kolom ```PaymentMethod``` gunakan kode berikut untuk melihat unique value pada kolom tersebut : 

```ruby
#unique element of PaymentMethod
telco_data.PaymentMethod.unique()

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225228076-2f3be775-be80-4a22-b8f3-1bc6aa5cf989.png)

Dapat kita lihat bahwa pada beberapa value terdapat kata `(automatic)` yang jika dimasukkan kedalam data menjadi cukup panjang dan tidak terlalu diperlukan. Sehingga kata tersebut perlu dihilangkan dengan kode berikut : 

```ruby
#remove (automatic) from payment method
telco_data['PaymentMethod'] = telco_data['PaymentMethod'].str.replace(' (automatic)', '', regex=False)

```

Selanjutnya adalah menghapus kolom ```customerID``` karena tidak menjelaskan apakah pelanggan akan churn atau tidak, gunakan kode berikut untuk melakukan drop kolom : 

```ruby
telco_data.drop(columns = 'customerID', inplace = True)
```

Selanjutnya adalah menghapus nilai yang null pada kolom `TotalCharges`. 

Mengapa nilai null tersebut dihapus? Mempertahankan nilai null akan mempengaruhi saat melakukan pemodelan, jika diganti juga dengan nilai lain nilai pada TotalCharges seharusnya lebih besar atau sama dengan nilai pada `MonthlyCharges` sedangkan penggantian nilai belum tentu menghasilkan nilai yang sama. Pada data tidak terdapat informasi yang mendukung mengapa nilai null pada data harus dipertahankan.

Untuk menghapus nilai null gunakan kode berikut : 

```ruby
# drop null values
telco_data = telco_data.dropna()

#show null values and its percentage
checking_null_values(telco_data)

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225230090-fdabfce9-fe02-4fd9-9609-d08b49de6bcf.png)


## [Exploratory Data Analysis](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/5.%20EDA/EDA.ipynb)

Pada tahapan ini data akan dianalisis untuk mengetahui apa saja yang dapat diambil dari data tersebut.

### Data Visualization

#### 1. Response Variable
Pada bagian ini ingin diketahui berapa banyak persentasi dari hasil churn dan tidak churn. Kolom yang dianalisis adalah kolom ```Chun```


```ruby

# create a figure
fig = plt.figure(figsize=(10, 6)) 
ax = fig.add_subplot(111)

data = telco_data['Churn']
# proportion of observation of each class
totals = data.value_counts(normalize=True)

# create a bar plot showing the percentage of churn
totals.plot(kind='bar', 
                   ax=ax,
                   color=['springgreen','salmon'])

# set title and labels
ax.set_title('Proportion of observations of the response variable',
             fontsize=18, loc='left')
ax.set_xlabel('churn',
              fontsize=14)
ax.set_ylabel('proportion of observations',
              fontsize=14)
ax.tick_params(rotation='auto')

# eliminate the frame from the plot
spine_names = ('top', 'right', 'bottom', 'left')
for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225347658-043746ba-027d-4338-8034-79811b0b849d.png)

Setelah menampilkan diagram nya, untuk memudahkan dalam membaca hasil analisis perlu ditampilkan juga dalam bentuk tabel, berapa banyak yang churn atau tidak churn dan berapa persentasi nya.

Gunakan kode berikut untuk menampilkan tabel : 

```ruby
#count totals of response variable
df_response_total = pd.DataFrame(telco_data['Churn'].value_counts())

#Count the percentage
pd.set_option('display.float_format', '{:.2%}'.format)
df_response_percentage = pd.DataFrame(telco_data['Churn'].value_counts(normalize=True))

#concat the dataframe
df_concats = pd.concat([df_response_total,df_response_percentage], axis=1, join="inner")
df_concats.columns = ["Total", "Percentage"]

df_concats

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225350486-63fc76cb-2e82-422e-a773-303dbf49b0d2.png)


#### 2. Numerical Variables

Pada bagian ini, dengan variabel numerical yaitu MontlyCharges dan TotalCharges kami ingin melihat bagaimana pengaruh kedua variabel tersebut pada churn atau tidaknya pelanggan.

Kode yang digunakan untuk menampilkan diagram nya adalah sebagai berikut : 

```ruby
def histogram_plots(columns_to_plot, super_title):
    
    # set number of rows and number of columns
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)

    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # histograms for each class (normalized histogram)
        telco_data[telco_data['Churn']=='No'][column].plot(kind='hist', ax=ax, density=True, 
                                                       alpha=0.5, color='springgreen', label='No')
        telco_data[telco_data['Churn']=='Yes'][column].plot(kind='hist', ax=ax, density=True,
                                                        alpha=0.5, color='salmon', label='Yes')
        
        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

# customer account column names
account_columns_numeric = ['MonthlyCharges', 'TotalCharges']
# histogram of costumer account columns 
histogram_plots(account_columns_numeric, '')

```

```output```

![image](https://user-images.githubusercontent.com/70925629/225355842-daddb967-ef0b-4a2e-ba8b-465fab6b98fc.png)

Berdasarkan analisis pada Monthly Charges dan Total Charges maka didapatkan beberapa kesimpulan yaitu : 

* Semakin besar monthly charges maka probabilitas churn juga mayoritas akan semakin tinggi
* Customer dengan total charges yang tinggi kemungkinan untuk churn semakin rendah

#### 3. Percentage of Churn for Gender Category

Pada bagian ini kami ingin melihat pengaruh variabel gender pada churn atau tidak nya pelanggan


Tampilkan grafik nya dengan kode berikut : 

```ruby

cross_tab_prop_gender.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("Gender")
plt.ylabel("Churn")
plt.show()
```

```output```

![image](https://user-images.githubusercontent.com/70925629/225357069-0c495100-4925-461f-bf84-5ee3c148a357.png)

Berdasarkan analisis pada persentasi churn terhadap gender maka didapatkan beberapa kesimpulan yaitu : 

* Baik perempuan dan laki-laki lebih dominan untuk tidak churn.

* Perempuan memiliki persentasi 73.04% untuk tidak churn 

* Laki-laki memiliki persentasi 73.80%	untuk tidak churn

#### 4. Percetage of Churn for Payment Method Category

Pada bagian ini kami ingin melihat bagaimana pengaruh Payment Method pada churn atau tidak nya pelanggan.

Tampilkan grafik dengan kode berikut : 

```ruby
cross_tab_prop_pm.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("PaymentMethod")
plt.ylabel("Churn")
plt.show()

```

![image](https://user-images.githubusercontent.com/70925629/225359123-49c35a68-9e5b-4f69-9225-49baf65d1cf2.png)


Berdasarkan analisis pada persentasi churn terhadap payment method maka didapatkan beberapa kesimpulan yaitu : 

* Seluruh payment method memiliki persentasi yang dominan untuk tidak churn

* Bank transfer memiliki persentasi 83,27% untuk tidak churn 

* Credit card memiliki persentasi 84,75% untuk tidak churn

* Electronic check memiliki persentasi 54,71% untuk tidak churn

* Mailed check memiliki persentasi 80,80% untuk tidak churn


#### 5. ####Correlation for the numerical variables

Pada bagian ini kami ingin melihat bagaimana korelasi antar kedua variabel numerical.

Tampilkan grafik dengan kode berikut : 

``` ruby
plt.figure(figsize=(12,7))
sns.heatmap(telco_data.corr(), cmap='YlGnBu', annot=True) 
plt.show()
```

```output```

![image](https://user-images.githubusercontent.com/70925629/225359756-559e1400-a519-4180-93c5-2fd22cc15bfd.png)

Analisis ini menghasilkan bahwa korelasi antara Montly Charges dan Total Charges tidak terlalu besar hanya 0.65.

## [Feature Importance](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/6.%20Feature-Importance/feature_importance.ipynb)

Feature importance merupakan tahapan untuk melihat seberapa berpengaruh variabel kategorikal terhadap variabel target

Menggunakan Mutual Information selain membantu dalam memahami data, kita juga dapat mengidentifikasi variabel mana yang mempengaruhi target variabel yang ada pada data. 

Score yang dimiliki oleh variabel PaymentMethod dan gender, apabila semakin mendekati 0 maka dapat disimpulkan bahwa variabel tersebut tidak mempengaruhi variabel target.

Oleh sebab itu kita dapat menyimpulkan bahwa variabel PaymentMethod yang memiliki score 4,45% mempengaruhi hasil pada variabel target.

menggunakan kode berikut kita dapat menghitung persentase setiap variabel : 

```ruby
# function that computes the mutual infomation score between a categorical serie and the column Churn
def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, telco_data.Churn)

# select categorial variables excluding the response variable 
categorical_variables = telco_data.select_dtypes(include=object).drop('Churn', axis=1)

# compute the mutual information score between each categorical variable and the target
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

# visualize feature importance
print(feature_importance)
```

`output`

![image](https://user-images.githubusercontent.com/70925629/225360920-b0d2a30d-a163-44dd-a652-68899b33a22b.png)

Oleh sebab itu kita dapat menyimpulkan bahwa variabel PaymentMethod yang memiliki score 4,45% mempengaruhi hasil pada variabel target.

## [Feature Engineering](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/7.%20Feature-Engineering/feature_engineering.ipynb)

Pda bagian ini fitur akan diekstraksi dan diubah menjadi format yang sesuai dengan model machine learning. 

Pada proyek ini variabel numerik dan kategorikal perlu diubah. Sebagian besar algoritma machine learning memerlukan nilai numerik, oleh sebab itu semua variabel kategorikal akan diubah menjadi label numerik sedangkan variabel numerik akan diubah menjadi skala umum. 

### Label Encoding

Label encoding digunakan untuk mengubah nilai kategorikal dengan nilai numerical. Encoding ini mengubah setiap kategori dengan label numerical.

```ruby
telco_data_transformed = telco_data.copy()

# label encoding (binary variables)
label_encoding_columns = ['gender', 'Churn']

# encode categorical binary features using label encoding
for column in label_encoding_columns:
    if column == 'gender':
        telco_data_transformed[column] = telco_data_transformed[column].map({'Female': 1, 'Male': 0})
    else: 
        telco_data_transformed[column] = telco_data_transformed[column].map({'Yes': 1, 'No': 0}) 
        
telco_data_transformed.head()

```

`output`

![image](https://user-images.githubusercontent.com/70925629/225362538-e9718de3-ab77-4804-98bd-166af199fc00.png)

### One Hot Encoding

one hot encoding menciptakan kolom binary baru untuk setiap level dari variable kategorikal. Berisi 0 dan 1 untuk mengindikasikan ada atau tidak nya kategori pada data. One hot encoding diterapkan pada PaymentMethod pada penelitian ini.

```ruby

# one-hot encoding (categorical variables with more than two levels)
one_hot_encoding_columns = ['PaymentMethod']

# encode categorical variables with more than two levels using one-hot encoding
telco_data_transformed = pd.get_dummies(telco_data_transformed, columns = one_hot_encoding_columns)

```

`output`

![image](https://user-images.githubusercontent.com/70925629/225362918-3203e480-275d-4af2-9ed5-8bf42b652a02.png)

### Normalization

Normalisasi digunakan untuk mengubah kolom numerik menjadi nilai skala yang lebih umum.

```ruby
# min-max normalization (numeric variables)
min_max_columns = ['MonthlyCharges', 'TotalCharges']

# scale numerical variables using min max scaler
for column in min_max_columns:
        # minimum value of the column
        min_column = telco_data_transformed[column].min()
        # maximum value of the column
        max_column = telco_data_transformed[column].max()
        # min max scaler
        telco_data_transformed[column] = (telco_data_transformed[column] - min_column) / (max_column - min_column)   


```

`output`

![image](https://user-images.githubusercontent.com/70925629/225363172-350d60ef-cf78-4b4c-9f9c-3074e96779d7.png)

## [Modelling](https://github.com/naomiachoo/data-science-project-one-voice/blob/main/Script/8.%20Model/model.ipynb)

Pada project ini model yang akan digunakan ada tiga yaitu 

* Logistic Regression
* Decision Tree
* Gradient Boosting

Kode untuk membangun model ini dapat dilihat pada tautan diatas.

## Conclusion

Project ini menghasilkan sebuah kesimpulan bahwa model yang terbaik untuk digunakan adalah Gradient Boosting hal ini dikarenakan pada project ini kami menitik beratkan hasil evaluasi pada precission, dimana data pelanggan yang benar benar churn lah yang ingin dilihat bagaimana pola nya dan membantu perusahaan dalam mengambil keputusan.

Untuk hasil evaluasi dari setiap model dapat dilihat pada gambar berikut : 

![image](https://user-images.githubusercontent.com/70925629/225364690-7de52923-2177-4663-b4a9-2af0bca18f03.png)






