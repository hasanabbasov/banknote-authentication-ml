Kümeleme ve Regresyon Bölümleri İçin Plan ve Prompt

Bu bölümde kümeleme ve regresyon adımlarını, projede nasıl uygulayacağımı ve kod yapısını nasıl organize edeceğimi açıklıyorum. Bu plan aynı zamanda yeni oluşturacağım Python dosyalarındaki sınıfların yapısını belirlemektedir.

1) Veri Kaynağı ve Kullanım Planı

Bu ödevde kümeleme ve regresyon için oteller ile ilgili gerçek veri kullanılacaktır. Elimde iki temel veri kümesi bulunmaktadır:

Hotel Profile Data

Otelin konumu, şehir, yıldız sayısı, oda sayısı gibi temel özellikler içeriyor.

Ayrıca otelin bulunduğu bölge, zincir bilgisi gibi kategorik alanlar da mevcut.

Bu veri kümesi kümeleme için uygundur çünkü sayısal ve kategorik karışık çok özellik var ve doğal gruplar oluşturabilir.

Hotel Email Campaign / Lead Scores

Otellere gönderilen e-posta kampanyalarının aldığı skorlar, toplam kampanya sayısı, minimum–maksimum skor gibi performans ölçümleri içeriyor.

Bu veri kümesi regresyon için uygundur çünkü kampanya sonuçları sayısal olup bir değerin tahmini yapılabilir (örneğin bir otelin gelecekteki kampanya skorunu tahmin etmek).

Veriler aynı hotelId alanı üzerinden birleştirilebilir. Ancak ödev gereği, hem kümeleme hem regresyon işlemleri için verileri ayrı olarak kullanmak daha açık ve anlaşılır olacaktır.

2) Proje Yapısı ve Class Düzeni

Bu bölümde iki sınıf oluşturulacaktır. Kod içinde yorum kullanmayacağım; açıklamalar sadece class dokümantasyonu üzerinde olacak.

(A) Kümeleme – Clustering Class

Dosya adı: ex_4_clustering.py
Sınıf adı: Ex4Clustering

Class açıklamasında şunlar yer alacak:

Bu sınıfın otel profil verisini kullanarak kümeleme yaptığı,

Seçilecek 2 algoritmanın (ör: K-Means, Agglomerative Clustering) neden tercih edildiği,

Verinin nasıl işlendiği (normalizasyon, kategorik encoding),

Sonuçların nasıl üretildiği (inertia, silhouette score),

KISS ve functional yapıya uyulduğu.

Kod:

Veri yükleme fonksiyonu

Preprocessing fonksiyonu

K-Means modeli

Agglomerative Clustering modeli

Sonuç skorlarının döndürülmesi
şeklinde sade fonksiyonlardan oluşacak.

(B) Regresyon – Regression Class

Dosya adı: ex_4_regression.py
Sınıf adı: Ex4Regression

Class açıklamasında şunlar yer alacak:

Kampanya skor verisinin regresyon için kullanıldığı,

Kullanılacak 2 algoritmanın (ör: Linear Regression & Random Forest Regressor) tanımı,

Verideki hedef değişkenin (score veya averageScore) açık tanımı,

Eğitim/test ayrımı, MSE ve R² gibi metriklerle değerlendirme yapıldığı.

Kod:

Veri yükleme fonksiyonu

Preprocessing

Linear Regression modeli

Random Forest modeli

Metriklerin hesaplanması
şeklinde sade fonksiyonlardan oluşacak.

3) Kodlama Prensiplerim

Bu bölümde yazacağım tüm sınıflar için ortak kurallar:

Kod içinde yorum (#) kullanılmayacak.

Açıklamalar sadece class açıklamalarında olacak.

Try/except, gereksiz if blokları, loglama kullanılmayacak.

Kod tamamen sade ve fonksiyonel olacak.

Sınıflar birbirinden bağımsız modüller olacak.

İstenirse ileride notebook veya CLI üzerinden çağrılabilir.

4) Ödeve Uygunluk

Bu plan aşağıdaki ödev maddelerinin birebir karşılığıdır:

“Kümeleme veri seti bulunuz” → Hotel Profile Data kullanılacak

“2 yöntem işletiniz ve başarı değerlerini karşılaştırınız” → K-Means + Agglomerative

“Regresyon veri seti bulunuz” → Hotel Campaign Scores kullanılacak

“2 yöntemle regresyon yapınız” → Linear Regression + Random Forest

“Başarı ölçütleri yazınız” → MSE, MAE veya R²