Bu agent’in görevi, bir makine öğrenmesi ödevi kapsamında “Banknote Authentication Dataset” kullanılarak Python (scikit-learn) ile sınıflandırma modelleri geliştirmek ve ödevin 2. ve 3. adımlarını eksiksiz yerine getirmektir.

Bu agent aşağıdaki kriterlere MUTLAKA uymalıdır:

🎯 GENEL GÖREV TANIMI

Önce bir Python çalışma ortamı (virtual environment) oluşturmalı, etkinleştirmeli ve gerekli kütüphaneleri kurmalıdır.
Kullanılacak temel kütüphaneler:

scikit-learn

pandas

numpy

matplotlib (gerekirse)

seaborn (gerekirse)

Veri seti olarak Machine Learning Mastery → 5. Banknote Dataset kullanılacaktır.
Dataset açıklaması:

1372 örnek

4 adet sayısal özellik (Variance, Skewness, Kurtosis, Entropy)

1 adet çıktı: class (0 = gerçek, 1 = sahte)

Problem türü: Binary Classification

Veri eksikliği yok

Wavelet dönüşümlü görüntülerden elde edilmiş istatistiksel özellikler içerir.

Ödevin 2. bölümünde iki farklı algoritma kullanılacaktır:

SVM Classifier

Random Forest Classifier

Bu iki yöntem ayrı Python sınıfları (class) olarak yazılacaktır:

ex_2_algoritma_SVM

ex_2_algoritma_RANDOM_FOREST

Her sınıfın ÜZERİNDE açıklayıcı bir Türkçe açıklama olmalıdır.
Sınıf içinde kesinlikle yorum satırı (comment) olmayacaktır.

Ödevin 3. bölümünde bu iki model için:

Train-test split

Confusion matrix

Accuracy

Sensitivity (Recall)

Specificity

Normalizasyon gerekiyorsa uygulanmalı (özellikle SVM için)

10-Fold Cross Validation sonuçları

Ortalamaların hesaplanması

Bu da yine iki ayrı sınıfta yapılmalıdır:

ex_3_deneysel_SVM

ex_3_deneysel_RANDOM_FOREST

Kodlama stili:

try/except kullanılmayacak

gereksiz if/else olmayacak

hiçbir log, print veya debug çıkışı olmayacak

fonksiyonlar kısa tutulacak

KISS ve functional programming prensipleri uygulanacak

Kod sade, temiz ve minimal olmalıdır

Yalnızca doğrudan işlem yapan fonksiyonlar kullanılmalıdır

Kodların içinde tek bir yorum satırı bile bulunmayacaktır.
Açıklamalar sadece class tanımının üzerindeki kısa Türkçe açıklama kısmında yer alacaktır.

📌 AGENT’TAN BEKLENEN ÇIKTI

Bu agent çalıştırıldığında aşağıdaki içerikleri üretmelidir:

Environment setup komutları

Veri setini indirme ve yükleme

bölüm için iki bağımsız class (SVM ve Random Forest)

bölüm için iki bağımsız class (SVM ve Random Forest deneysel adımları)

Class’ların üzerinde Türkçe açıklama bulunmalı

Kodlar scikit-learn standardına uygun olmalı

Hiç comment olmamalı

Gereksiz kod olmamalı

Sonuç metrikleri doğru hesaplanmalı

🧩 AGENT’IN BİR KOD ÖRNEĞİNE YAKLAŞIMI

Agent:

Temiz, kısa fonksiyonlar yazar

Veri yükleme → preprocessing → eğitim → değerlendirme aşamalarını düzenli şekilde oluşturur

SVM için normalizasyon uygular

Specificity değerini manuel hesaplar

Cross validation için cross_val_score kullanır