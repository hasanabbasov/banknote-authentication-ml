Derin Öğrenme ile Görüntü Tanıma / Sınıflandırma – Proje Planı

Bu bölümde ödev kapsamında geliştireceğim derin öğrenme tabanlı görüntü sınıflandırma uygulamasının tüm planı, veri seti yapısı, kullanılacak modeller ve değerlendirme ölçütleri açıklanmaktadır.

1) Veri Seti: Sınıflar ve Görseller

Ödev gereği en az 3 sınıf ve her sınıfta en az 10 farklı görüntü kullanılmalıdır.

Bu proje için aşağıdaki sınıflar seçilmiştir:

📌 Sınıflar

Kalem – 10 farklı kalem fotoğrafı

Defter – 10 farklı defter fotoğrafı

Mouse – 10 farklı mouse fotoğrafı

📸 Görüntülerin Özellikleri

Tümü gerçek nesnelerin kendim tarafından farklı açılardan çekilmiş fotoğrafları olacaktır.

Her sınıf için 10 farklı fiziksel ürün kullanılacaktır.

Görseller farklı açılardan (üst, yan, çapraz) çekilecektir.

Farklı ışık koşullarından faydalanılacaktır.

Tüm resimler modele verilmeden önce 224×224 boyutuna dönüştürülecektir.

Bu yapı, hem küçük veri üzerinde derin öğrenme testleri için uygundur hem de sınıflar görsel olarak birbirinden kolayca ayrılabildiği için model başarısı açık şekilde gözlemlenebilir.

2) Veri Bölünmesi: Eğitim ve Test

Toplanan 30 görüntü aşağıdaki gibi ikiye ayrılacaktır:

Eğitim seti (%80) → Model öğrenme sürecinde kullanılır

Test seti (%20) → Modelin performansını ölçmek için tutulur

Küçük dataset nedeniyle eğitim verisinin çeşitliliğini artırmak için Data Augmentation (veri artırma) uygulanacaktır.

3) Kullanılacak 2 Derin Öğrenme Modeli

Ödev gereği iki farklı derin öğrenme modeli ile sınıflandırma yapılacaktır.

Model 1: CNN (Convolutional Neural Network) – Sıfırdan Oluşturulmuş

Bu model tamamen sıfırdan aşağıdaki yapıda tasarlanacaktır:

Conv2D + ReLU

MaxPooling

Dropout

Flatten

Dense (Softmax çıkış katmanı)

Bu model küçük datasetlerde temel bir karşılaştırma noktası sağlar.

Model 2: Transfer Learning – MobileNetV2 veya EfficientNetB0

Bu model daha gelişmiş olup:

Önceden büyük veri üzerinde eğitilmiş

Özellik çıkarımı güçlü

Küçük veri üzerinde yüksek doğruluk sağlayan

bir mimaridir.

Kullanılacak yapı:

Pretrained base model (MobileNetV2)

Base model dondurulacak (fine-tuning yapılmayabilir)

Üzerine:

GlobalAveragePooling

Dense katmanlar

Softmax çıkış katmanı

Bu yaklaşım küçük veri setlerinde özellikle yüksek performans sağlar.

4) Eğitim Süreci

Her iki model için ortak adımlar:

✔ Veri Yükleme

Keras ImageDataGenerator ile klasör bazlı otomatik yükleme.

✔ Veri Artırma (Augmentation)

Rotation (10–20 derece)

Width/height shift

Zoom

Horizontal flip

✔ Eğitim Parametreleri

Batch size: 16

Epoch: 10–20 (final seçim eğitimdeki duruma göre)

Loss: Categorical Crossentropy

Optimizer: Adam

✔ Kaydedilecek Çıktılar

Eğitim ve doğrulama loss/accuracy grafikleri

Confusion matrix

Her iki modelin başarı karşılaştırması

5) Değerlendirme ve Sonuçların Raporlanması

Her iki model için aşağıdaki metrikler karşılaştırılacaktır:

Accuracy (en önemli metrik)

Loss

Precision / Recall / F1-score (istenirse)

Confusion Matrix

Rapor kısmında yer alacak:

Eğitime ait grafikler

3 sınıftan örnek görseller

Her modelin test doğruluk oranı

Hangi modelin neden daha başarılı olduğuna dair kısa yorum

Genellikle MobileNetV2 gibi transfer learning modelleri:

küçük datasetlerde çok daha iyi sonuç verir

hızlı öğrenir

daha az parametre ile daha kararlı performans sağlar

Bu nedenle sonuç kısmında bu durum açıkça raporlanacaktır.

✔ Ödev Gereksinimlerine Uyum
Ödev Maddesi	Karşılığı
En az 3 sınıf	Kalem – Defter – Mouse
Her sınıfta 10 görüntü	10 gerçek farklı ürün fotoğrafı
Görüntü toplama	Telefon kamerası ile çekilmiş
Eğitim / Test ayrımı	%80 - %20
İki model kullanma	CNN + Transfer Learning
Performans karşılaştırma	Accuracy, confusion matrix, grafikler
Raporlama	Görseller + eğitim sonuçları + yorum