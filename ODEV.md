Python ile makine öğrenmesi ve Keras ile derin öğrenme konuları başta olmak üzere Ders Notlarını inceleyiniz. Proje tek kişi yapılmalıdır.  

 

    Hazır Veri Seti Kullanarak Makine Öğrenmesi Kapsamında Bir Sınıflandırma Uygulaması Geliştirilmesi   

 

Python scikit-learn kütüphanesini kullanarak bir makine öğrenmesi uygulaması yazınız.  

 

    Problem ve Veri Seti (10 p) 

 
UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets.php 

https://machinelearningmastery.com/standard-machine-learning-datasets/  

https://towardsdatascience.com/top-sources-for-machine-learning-datasets-bb6d0dc3378b 

https://medium.com/datadriveninvestor/the-50-best-public-datasets-for-machine-learning-d80e9f030279 

Kaggle Datasets bağlantılarındaki veri setlerinden derste anlatılandan (Zambak) farklı herhangi bir veri seti seçiniz.  

 

1.a Seçtiğiniz veri setindeki problemi birkaç cümle ile tanımlayınız (otomobillerin özelliklerine göre çok iyi, iyi, orta, kötü olarak etiketlenmesi… gibi).  

 

1.b Verisetini inceleyip özet bilgileri rapora yazınız: Veri (örnek) sayısını, öznitelik (girdi) sayısını, özniteliklerin neler olduğunu, sınıf sayısını ve sınıfların neler olduğunu rapora yazınız. Örnek: Araba sınıflandırması uygulaması için UCI ML Repository Data Folder’dan Car Evaluation veriseti -> 6 adet feature – öznitelik ve 4 adet kategori sınıf içeriyor : https://archive.ics.uci.edu/ml/datasets/Car%20Evaluation. 

 

    2) Yöntem Kodlama: İki farklı yöntem (MLP Classifier YSA, SVM, k-NN, Decision Tree, Random Forest …) kullanarak sınıflandırma işlemini yapan Python kodunu yazınız. Rapora, seçtiğiniz yöntemlerin isimlerini yazınız ve kodu ekleyiniz (5)  

 

    Deneysel Çalışma (10)   

3.a) Verileri eğitim ve test verisi olarak ayırınız. Madde 2’de seçtiğiniz her 2 yöntem için de Hata (Confusion) matrisini, Doğruluk (accuracy), Duyarlık (sensitivity), Özgüllük (Specificity) değerlerini elde edip rapora ekleyiniz. Gerekiyorsa normalizasyon gibi işlemleri de internetten araştırarak uygulayınız.  

3.b) Madde 2’de seçtiğiniz her 2 yöntem için de, 10-fold cross validation ile de doğruluk değerlerini (ve ortalamasını) elde ederek rapora ekleyiniz. http://scikit-learn.org/stable/modules/cross_validation.html. Sonuçları yorumlayınız.  

 

 

 

    Makine Öğrenmesi Kümeleme ve Regresyon Uygulaması 

     

    Kümeleme (Clustering) ne demektir? Tanımlayınız. Bir kümeleme veri seti bulunuz veya ChatGPT gibi bir Gen AI kullanarak veri seti üretiniz. Veri seti ve problem hakkında kısaca bilgi veriniz. (10)   

    Belirleyeceğiniz 2 adet kümeleme yöntemini kendi cümlelerinizle anlatınız. Veri setiniz üzerinde scikit-learn veya diğer bir kütüphane kullanarak bu kümeleme yöntemlerini işletiniz, kodu rapora ekleyiniz ve başarı değerlerini elde ederek iki yöntemi karşılaştırınız. (10)   

    Regresyon ne demektir? Tanımlayınız. Regresyon yöntemlerinden 2 tanesinin isimlerini belirtip yöntemleri kısaca anlatınız. Regresyon için basit bir veri seti araştırarak bulunuz veya bir Gen AI kullanarak üretiniz. Rapora veri setinin adını, bağlantısını veya veri setini ekleyiniz. Veri seti hakkında kısaca bilgi veriniz. (5) 

    scikit-learn veya diğer bir kütüphane kullanarak bu veri seti üzerinde istediğiniz bir yöntemle regresyon işlemini yaptırınız ve kodu rapora ekleyiniz. Regresyon yöntemlerinin başarısı hangi ölçütler kullanılarak ölçülür. 2 tanesinin isimlerini yazınız. Formüllerini verip kısaca açıklayınız. Yönteminizin başarı değerlerini rapora ekleyiniz ve sonuçları yorumlayınız. (5)  

        Derin Öğrenme ile Görüntü Tanıma / Sınıflandırma 

    Kendi çekeceğiniz görüntüler (cep telefonu vb. aracılığı ile) üzerinde sınıflandırma yapan bir Derin Öğrenme uygulaması geliştiriniz. Çevrenizde kolaylıkla bulabileceğiniz nesneleri tercih edebilirsiniz.  Mobilyalar, Elektronik Eşyalar, Kırtasiye Ürünleri, Yiyecekler, Giysiler, … gibi konulardan birisi olabilir. Ayakkabı türleri (bot, klasik ayakkabı) gibi bir nesnenin farklı çeşitleri de düşünülebilir.  

 

    En az 3 sınıftan ve her bir sınıf için en az 10’ar tane görüntüden oluşan bir veri seti oluşturunuz (Örnek: Kalem, Defter ve Çanta sınıflarındaki görüntüler gibi). Raporda, örnek görüntülerle birlikte oluşturduğunuz veri seti hakkında bilgi veriniz. (10) 

    Görüntüleri Eğitim ve Test verisi olarak ayırınız. İki farklı derin öğrenme modeli oluşturarak veya kullanarak tanıma / sınıflandırma işlemini yapınız. Kaynak kodla birlikte, 2 derin öğrenme modelinin ismini, özelliklerini ve elde edilen başarı değerlerini rapora ekleyiniz. Sonuçları yorumlayıp karşılaştırınız. (15) 

 

    Özdeğerlendirme Tablosu: (Aşağıdaki tablo da öğrenci tarafından doldurulmalıdır) 

Özdeğerlendirme tablosu teslim edilmeyen projeler değerlendirilmemektedir.  