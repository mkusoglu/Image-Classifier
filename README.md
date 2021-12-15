# Image-Classifier
CIFAR-100 veri setinden 5 sınıfın görüntüsü alınmıştır. Bu görüntülerin sınıflandırılması için test ve eğitim için ayrıldılar. Görüntülerin sınıflandırılması CNN ile yapılmıştır.

## Kullanılan Sınıflar
![image](https://user-images.githubusercontent.com/46621453/146280575-5cde1dc5-413b-4762-8cfe-3594e1eb8a1e.png)

## Modelin Blok Şeması
Kurulan CNN modelinin blok şeması aşağıdaki gibidir.
![image](https://user-images.githubusercontent.com/46621453/146280609-d12ce8c0-4420-43c3-9d02-d020b58fefc6.png)

## Eğitim
![image](https://user-images.githubusercontent.com/46621453/146280709-bafa51b7-c81e-4ab8-9071-af2fb1ea3161.png)
![image](https://user-images.githubusercontent.com/46621453/146280713-a9654292-628b-4368-8cb1-144b8909c10a.png)

# TAHMİN SONUÇLARI
Tahmin sonuçlarından önce bir açıklama yapmalıyım. Bu çalışmada kullanılan beş sınıf rakamları ile sırasıyla:
- 23 = cloud
- 24 = cockroach
- 37 = house
- 40 = lamp
- 80 = squirrel

Test veri kümesi ile yapılan tahminlere göre elde edilen confusion matrix:
![image](https://user-images.githubusercontent.com/46621453/146281325-0ad559ce-9a45-4e52-8e32-5eb4b046dc30.png)


Çalışmada bu sınıflara yeni rakamlar verildi. Bu yeni rakamlar göre de kategorikleştirme yapıldı. Yeni rakamları ve kategorik halleri ile sınıflar şu şekilde oldu:
- 1= cloud - Kategorik hali= [1 0 0 0 0]
- 2= cockroach - Kategorik hali= [0 1 0 0 0]
- 3 = house - Kategorik hali= [0 0 1 0 0]
- 4 = lamp - Kategorik hali= [0 0 0 1 0]
- 5 = squirrel - Kategorik hali= [ 0 0 0 0 1 ]

### Cloud Sınıfı için Sonuçlar
Tahmin edilecek görüntü:
![image](https://user-images.githubusercontent.com/46621453/146280880-3e530c93-ac44-40c8-a4d2-22cefc54ae09.png)

Tahmin sonucu:
![image](https://user-images.githubusercontent.com/46621453/146280936-f013e295-8ac5-4a99-8261-9b2174044246.png)

### Cockroach  Sınıfı için Sonuçlar
Tahmin edilecek görüntü:
![image](https://user-images.githubusercontent.com/46621453/146280966-9501d94b-e81c-4f8e-a3a5-a8e93aa062d2.png)

Tahmin sonucu:
![image](https://user-images.githubusercontent.com/46621453/146280990-e565dc36-ffc9-447e-99dc-f5bbc5070f02.png)

### House Sınıfı için Sonuçlar
Tahmin edilecek görüntü:
![image](https://user-images.githubusercontent.com/46621453/146281029-5bde26ca-39a4-4b86-b96e-7af40a5f1044.png)

Tahmin sonucu:
![image](https://user-images.githubusercontent.com/46621453/146281040-d3db5150-f9d1-48a7-ad54-02c8759d2d4c.png)

### Lamp Sınıfı için Sonuçlar
Tahmin edilecek görüntü:
![image](https://user-images.githubusercontent.com/46621453/146281079-9030d289-098e-4c71-be76-dac9b74dca9d.png)

Tahmin sonucu:
![image](https://user-images.githubusercontent.com/46621453/146281082-3f4494ac-b61c-486e-b8a8-358753e70e6d.png)

### Squirrel Sınıfı için Sonuçlar
Tahmin edilecek görüntü:
![image](https://user-images.githubusercontent.com/46621453/146281156-299c28f9-3d61-464f-a64b-4278dc0b3896.png)

Tahmin sonucu:
![image](https://user-images.githubusercontent.com/46621453/146281164-2fff4374-fc50-4cab-9683-907d5e522a75.png)




