- 測試影片  
> https://github.com/knnv5h/ITEE2024/assets/43922704/b482270a-d944-4e30-bc31-d8199efb0d45
>
- Code  
```python
from tensorflow import keras
(train_image, train_label), (test_image, test_label) = keras.datasets.mnist.load_data()

print("train image dataset =", train_image.shape)
print("train label dataset =",train_label.shape)
print("test image dataset =",test_image.shape)
print("test label dataset =",test_label.shape)
```

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14,14)) #設定圖片呈現大小

for i in range(0,10):
  ax=plt.subplot(5,5,1+i)
  ax.imshow(train_image[i])
  title= "label=" +str(train_label[i])
  ax.set_title(title, fontsize=14)
plt.tight_layout()
plt.show()
```
