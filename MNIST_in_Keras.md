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

```python
import numpy as np
# convert 28x28 image to 784 array, datatype unit8 -> float32
x_train = train_image.reshape(60000, 784).astype('float32') #np.float32
x_test = test_image.reshape(10000, 784).astype('float32')

# normalize the image numbers to 0~1
x_train /= 255   # x_train = x_train / 255
x_test /= 255
```

```python
# convert label numbers to one-hot encoding
from tensorflow.keras.utils import to_categorical

y_train=to_categorical(train_label, 10)
y_test=to_categorical(test_label, 10)
```

```python
# import 建 model 需要的套件
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 定義此 model 模式為 sequential model
model = Sequential()
model.add(Dense(units=64, input_dim=784, kernel_initializer='normal', activation='relu')) #輸入+隱藏層
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax')) #輸出層
print(model.summary())
```

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=x_train, y=y_train, epochs=20, batch_size=64, validation_split=0.2)
```

```python
from google.colab import drive
# 將自己的雲端硬碟掛載上去
drive.mount('/content/gdrive')

from PIL import Image
img = Image.open('num.png').convert('L') #灰階圖片
img = img.resize((28,28))

# 轉換格式
im = np.array(img).astype('float32')
im = (255-im)/255
im = im.reshape(784,1)
im = np.expand_dims(im, 0)

# 預測
from PIL import Image
im = Image.open('num.png').convert('L')
im = img.resize((28,28))
im
```

```python
im = np.array(img).astype('float32')
im = (255-im)/255
im = im.reshape(784,1)
im = np.expand_dims(im, 0)
print("辨識結果為")
np.argmax(model.predict(im))
```
