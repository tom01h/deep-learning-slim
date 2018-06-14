# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.cifar10 import load_cifar10
from PIL import Image

np.set_printoptions(threshold=100)

(x_train, t_train), (x_test, t_test) = load_cifar10(flatten=False)
x_test = x_test[:, :, :, ::-1]
padded = np.pad(x_test, ((0, 0), (0, 0), (4, 4), (0, 0)), mode='constant')
crops = np.random.randint(8, size=(len(x_test), 1))
x_test = np.array([padded[i, :, c[0]:(c[0]+32), :] for i, c in enumerate(crops)])

sample_image = x_test[0:100].reshape((10, 10, 3, 32, 32)).transpose((0, 3, 1, 4, 2)).reshape((320, 320, 3)) # 先頭100個をタイル状に並べ替える
Image.fromarray(np.uint8(sample_image*255)).save('sample.png')
print(t_test[0:100].reshape(10,10))
#pil_img = Image.fromarray(np.uint8(sample_image*255))
#pil_img.show()
