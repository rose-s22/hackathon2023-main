import skimage
import os
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

num0 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num1 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,1,0,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num2 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,1,0,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num3 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num4 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,1,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num5 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,1,0,0,0,0,0,1],
        [1,0,0,1,1,1,0,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num6 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,0,0,0,0,1],
        [1,0,0,1,1,1,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num7 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,1,0,0,0,0,1],
        [1,0,0,0,1,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num8 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

num9 = [[1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,1,1,0,0,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]]

numbers = [num0,num1,num2,num3,num4,num5,num6,num7,num8,num9]

for i in range(len(numbers)):
  for j in range(len(numbers[i])):
    for k in range(len(numbers[i][j])):
      if numbers[i][j][k] == 1:
        numbers[i][j][k] = (0,0,0)
      else:
        numbers[i][j][k] = (255,255,255)

class pixelator:
  def __init__(self, im=None, desiredSize=100):
    self.desiredSize = desiredSize
    #self.image = None;
    if im is not None:
        self.image = io.imread(im)
  def get_pixelated(self):
    if self.image is None:
      image = io.imread(self.image)
    x = int(self.image.shape[1]/np.sqrt(self.desiredSize))
    y = int(self.image.shape[0]/np.sqrt(self.desiredSize))
    
    self.image = skimage.measure.block_reduce(self.image,block_size=(x,y,1),func=np.mean,func_kwargs={'dtype': np.int64})
    self.image = self.image[:-1, :-1]
    new_image = numberize(self.image)
    return new_image

def numberize(img):
    img1 = skimage.transform.resize(img, (img.shape[0]*10,img.shape[1]*10,3))
    # using kmeans clustering to get the main color pallete of the image
    originalShape = img.shape
    print((img.shape[0]*10,img.shape[1]*10,3))
    clt = KMeans(n_clusters = 10)
    clt.fit(img.reshape(-1,img.shape[2]))
    labeledImg = clt.labels_
    labeledImg = labeledImg.reshape(originalShape[0], originalShape[1])
    print(labeledImg)
    commonColors = clt.cluster_centers_
    print(commonColors)

    for i in range(0,img1.shape[0]):
        for j in range(0,img1.shape[1]):
            img1[i][j] = numbers[labeledImg[i//10][j//10]][i%10][j%10]

    # recoloring image to match final colored-by-number picture
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            x = labeledImg[i,j]
            img[i][j] = commonColors[x]

    patches = []
    for i in range(len(commonColors)):
        patches.append(mpatches.Patch(color=np.round(commonColors[i]/255,3), label=i))

    new_path = os.path.join("static\\uploads\\", "pixelated.png")
    colored_path = os.path.join("static\\uploads\\", "colored.png")

    ax = plt.legend(handles=patches, bbox_to_anchor=(1.135,.5),loc=6)
    plt.axis('off')

    plt.imshow(img1)
    plt.savefig(new_path, bbox_inches='tight')

    plt.clf()
    plt.axis('off')

    plt.imshow(img)
    plt.savefig(colored_path, bbox_images='tight')



    return new_path, colored_path
