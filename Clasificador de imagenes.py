#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy==1.21')
get_ipython().system('pip install numpy --upgrade')
get_ipython().system('pip install -Uqq fastai ')

from PIL import Image
from numpy import asarray
from fastai.vision.all import *


# In[3]:


foodPath = untar_data(URLs.FOOD)


# In[4]:


#How many images are we dealing with
len(get_image_files(foodPath))


# In[5]:


pd.read_json('/home/freddy.sumba/.fastai/data/food-101/test.json')


# In[6]:


labelA = 'nachos'
labelB = 'ice_cream'


# In[7]:


#Loop through all Images downloaded
for img in get_image_files(foodPath):    
  #Rename Images so that the Label (Samosa or Churros) is in the file name
  if labelA in str(img):
    img.rename(f"{img.parent}/{labelA}-{img.name}")
  elif labelB in str(img):
    img.rename(f"{img.parent}/{labelB}-{img.name}")
  else: os.remove(img) #If the Images are not part of labelA or labelB

len(get_image_files(foodPath))


# In[8]:


def GetLabel(fileName):
  return fileName.split('-')[0]

GetLabel("churros-734186.jpg") #Testing


# In[9]:


dls = ImageDataLoaders.from_name_func(
    foodPath, get_image_files(foodPath), valid_pct=0.2, seed=420,
    label_func=GetLabel, item_tfms=Resize(32))

dls.train.show_batch()


# In[10]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade Pillow')


# In[11]:


learn = vision_learner(dls, resnet34, metrics=error_rate, pretrained=True)


# In[12]:


learn.fine_tune(epochs=1)


# In[16]:






label,_,probs = learn.predict("/home/freddy.sumba/ImagesTest/helado1.jpg")

print(f"This is a {label}.")
print(f"{labelA} {probs[1].item():.6f}")
print(f"{labelB} {probs[0].item():.6f}")


# In[ ]:






label,_,probs = learn.predict("/home/freddy.sumba/ImagesTest/HeladoYNacho.jpg")

print(f"This is a {label}.")
print(f"{labelA} {probs[1].item():.6f}")
print(f"{labelB} {probs[0].item():.6f}")


# In[17]:






label,_,probs = learn.predict("/home/freddy.sumba/ImagesTest/HeladoSalcedo.jpg")

print(f"This is a {label}.")
print(f"{labelA} {probs[1].item():.6f}")
print(f"{labelB} {probs[0].item():.6f}")


# In[18]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for i in range(0,10):
  #Load random image
  randomIndex = random.randint(0, len(get_image_files(foodPath))-1)
  img = mpimg.imread(get_image_files(foodPath)[randomIndex])
  #Put into Model
  label,_,probs = learn.predict(img)

  #Create Figure using Matplotlib
  fig = plt.figure()
  ax = fig.add_subplot() #Add Subplot (For multiple images)
  imgplot = plt.imshow(img) #Add Image into Plot
  ax.set_title(label) #Set Headline to predicted label

  #Hide numbers on axes
  plt.gca().axes.get_yaxis().set_visible(False)
  plt.gca().axes.get_xaxis().set_visible(False)


# In[19]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(6)


# In[20]:


learn.export() #exports model as 'export.pkl' by default


# In[21]:


modelPath = get_files(foodPath, '.pkl')[0]
modelPath


# In[22]:


learn_inf = load_learner(modelPath)
learn_inf.predict(mpimg.imread(get_image_files(foodPath)[0])) #raw prediction


# In[23]:


learn_inf.dls.vocab #Get the labels


# In[24]:


shutil.move(str(modelPath), '/home/freddy.sumba/ImagesTest')


# In[ ]:




