import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio as io
import glob 
import seaborn as sns; sns.set(style="ticks", color_codes=True)
Bands = ["Name"]

df = pd.read_csv("band.csv", usecols=Bands)


image_folder_name = 'EO1H1480472016328110PZ' 
image_format = 'tif' 
band_names = [Bands] 
Nsamples = 20000 
NUMBER_OF_CLUSTERS = 4  
colour_map = 'terrain' 

# creating an image dictionary
images = dict();
for image_path in glob.glob(image_folder_name+'/*.'+image_format):
    print('reading ',image_path)
    temp = io.imread(image_path)
    temp = temp[:].squeeze()
    images[image_path[:]] = tempprint('images have ', np.size(temp),' pixels each')

print (images)
#to Create a 3D numpy array of data.
imagecube = np.zeros([images['EO1H1480472016328110PZ\\EO1H1480472016328110PZ_B001_L1GST.TIF'].shape[0],images['EO1H1480472016328110PZ\\EO1H1480472016328110PZ_B001_L1GST.TIF'].shape[1],np.size(band_names)])
for j in np.arange(np.size(band_names)):
    imagecube[:,:,j] = images[tuple(band_names[j])] # 
imagecube=imagecube/256 #  scaling to between 0 and 1

# display a false colour image
thefigsize = (10,8)# figure size
#plt.figure(figsize=thefigsize)
#plt.imshow(imagecube[:,:,0:3])

# sample random subset of images
imagesamples = []
for i in range(Nsamples):
    xr=np.random.randint(0,imagecube.shape[1]-1)
    yr=np.random.randint(0,imagecube.shape[0]-1)
    imagesamples.append(imagecube[yr,xr,:])
# convert to pandas dataframe
imagessamplesDF=pd.DataFrame(imagesamples,columns = band_names)


# make pairs plot (each band vs. each band)
seaborn_params_p = {'alpha': 0.15, 's': 20, 'edgecolor': 'k'}
#pp1=sns.pairplot(imagessamplesDF, plot_kws = seaborn_params_p)#, hist_kws=seaborn_params_h)

# fit kmeans to samples:
from sklearn.cluster import KMeans

KMmodel = KMeans(n_clusters=NUMBER_OF_CLUSTERS) 
KMmodel.fit(imagessamplesDF)
KM_train = list(KMmodel.predict(imagessamplesDF)) 
i=0
for k in KM_train:
    KM_train[i] = str(k) 
    i=i+1
imagessamplesDF2=imagessamplesDF
imagessamplesDF2['group'] = KM_train
# pair plots with clusters coloured:
pp2=sns.pairplot(imagessamplesDF,vars=band_names, hue='group',plot_kws = seaborn_params_p)
pp2._legend.remove()

# create clustered image 
imageclustered=np.empty((imagecube.shape[0],imagecube.shape[1]))
i=0
for row in imagecube:
    temp = KMmodel.predict(row) 
    imageclustered[i,:]=temp
    i=i+1
# plot the clustered data map
plt.figure(figsize=thefigsize)
plt.imshow(imageclustered, cmap=colour_map) 