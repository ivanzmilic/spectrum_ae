import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits 
from sklearn.cluster import MiniBatchKMeans

import sys

filename = sys.argv[1]
file = fits.open(filename)

spectra = np.copy(file[0].data[:,:,0,:])
#spectra = spectra.transpose(1,2,0)

NX = spectra.shape[0]
NY = spectra.shape[1]
NL = spectra.shape[2]

print (NX, NY, NL)

spectra = spectra.reshape(NX*NY, NL)

mean_qs = np.mean(spectra[:,0:10])

spectra[:,:] /= mean_qs

kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=128).fit(spectra)

labels = kmeans.labels_

# Plot the label distributions 
plt.figure(figsize=[9,8])
plt.imshow(labels.reshape(NX,NY).T,origin='lower')
plt.colorbar()
plt.tight_layout()
plt.savefig('labels.png',bbox_inches='tight')


zeros = np.asarray(np.where(labels == 0))[0]
ones = np.asarray(np.where(labels == 1))[0]
twos = np.asarray(np.where(labels == 2))[0]
threes = np.asarray(np.where(labels == 3))[0]
fours = np.asarray(np.where(labels == 4))[0]

print (len(zeros))
print (len(ones))
print (len(twos))
print (len(threes))
print (len(fours))

import random

N = 30000
sample = random.sample(zeros.tolist(),N)
sample = np.append(sample, random.sample(ones.tolist(),N))
sample = np.append(sample, random.sample(twos.tolist(),N))
sample = np.append(sample, random.sample(threes.tolist(),N))
sample = np.append(sample, random.sample(fours.tolist(),N))

training_sample = spectra[sample,:]
print (training_sample.shape)

kek = fits.PrimaryHDU(training_sample)
kek.writeto("training_sample_5c_scala.fits",overwrite=True)

fullstokes = file[0].data.reshape(768*768,4,112)
vector_training_sample = fullstokes[sample,:,:]

kek = fits.PrimaryHDU(vector_training_sample)
kek.writeto("training_sample_5c_vector.fits",overwrite=True)
