import numpy as np
import sys, os, errno
from skimage import io, transform

choose_downsize = False

all_aberdeen_images = []
def file_num(x):
    return(int(x.split('.')[0]))
for filename in sorted(os.listdir(sys.argv[1]), key = file_num):
    all_aberdeen_images.append(io.imread('{}/{}'.format(sys.argv[1], filename)))

all_aberdeen_images = np.array(all_aberdeen_images)
ave_face = all_aberdeen_images.mean(axis=0)
all_aberdeen_images_centered = all_aberdeen_images - ave_face

if choose_downsize:
	downsize = 100
	ave_face_resized = transform.resize(ave_face, (downsize,downsize,3))

	all_aberdeen_images_centered_reshaped = []
	for i in range(all_aberdeen_images_centered.shape[0]):
	    all_aberdeen_images_centered_reshaped.append(transform.resize(all_aberdeen_images_centered[i],
	                                                                  (downsize,downsize,3)))
	all_aberdeen_images_centered_reshaped = np.array(all_aberdeen_images_centered_reshaped)
else:
	downsize = 600
	all_aberdeen_images_centered_reshaped = all_aberdeen_images_centered
	ave_face_resized = ave_face

U, s, V = np.linalg.svd(all_aberdeen_images_centered_reshaped.reshape(415,downsize*downsize*3).T,
                        full_matrices=False)

top_4_eigenfaces_vector = U[:,0:4].copy().T
top_4_eigenfaces_img = top_4_eigenfaces_vector.reshape(4,downsize,downsize,3)

original_face_centered = io.imread('{}/{}'.format(sys.argv[1], sys.argv[2]))
original_face_centered = original_face_centered - ave_face
original_face_centered = transform.resize(original_face_centered, (downsize,downsize,3))
reconstructed_face = np.zeros((downsize,downsize,3))
for a in range(4):
    eigenface = top_4_eigenfaces_img[a]
    coeff = np.sum(np.multiply(original_face_centered, eigenface))
    reconstructed_face += coeff * eigenface

reconstructed_face += ave_face_resized
reconstructed_face -= np.min(reconstructed_face)
reconstructed_face /= np.max(reconstructed_face)
reconstructed_face *= 255

io.imsave('reconstruction.jpg', reconstructed_face.astype(np.uint8))
    
print('finished reconstructing image!')
