import os, sys, numpy, skimage.io

img_dim = (600, 600, 3)
choose = [200, 247, 331, 354]

def out_trans(M):
    M -= numpy.min(M)
    M /= numpy.max(M)
    return (M * 255).astype(numpy.uint8)

img_paths = os.listdir( sys.argv[1] )
X = numpy.zeros((numpy.prod(img_dim), len(img_paths) ), dtype=numpy.float)
for i in range(len(img_paths)):
    X[:,i] = skimage.io.imread(os.path.join(sys.argv[1], img_paths[i])).reshape(-1)

print('X( M, N(samples))', X.shape)
X_mean = numpy.average(X, axis=1).reshape(-1, 1)

if len(sys.argv) < 3:
    skimage.io.imsave('ave.jpg', numpy.copy( X_mean ).reshape( img_dim ).astype(numpy.uint8) )

    import time
    time_start = time.time()
    U, s, V = numpy.linalg.svd( X - X_mean , full_matrices=False)
    print('SVD run time:', round( time.time()-time_start ) )
    print('shape - U:', U.shape, 's:', (s.shape), 'V:', V.shape)

    for ei in range(10):
        skimage.io.imsave( 'Eigenface0' + str(ei+1) + '.jpg', out_trans( numpy.copy( U[ : , ei ] ).reshape( img_dim ) ) )

    for ei in range(4):
        k = 4
        weights = numpy.dot( (skimage.io.imread( os.path.join( sys.argv[1], str(choose[ei]) + '.jpg' ) ).flatten() - X_mean.reshape(1, -1)), numpy.copy( U[ : , :k ]))
        weights = weights.reshape(4, 1)
        A3output = numpy.dot( U[ : , :k ] , weights )
        A3output += X_mean
        skimage.io.imsave( 'Reconstruction0' + str(ei+1) + '.jpg', out_trans( A3output.reshape( img_dim ) ) )

    ratio = s / numpy.sum( s )
    for ei in range(10):
        print( round( 100 * ratio[ei] , 1), '%')
else:
    U, s, V = numpy.linalg.svd( X - X_mean , full_matrices=False)
    k = 4
    weights = numpy.dot( (skimage.io.imread( os.path.join( sys.argv[1], sys.argv[2] ) ).flatten() - X_mean.reshape(1, -1)), numpy.copy( U[ : , :k ]))
    weights = weights.reshape(4, 1)
    A3output = numpy.dot( U[ : , :k ] , weights )
    A3output += X_mean
    skimage.io.imsave( 'reconstruction.jpg', out_trans( A3output.reshape( img_dim ) ) )