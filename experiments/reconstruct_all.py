#!/bin/python
#-----------------------------------------------------------------------------
# File Name : reconstruct_all.py
# Purpose: Reconstructs MNIST digits from pre-trained RBM
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Fri 27 Jun 2014 02:27:16 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy
import meta_parameters
meta_parameters.parameters_script = 'parameters_reconstruct_all'
from common import *
from MNIST_IF_STDP_SEQ_UB import main

#Load pre-trained RBM
Wh,Wc,b_init = load_NS_v2(N_v, N_h, N_c, dataset = '../data/WSCD.pkl')
W = np.zeros([N_v+N_c,N_h])
W[:(N_v),:] = Wh
W[N_v:(N_v+N_c),:] = Wc.T
b_h = b_init[(N_v+N_c):]
b_v = b_init[:N_v]
b_c = b_init[N_v:(N_v+N_c)]

N = 10

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def create_no_Id_class(min_p = 1e-32, max_p = 1-1e-8, seed = None):
    iv_l_seq = range(10)
    Idp = np.ones([N, N_v+N_c])*min_p
    for i in range(N):
        cl = np.zeros(N_c)
        cl[(iv_l_seq[i]*n_c_unit):((iv_l_seq[i]+1)*n_c_unit)] = max_p
        Idp[i,N_v:] = clamped_input_transform(cl, min_p = min_p, max_p = max_p)
        Idp[i,:N_v] = 0.
    Id = (Idp /beta)
    return Id


def wrap_run(Id):
    out = main(W, b_v, b_c, b_h, Id = np.array([Id]))
    Mh, Mv= out['Mh'], out['Mv']
    res = np.array(spike_histogram(Mv,t_sim/2,t_sim)).T[1][:N_v].reshape(28,28)
    return res

if __name__ == '__main__':

    Ids = create_no_Id_class()
    d = et.mksavedir()
    import multiprocessing
    pool = multiprocessing.Pool(10)
    pool_out = pool.map(wrap_run, Ids)
    imshow(tile_raster_images(np.array(pool_out), np.array((28,28)), np.array((2,5)), tile_spacing = (1,1)))
    xticks([]), yticks([])
    bone()
    et.globaldata.pool_out = pool_out
    et.savefig('reconstructed_binary.png', format = 'png')

#    print np.mean(np.array(pool_out) == test_labels[:N])
#    print os.path.dirname(os.path.abspath(__file__))
#    et.mksavedir()
#    et.globaldata.pool_out = pool_out
#    et.globaldata.params = [Wh, Wc, b_vch]
#    et.save()
#    import matplotlib, pylab
#    matplotlib.rcParams['savefig.dpi']=180.
#    matplotlib.rcParams['font.size']=26.0
#    matplotlib.rcParams['figure.figsize']=(6.0,6.0)
#    matplotlib.rcParams['axes.formatter.limits']=[-10,10]
#    pylab.rc('legend', borderaxespad=0., borderpad=.4,
#    handlelength=1.4, labelspacing=0.4)
#
#    figure()
#    ion()
#    raster_plot(Mv, Mh, Mc)
#    axhline(1, color='k', linewidth=2, alpha=0.8)
#    axhline(2, color='k', linewidth=2, alpha=0.8)
#    yticks([.5, 1.5, 2.5],['v$','$h$','$c$'])
#    ylabel('')
#    xlim([0,500])
#    pylab.savefig('paper/raster_reconstruction.png', format='png')
#
#    figure()
#    imshow(np.array(spike_histogram(Mv,.1,1)).T[1].reshape(28,28))
#    xticks([])
#    yticks([])
#    pylab.savefig('paper/reconstruction.png', format='png')
#
#    figure()
#    N = MV.values.shape[0]
#    for i in range(N):
#    if i==9:
#        c='r'
#    else:
#        c='k'
#    plot(np.concatenate([np.array([-0.1]),MV.times]),np.concatenate([np.array([i]),0.7*MV.values[i,:]+i]), c)
#    xlim([-0.1,0.5])
#    ylim([-1,10])
#    yticks(range(10))
#    xticks([0,0.5])
#    xlabel('Time[s]')
#    ylabel('Class Label Neuron #')
#    #gca().add_patch(Rectangle((-0.05,0),0.02,.7, color='k'))
#    #text(-0.07,-0.6, '1.0V')
#    pylab.savefig('paper/vmem.png', format='png')
