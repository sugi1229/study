import mxnet as mx
import inception
import imageiter
import numpy as np
import os


model_prefix = 'mx_mlp'
num = 50
batch = 20

data_iter = imageiter.ImageIter(
    batch_size=batch,
    data_shape=(3, 299, 299),
    path_imgrec="./test.rec",
    path_imgidx="./test.idx",
    erasing=True 
)

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, num)

mod = mx.mod.Module(symbol=sym,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

mod.bind(for_training=False, data_shapes=[('data', (batch,3,299,299))])
mod.set_params(arg_params, aux_params)

y = mod.predict(data_iter)
print(y)

mx.nd.save('result.npz', y)
