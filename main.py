import mxnet as mx
import inception

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageiter

import logging
logging.basicConfig(level=logging.INFO)

gpu_device=mx.cpu(0)


#batch=50
train_iter = imageiter.ImageIter(
    batch_size=30,
    data_shape=(3, 299, 299),
    path_imgrec="./test.rec",
    path_imgidx="./test.idx"
)

#train_iter.reset()

net = inception.get_symbol(num_classes=3)

model_prefix = 'mx_mlp_v09'
checkpoint = mx.callback.do_checkpoint(model_prefix, 10)
progress_bar = mx.callback.ProgressBar(total=10)
#progress_bar = mx.callback.Speedometer(20, 200)

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

lr_sch = mx.lr_scheduler.FactorScheduler(step=10, factor=0.9)
#mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.005),))

optimizer = {
    'learning_rate': 0.01,
    'wd': 1e-6,
    'momentum': 0.9
}

mod.fit(train_iter,
        eval_data=train_iter,
        optimizer='sgd',
        optimizer_params=optimizer,
        eval_metric='acc',
        num_epoch=100)
