import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import pandas as pd

import tensorflow as tf


from ap_architectures.utils import load_obj, PRM
from ap_architectures.unet import UNET


def get_flops(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    
    concrete = tf.function(lambda inputs: model(inputs))
    
    concrete_func = concrete.get_concrete_function(
    [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops

#%%

# log_path = '/media/thomas/SSD/phase_retrival/Ckp/Training/UNET_5_skip_no'
log_path = '/media/thomas/SSD/phase_retrival/Ckp/Training/UNET_DCR_3_16_CONV_CONV'
prms, prms_net = load_obj(log_path+'/hyperparameters.txt')
PRM.read_parameters(prms)

model = UNET(PRM.X_SHAPE, prms_net).model
# model = CNN2D(PRM.X_SHAPE, prms_net).model

print(get_flops(model))

#%%

listOutput = (glob.glob("/media/thomas/SSD/phase_retrival/Logs/Training/**/train/*.v2",recursive=True))
# listOutput = (  '/media/thomas/SSD/phase_retrival/Logs/Training/UNET_4_skip/bs_16_m_0.9_lr_5.00E-03/train/events.out.tfevents.1608899890.b02bc07003c4.10885.2649.v2',
                # '/media/thomas/SSD/phase_retrival/Logs/Training/UNET_5_skip/bs_16_m_0.9_lr_5.00E-03/train/events.out.tfevents.1608900880.Aurora-R3.16848.3223.v2')
print(listOutput)
listDF = []
#%%
for tb_output_folder in listOutput:
    try:
        x = EventAccumulator(path=tb_output_folder)
        x.Reload()
        x.FirstEventTimestamp()
        keys = ['batch_loss', 'batch_pix_int', 'batch_pix_phase','batch_ssim_int','batch_ssim_phase'] 

        listValues = {}

        count = [e.count for e in x.Scalars(keys[0])]
        n_steps = np.minimum(1500, len(count)) - 1
        listRun = [tb_output_folder] * n_steps
        printOutDict = {}

        data = np.zeros((n_steps, len(keys)))
        steps = np.zeros(n_steps)

        for i in range(len(keys)):
            for v in range(n_steps):
                if i == 0:
                    steps[v] = x.Scalars(keys[i])[v].step
                data[v,i] = x.Scalars(keys[i])[v].value
                

        printOutDict = {'steps':steps, keys[0]: data[:,0], keys[1]: data[:,1],keys[2]: data[:,2],keys[3]: data[:,3],keys[4]: data[:,4]}

        printOutDict['Name'] = listRun

        DF = pd.DataFrame(data=printOutDict)

        listDF.append(DF)

    except:
        print(tb_output_folder)
    
   
#%%
plt.figure(figsize=(16, 6))
# sns.set_theme(style="whitegrid")
window = 100
regex = re.compile(r'(?<=Training/).*(?=/bs)')
for DF in listDF:
    na = DF['Name'].unique()
    y_m = DF.rolling(window).mean().shift(-window//2)
    y_std = DF.rolling(window).std().shift(-window//2)
    y_lb = y_m - y_std
    y_ub = y_m + y_std
    plt.subplot(1, 2, 1)
    plt.plot(y_m['steps'],y_m['batch_ssim_phase'])
    plt.fill_between(y_m['steps'], y_lb['batch_ssim_phase'], y_ub['batch_ssim_phase'], alpha=0.3)
    plt.legend([regex.search(n).group(0) for n in na])
    plt.show()
    # sns.lineplot(data=DF, x="steps", y="batch_ssim_phase")
    # sns.lineplot(data=mov_avg, x="steps", y="batch_ssim_phase",ci=50)
    # plt.subplot(1, 2, 2)
    # sns.lineplot(data=DF, x="steps", y="batch_loss")
    # sns.lineplot(data=mov_avg, x="steps", y="batch_loss",ci=50)
# plt.plot(xdata, ydata, 'or')
# plt.plot(xfit, yfit, '-', color='gray')



#%%
df = pd.concat(listDF)
df.to_csv('Output.csv')   
# %%
