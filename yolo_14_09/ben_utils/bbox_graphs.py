import pandas as pd
import matplotlib.pyplot as plt

names = {0: 'Person', 1: 'Person on board a vessel', 2: 'Swimmer', 3: 'Sail boat', 4: 'Floating object',  5: 'Dvora',
         6: 'Zeara', 7: 'PWC', 8: 'Merchant Ship', 9: 'Inflatable Boat', 10: 'Vessel'}
df = pd.read_csv(r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\all_together.csv')
df['area'] = df['width'] * df['height']
for i in range(11):
    type_df = df[df['Label'] == i]
    for name in ['width', 'height' , 'area']:
        plt.xlim(0, max(type_df[name].values))
        plt.xlabel('%s val(Pixels)' % name)
        plt.ylabel('amount')
        plt.title('%s graph for %s'% (name,names[i]))
        type_df[name].hist(bins=100)
        plt.savefig(r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\graphs\%s_graph_for_%s.png' % (name, names[i]))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
ax = df.hist(bins=10)

fig.savefig(r'Z:\deepsort_yolov5\deepsort_ben\runs\track\mul_4_custom_weights_all_data_deep_crop_226\object_sizes\graph.png')
