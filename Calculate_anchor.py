# import utils.autoanchor as autoAC
#
# # 对数据集重新计算 anchors
# new_anchors = autoAC.kmean_anchors('./data/voc.yaml', 9, 512, 4, 1000, True)
# print(new_anchors)
#
# '''
#     kmean_anchor()函数参数含义：
#         path：包含数据集文件路径等相关信息的 yaml 文件（比如 coco128.yaml）， 或者 数据集张量（yolov5 自动计算锚定框时就是用的这种方式，先把数据集标签信息读取再处理）
#         n：锚定框的数量，即有几组；默认值是 9
#         img_size：图像尺寸。计算数据集样本标签框的宽高比时，是需要缩放到 img_size 大小后再计算的；默认值是 640
#         thr：数据集中标注框宽高比最大阈值，默认是使用 超参文件 hyp.scratch.yaml 中的 “anchor_t” 参数值；默认值是 4.0；自动计算时，会自动根据你所使用的数据集，来计算合适的阈值。
#         gen：kmean 聚类算法迭代次数，默认值是 1000
#         verbose：是否打印输出所有计算结果，默认值是 true
# '''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('wh.csv')  # 打开csv文件
X = np.array(df)
n_clusters=9
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
centroid=cluster.cluster_centers_
print(centroid)
