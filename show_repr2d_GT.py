import os.path
import matplotlib.pyplot as plt
from PIL import Image
from src import misc
import pandas as pd
import numpy as np
from src import utils, app
from glob import glob
markers = misc.get_markers()
marker_labels = [f"{i}: {name}" for i, name in enumerate(markers)]
links = misc.get_skeleton()

data_path = r'D:\python_code\3Dproject\AcinoSet\PigSet\SOWOA\dataset'
root_labeled = r'E:\Shirley_Code\python_code\Shirely_remote\Multi-view-pig-joints-shirley-2023-06-26\pose-with-likelihood' #
pig = 'middle1'
behavior = 'Walk10'
image_path = os.path.join(data_path, pig, behavior, 'images')
colors = list(range(len(markers)))
dlc_points_fpaths = sorted(glob(os.path.join(os.path.join(root_labeled, behavior + '_' + pig), '*.h5')))
points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
points_2d_df = points_2d_df.drop(['likelihood', 'frame_index'], axis=1)
excel_path = os.path.join(data_path, pig, behavior, 'GRU-SRE4-', 'points_2d_repr.csv')

if not os.path.exists(excel_path):
    pass
else:
    points_2d_repr = pd.read_csv(excel_path)
    points_2d_repr = points_2d_repr.drop(
        ['likelihood', 'frame_index', 'point_index', 'x', 'y', 'z'], axis=1)
    points_df = points_2d_df.merge(points_2d_repr, how='inner', on=['frame', 'camera', 'marker'])
    frames_index = np.unique(points_df['frame'].values).tolist()
    for cam in [0, 1, 2, 3]:
        data = points_df[points_df['camera'] == cam]
        output_path = os.path.join(data_path, pig, behavior, 'Repr-2dimg-val', 'cam' + str(cam))
        os.makedirs(output_path, exist_ok=True)
        for frame_index in frames_index:
            image = Image.open(os.path.join(image_path, 'cam' + str(cam), 'Image'+str(frame_index)+'.jpg'))
            image = np.array(image)
            data_frame = data[data["frame"] == frame_index]
            dlc_df = pd.DataFrame(columns=['marker', 'x', 'y', 'x1', 'y1', 'x2', 'y2'])
            dlc_df['marker'] = markers
            for name in data_frame['marker']:
                index_3d = dlc_df[dlc_df['marker'] == name].index
                dlc_df.loc[index_3d, 'x'] = data_frame[data_frame["marker"] == name][["x"]].values  # 真实值
                dlc_df.loc[index_3d, 'y'] = data_frame[data_frame["marker"] == name][["y"]].values

                dlc_df.loc[index_3d, 'x1'] = data_frame[data_frame["marker"] == name][["x_cam"]].values  # 2D预测值
                dlc_df.loc[index_3d, 'y1'] = data_frame[data_frame["marker"] == name][["y_cam"]].values

                dlc_df.loc[index_3d, 'x2'] = data_frame[data_frame["marker"] == name][["repr_x"]].values  # 重投影值
                dlc_df.loc[index_3d, 'y2'] = data_frame[data_frame["marker"] == name][["repr_y"]].values

            X = dlc_df['x'].values
            Y = dlc_df['y'].values
            X1 = dlc_df['x1'].values
            Y1 = dlc_df['y1'].values
            X2 = dlc_df['x2'].values
            Y2 = dlc_df['y2'].values

            # 创建图形
            fig, ax = plt.subplots(figsize=(19.2, 10.8))
            # 添加标题和标签
            ax.set_title('Cam {} 2d joints for frame {}'.format(cam, frame_index))
            ax.set_xlabel('X axis', fontsize=10, color='black', fontweight='bold', loc='center', labelpad=6)
            ax.set_ylabel('Y axis', fontsize=10, color='black', fontweight='bold', loc='center', labelpad=6)
            ax.grid(False)

            for link in links:
                index_0 = dlc_df[dlc_df['marker'] == link[0]].index
                index_1 = dlc_df[dlc_df['marker'] == link[1]].index
                x_ = [X[index_0][0], X[index_1][0]]
                y_ = [Y[index_0][0], Y[index_1][0]]
                plt.plot(x_, y_, linewidth=1, color='#449945')
                x_1 = [X1[index_0][0], X1[index_1][0]]
                y_1 = [Y1[index_0][0], Y1[index_1][0]]
                # plt.plot(x_1, y_1, linewidth=1, color='#ea7827')
                x_2 = [X2[index_0][0], X2[index_1][0]]
                y_2 = [Y2[index_0][0], Y2[index_1][0]]
                plt.plot(x_2, y_2, linewidth=1, color='#c22f2f')
            # 绘制散点图
            scatter = plt.scatter(X, Y, s=6, c=colors, cmap='viridis')
            # scatter1 = plt.scatter(X1, Y1, s=6, c=colors, cmap='viridis')
            scatter2 = plt.scatter(X2, Y2, s=6, c=colors, cmap='viridis')
            ax.imshow(image, extent=[0, image.shape[1], image.shape[0], 0], origin='upper')
            # 保存图像
            plt.savefig(os.path.join(output_path, 'Cam'+str(cam)+'Image'+str(frame_index) + '.png'), dpi=100)
            print('Saved!')
            # plt.close()
            # 显示图形
            # plt.tight_layout()
            # plt.show()
