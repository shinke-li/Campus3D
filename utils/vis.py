import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_color_map(label_num, cmap_name='terrain', normalzation=False):
    cmap = cm.get_cmap(cmap_name, label_num)
    cmap_array = np.array([cmap(i)[:3] for i in range(label_num)])
    return cmap_array * 255.0 if normalzation else cmap_array

def get_color_legend(file_name, label_num, cmap_name='terrain', square_size=50):
    import cv2
    canvas = (np.ones(shape=(label_num * 2 * square_size, square_size, 3))
              * 255).astype(np.uint8)
    colors = get_color_map(label_num, cmap_name=cmap_name, normalzation=True).astype(np.uint8)
    for i in range(colors.shape[0]):
        start_i = 2 * i * int(square_size)
        end_i = (2 * i + 1) * int(square_size)
        canvas[start_i:end_i,:,:] = np.tile(colors[i], reps=(square_size, square_size, 1))
    cv2.imwrite(file_name, canvas[:, :, ::-1])

def vis_matplot(points, colors, size=0.01):
    ax = plt.axes(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=size)
    plt.show()

if __name__ == "__main__":
    get_color_legend('label3.jpg', 6, 'jet')