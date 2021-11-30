import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def run(count: int = None):

    dpi = mpl.rcParams['figure.dpi']

    figsize = (64/dpi, 64/dpi)

    fig = plt.figure(figsize=figsize)
    plt.axis('off')

    ax = fig.add_subplot(111)
    # ax.set_xlim(-12, 12)
    ax.set_xlim(0, 64)
    # ax.set_ylim(-12, 12)
    ax.set_ylim(0, 64)
    ax.axis('off')

    n_patches = 3

    exp = ['smooth', 'noisy']
    # exp = ['smooth', 'changing', 'noisy']
    # exp = ['smooth', 'no_kpts']

    # x = np.random.randint(-9, 9, size=(n_patches, )).astype(np.float)
    x = np.random.randint(5, 60, size=(n_patches,)).astype(np.float)
    # y = np.random.randint(-9, 9, size=(n_patches, )).astype(np.float)
    y = np.random.randint(5, 60, size=(n_patches,)).astype(np.float)
    kpt_x = np.copy(x)
    kpt_y = np.copy(y)
    kpt_patch_map = {}
    for n in range(n_patches):
        kpt_patch_map[n] = n

    d = 0.1
    rect_size = 6

    if 'noisy' in exp:
        noise_range = 1.0
    else:
        noise_range = 0.0

    if 'smooth' in exp:
        direction = np.random.randint(low=1, high=9, size=(n_patches,))
    else:
        direction = np.random.randint(low=1, high=5, size=(n_patches, ))

    exp_name = ""
    for e in exp:
        exp_name += f'{e}_'

    patch_list = []
    scatter_obj_list = []
    line = None
    color_list = ['blue', 'green', 'red']
    hatch_list = ['//', '--', '**']

    l = ax.plot(x, y, color='gray', linewidth=2, alpha=0.6, zorder=0)[0]

    for n in range(n_patches):
        patch_list.append(patches.Rectangle(xy=(x[n], y[n]),
                                            width=rect_size, height=rect_size,
                                            # color=color_list[n],
                                            color='gray',
                                            hatch=hatch_list[n],
                                            alpha=0.6,
                                            zorder=5))

    def init():
        for n in range(n_patches):
            ax.add_patch(patch_list[n])
            if 'no_kpts' not in exp:
                scatter_ob = ax.scatter(kpt_x[kpt_patch_map[n]] + int(rect_size/2), kpt_y[kpt_patch_map[n]] + int(rect_size/2),
                                        #color='black')
                                        color=color_list[n],
                                        s=2.0,
                                        zorder=10)
                scatter_obj_list.append(scatter_ob)
                if count is not None:
                    with open(f'{exp_name}{count}_{n}.txt', 'w') as coords_file:
                        pass
                else:
                    with open(f'{exp_name}{n}.txt', 'w') as coords_file:
                        pass
        return patch_list, scatter_obj_list

    def create_random_path(x, y, direction):
        if random.uniform(0, 1) > 0.8:
            direction = random.randint(1, 5)

        if direction == 1:
            return x, y+d, direction
        if direction == 2:
            return x+d, y, direction
        if direction == 3:
            return x, y-d, direction
        if direction == 4:
            return x-d, y, direction
        return x, y, direction

    def create_smooth_random_path(x, y, direction):
        if random.uniform(0, 1) > 0.9:
            # direction = np.clip(random.randint(-1, 2) + direction, a_min=1, a_max=8)
            direction = random.randint(max(direction - 1, 0), min(direction + 1, 8))

        if direction == 1:
            return x, y+d, direction
        if direction == 2:
            return x+d, y+d, direction
        if direction == 3:
            return x+d, y, direction
        if direction == 4:
            return x+d, y-d, direction
        if direction == 5:
            return x, y-d, direction
        if direction == 6:
            return x-d, y-d, direction
        if direction == 7:
            return x-d, y, direction
        if direction == 8:
            return x-d, y+d, direction
        return x, y, direction

    img_list = []
    kpt_coords_list = []

    max_frames = 1000  # 1000

    t_buff_kpt_x = np.copy(kpt_x)
    t_buff_kpt_y = np.copy(kpt_y)
    global offset_kpt_x
    global offset_kpt_y
    offset_kpt_x, offset_kpt_y = None, None

    def animate(i):

        print(f'\r##### Generating frames: {i} | {max_frames}', end="")

        l.set_data(x + int(rect_size/2), y + int(rect_size/2))

        #
        #   Swap positions in 'changing' case
        #

        if 'changing' in exp and i % 500 == 0 and i > 0:
            print(kpt_patch_map)
            rand_a = np.random.randint(low=0, high=n_patches)
            while True:
                rand_b = np.random.randint(low=0, high=n_patches)
                if not rand_a == rand_b:
                    break
            kpt_patch_map[rand_a] = rand_b
            kpt_patch_map[rand_b] = rand_a
            print(kpt_patch_map)

        for n in range(n_patches):

            #
            #   ('Smooth') direction changes
            #

            if 'smooth' in exp:
                x[n], y[n], direction[n] = create_smooth_random_path(x[n], y[n], direction[n])
            else:
                x[n], y[n], direction[n] = create_random_path(x[n], y[n], direction[n])

            """
            if x[n] > 8:
                x[n] = 8
            if x[n] < -8:
                x[n] = -8
            if y[n] > 8:
                y[n] = 8
            if y[n] < -8:
                y[n] = -8
            """
            if x[n] > 60:
                x[n] = 60
            if x[n] < 5:
                x[n] = 5
            if y[n] > 60:
                y[n] = 60
            if y[n] < 5:
                y[n] = 5

            patch_list[n].set_xy((x[n], y[n]))


            #
            #   'Noisy' keypoint positions
            #

            if 'noisy' in exp:
                global offset_kpt_x
                global offset_kpt_y
                if np.random.rand() > 0.8 or (offset_kpt_x is None and offset_kpt_y is None):
                    offset_kpt_x, offset_kpt_y = np.random.random() * noise_range, np.random.random() * noise_range
                    kpt_x[n] = x[kpt_patch_map[n]] + offset_kpt_x
                    kpt_y[n] = y[kpt_patch_map[n]] + offset_kpt_y
            else:
                kpt_x[n] = x[kpt_patch_map[n]]
                kpt_y[n] = y[kpt_patch_map[n]]

            #
            #   Write / read from position buffer in 'laggy' case
            #

            if 'laggy' not in exp:

                if 'no_kpts' not in exp:

                    scatter_obj_list[n].set_offsets((kpt_x[n] + int(rect_size/2),
                                                     kpt_y[n] + int(rect_size/2)))

                if count is not None:
                    with open(f'{exp_name}{count}_{n}.txt', 'a') as coords_file:
                        coords_file.write(f'{kpt_x[n]};\t{kpt_y[n]};\t1.0\n')
                else:
                    with open(f'{exp_name}{n}.txt', 'a') as coords_file:
                        coords_file.write(f'{kpt_x[n]};\t{kpt_y[n]};\t1.0\n')

            else:

                if i % 20 == 0:
                    global t_buff_kpt_x
                    global t_buff_kpt_y
                    t_buff_kpt_x = np.copy(kpt_x)
                    t_buff_kpt_y = np.copy(kpt_y)

                if 'no_kpts' not in exp:

                    scatter_obj_list[n].set_offsets((t_buff_kpt_x[n] + int(rect_size / 2),
                                                     t_buff_kpt_y[n] + int(rect_size / 2)))
                if count is not None:
                    with open(f'{exp_name}{count}_{n}.txt', 'a') as coords_file:
                        coords_file.write(f'{t_buff_kpt_x[n]};\t{t_buff_kpt_y[n]};\t1.0\n')
                else:
                    with open(f'{exp_name}{n}.txt', 'a') as coords_file:
                        coords_file.write(f'{t_buff_kpt_x[n]};\t{t_buff_kpt_y[n]};\t1.0\n')

        return patch_list, scatter_obj_list, l

    anim = animation.FuncAnimation(fig=fig,
                                   func=animate,
                                   init_func=init,
                                   frames=max_frames,
                                   interval=1,
                                   repeat=False)

    # Save video
    writervideo = animation.FFMpegWriter(fps=60)
    if count is not None:
        anim.save(filename=f'{exp_name[:-1]}_{count}.mp4', writer=writervideo)
    else:
        anim.save(filename=f'{exp_name[:-1]}.mp4', writer=writervideo)
    #plt.show()
    plt.close()


if __name__ == "__main__":

    for i in range(1, 2):
        run(i)





