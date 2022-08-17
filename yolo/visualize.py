from numpy import ndarray, zeros
import matplotlib.pyplot as plt
import matplotlib.patches as pc


def visualize_cloud(snippet: ndarray)->None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    col = [0, 0, 0, 1]
    #plot point cloud
    ax.plot(
        snippet['y_cc'], #y_cc
        snippet['x_cc'], #x_cc
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=2
    )
    ax.invert_xaxis()
    plt.show()
    return

def visualize_pixel_cloud(snippet: ndarray)->None:
    '''
    For testing box2clusters
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    col = [0, 0, 0, 1]
    #plot point cloud
    ax.plot(
        snippet['x'], #y_cc
        snippet['y'], #x_cc
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=1
    )
    ax.invert_yaxis() # because image coordinate is different from ax.plot()
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return

# rewrite with OO
def visualize_pixel_cloud_boxes(snippet: ndarray, boxes:ndarray)->None:
    '''
    For testing box2clusters
    '''
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1, 1, 1)
    col = [0, 0, 0, 1]
    #plot point cloud
    ax.plot(
        snippet['x'], #y_cc
        snippet['y'], #x_cc
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=1
    )
    # plot AABB
    for box in boxes:
        x1, y1, x2, y2 = box # w is always on x axis
        up_left_horizon = x1
        up_left_vertical = y1
        h = abs(x1 - x2)
        w = abs(y1 - y2)
        rect = pc.Rectangle((up_left_horizon, up_left_vertical), h, w,
                            angle=0, fill=False, edgecolor = 'red',linewidth=2)
        ax.add_patch(rect)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return


def visualize_cluster_boxes(snippet: ndarray, clusters, boxes:ndarray)->None:
    '''
    For testing box2clusters
    '''
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1, 1, 1)
    col = [0, 0, 0, 1]
    #plot point cloud
    ax.plot(
        snippet['x'], #y_cc
        snippet['y'], #x_cc
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=1
    )
    # plot clusters
    num = clusters.num
    for i in range(num):
        cluster = clusters.position_list[i]
        if cluster is not None:
            ax.plot(
                cluster['x'], #y_cc
                cluster['y'], #x_cc
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="g",
                markersize=2
            )
    # plot AABB
    for box in boxes:
        x1, y1, x2, y2 = box # w is always on x axis
        up_left_horizon = x1
        up_left_vertical = y1
        h = abs(x1 - x2)
        w = abs(y1 - y2)
        rect = pc.Rectangle((up_left_horizon, up_left_vertical), h, w,
                            angle=0, fill=False, edgecolor = 'red',linewidth=1)
        ax.add_patch(rect)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return

