from matplotlib import pyplot as plt


def plot(current, estimated, next_):
    fig = plt.figure()

    ax = fig.add_subplot(131)
    ax.imshow(current)
    ax.set_title("current image")

    ax = fig.add_subplot(132)
    ax.imshow(estimated)
    ax.set_title("estimated current image from next image")

    ax = fig.add_subplot(133)
    ax.imshow(next_)
    ax.set_title("next image")

    plt.show()
