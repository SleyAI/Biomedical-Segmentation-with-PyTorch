import random


def split_train_val(dataset, val_percent=0.10):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)

    return {'train': dataset[:-n], 'val': dataset[-n:]}


def resize(pilimg, scale=0.5):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    img = pilimg.resize((newW, newH))
    return img


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def normalize(x):
    return x / 255.0
