import numpy as np
import matplotlib.pyplot as plt
from WrighterClass import Writer
from ReaderClass import Reader


# TODO wright out l1,l2, l inf errors  take in command line args and make it work and adios part
def build_progressive_array(data, min_size=8):
    assert data.shape[0] == data.shape[1], "Input must be square"
    assert (data.shape[0] & (data.shape[0] - 1)) == 0, "Size must be power of two"
    N = data.shape[0]
    sizes = []
    s = N

    while s >= min_size:
        sizes.append(s)
        s //= 2
    sizes = sizes[::-1]  # From coarse to fine
    output_chunks = []
    for i, size in enumerate(sizes):
        if i == 0:
            # Coarsest level: take every N/size-th point (even-even)
            step = N // size
            down = data[::step, ::step]
            output_chunks.append(down.flatten())
        else:
            prev_size = sizes[i - 1]
            step = N // size
            # Create a mask for points present in the previous level
            mask = np.zeros((size, size), dtype=bool)
            mask[::2, ::2] = True  # even-even were already included
            # Extract new points (odd-even, even-odd, odd-odd)
            new_points = []
            full = data[::step, ::step]  # current level from source
            # Odd rows, even cols
            new_points.append(full[1::2, ::2].flatten())
            # Even rows, odd cols
            new_points.append(full[::2, 1::2].flatten())
            # Odd rows, odd cols
            new_points.append(full[1::2, 1::2].flatten())
            output_chunks.append(np.concatenate(new_points))
    return np.concatenate(output_chunks)


def build_progressive_array0(data):
    assert data.shape[0] == data.shape[1], "Input must be square"
    assert (data.shape[0] & (data.shape[0] - 1)) == 0, "Size must be power of two"
    N = data.shape[0]
    levels = int(np.log2(N)) - 3  # How many levels until 8x8
    pyramids = []
    # Build downsampled levels
    for i in range(levels + 1):
        step = 2 ** (levels - i)
        down = data[::step, ::step]
        print("level: ", i, (step, step))
        print(" datashape: ", down.shape)
        pyramids.append(down)

    # Construct output

    output_chunks = []
    for i in range(len(pyramids)):
        current = pyramids[i]

        if i == 0:

            # First (coarsest) level
            output_chunks.append(current.flatten())

        else:

            prev = pyramids[i - 1]
            up = current

            # Extract new points compared to previous level

            new_points = []
            # Points where only row is new
            new_points.append(up[1::2, ::2].flatten())
            # Points where only column is new
            new_points.append(up[::2, 1::2].flatten())
            # Points where both row and column are new
            new_points.append(up[1::2, 1::2].flatten())
            output_chunks.append(np.concatenate(new_points))

    return np.concatenate(output_chunks)


def extract_level(
    progressive_array, target_size, min_size=8, full_resolution_shape=(128, 128)
):
    full_size = full_resolution_shape[0]
    assert (
        full_resolution_shape[0] == full_resolution_shape[1]
    ), "Only square arrays supported"

    assert (full_size & (full_size - 1)) == 0, "Full resolution must be power of two"
    assert (target_size & (target_size - 1)) == 0, "Target size must be power of two"
    assert target_size <= full_size, "Target size must be ≤ full resolution"

    # Generate all powers-of-two sizes down to 8

    sizes = []
    s = full_size

    while s >= min_size:

        sizes.append(s)

        s //= 2

    sizes = sizes[::-1]  # From coarse to fine

    index = sizes.index(target_size)

    # Compute number of values in each level

    chunks = []

    for i, size in enumerate(sizes):

        if i == 0:
            chunks.append(size * size)

        else:
            prev = sizes[i - 1]
            added = size * size - prev * prev
            chunks.append(added)

    # Compute where this level starts in the flattened array

    start = sum(chunks[:index])
    end = start + chunks[index]
    flat = progressive_array[start:end]
    if index == 0:

        return flat.reshape((target_size, target_size))

    # Reconstruct from previous level

    prev_size = sizes[index - 1]

    prev_data = extract_level(
        progressive_array,
        target_size=prev_size,
        min_size=min_size,
        full_resolution_shape=full_resolution_shape,
    )
    out = np.zeros((target_size, target_size))
    out[::2, ::2] = prev_data
    # Fill in new values
    new_data = flat
    i = 0
    half = target_size // 2
    shape = (half, half)
    out[1::2, ::2] = new_data[i : i + half * half].reshape(shape)
    i += half * half
    out[::2, 1::2] = new_data[i : i + half * half].reshape(shape)
    i += half * half
    out[1::2, 1::2] = new_data[i : i + half * half].reshape(shape)

    return out


def plot_level(
    progressive_array, target_size, min_size=8, full_resolution_shape=(128, 128)
):

    level_data = extract_level(
        progressive_array,
        target_size,
        min_size=min_size,
        full_resolution_shape=full_resolution_shape,
    )

    plt.imshow(level_data, cmap="viridis")
    plt.title(f"{target_size}x{target_size} resolution")
    plt.colorbar()
    plt.show()


def create_data(size=128):
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    data = np.sin(r) * np.cos(xx / 4) * np.cos(yy / 4)
    return data


# data = np.random.rand(128, 128)

min_pow = 2
max_pow = 8

# progressive = build_progressive_array(data)


levels = [2**i for i in range(min_pow, max_pow + 1)]
min_val = 2**min_pow
max_val = 2**max_pow

# data = np.random.rand(max_val, max_val)
data = create_data(max_val)
progressive = build_progressive_array(data, min_size=min_val)
print(data.shape)
print(levels)

for lvl in levels:
    print("lvl=", lvl)
    plot_level(progressive, lvl, min_size=min_val, full_resolution_shape=data.shape)
