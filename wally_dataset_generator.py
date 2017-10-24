import math
import os
import random
import shutil
from PIL import Image, ImageDraw


def build_training_set():
    output_wally_directory = os.path.join(
        os.path.dirname(__file__), 'training/wally')
    if os.path.exists(output_wally_directory):
        shutil.rmtree(output_wally_directory)
    os.makedirs(output_wally_directory)

    output_not_directory = os.path.join(
        os.path.dirname(__file__), 'training/not')
    if os.path.exists(output_not_directory):
        shutil.rmtree(output_not_directory)
    os.makedirs(output_not_directory)

    # Horrible array of training sources. Sources are the original scenes
    # containing Wally. Each scene has the x1, y1, x2, y2 cooridnate of a
    # bounding box that surrounds Wally. This is to ensure we don't accidentally
    # add a Wally to the not-Wally class.
    wally_training_sources = [{
        'path': 'src/scene/waldo10.jpg',
        'x1': 1476,
        'y1': 274,
        'x2': 1513,
        'y2': 310
    }, {
        'path': 'src/scene/waldo9.jpg',
        'x1': 1319,
        'y1': 1365,
        'x2': 1356,
        'y2': 1403
    }, {
        'path': 'src/scene/waldo8.jpg',
        'x1': 1151,
        'y1': 286,
        'x2': 1196,
        'y2': 332
    }, {
        'path': 'src/scene/waldo4.jpg',
        'x1': 1383,
        'y1': 459,
        'x2': 1429,
        'y2': 504
    }]

    # Horrible array of wally head sources. Wally heads are cutouts of Wally's
    # head from the training sources. They are RGBA images that will be used
    # to generate the Wally class of training data.
    wally_head_sources = [{
        'path': 'src/head/waldo10.png'
    }, {
        'path': 'src/head/waldo9.png'
    }, {
        'path': 'src/head/waldo8.png'
    }, {
        'path': 'src/head/waldo4.png'
    }]

    # Populate source and head images.
    for wts in wally_training_sources:
        wts['image'] = Image.open(os.path.join(os.path.dirname(__file__), wts['path']))
        wally_rect = (wts['x1'], wts['y1'], wts['x2'], wts['y2'])
        draw=ImageDraw.Draw(wts['image'])
        draw.rectangle(wally_rect, fill=(0,)*3)
    for whs in wally_head_sources:
        whs['image'] = Image.open(os.path.join(os.path.dirname(__file__), whs['path']))

    sizes = [50, 100, 150]
    rotations = [i for i in range(-20, 20+1, 4)]
    scales = [float(i)/100 for i in range(90, 100, 1)]

    # Iterate over all the training images. Cut images into squares at several
    # zoom levels. For each square create a wally and not-wally sameple image.
    for wts_idx, wts in enumerate(wally_training_sources):
        img = wts['image']

        for size in sizes:
            for j in range(int(math.floor(img.size[1] / size) - 1)):
                for i in range(int(math.floor(img.size[0] / size) - 1)):
                    image_rect = (i * size, j * size, (i + 1) * size, (j + 1) * size)

                    # Create a background cutout and save.
                    filename = '{0}-{1}-{2}-{3}.jpg' \
                        .format(wts_idx, size, i, j)
                    cropped = img.crop(image_rect)
                    cropped = cropped.convert('RGB')
                    cropped.save(os.path.join(output_not_directory, filename))
                    bg_w, bg_h = cropped.size

                    # Superimpose a random wally head with random scale and
                    # rotation over the background image and save.
                    walley_head_idx = random.randrange(len(wally_head_sources))
                    wally_head = wally_head_sources[walley_head_idx]['image'].copy()

                    rotation_idx = random.randrange(len(rotations))
                    rotation = rotations[rotation_idx]
                    scale_idx = random.randrange(len(scales))
                    scale = scales[scale_idx]

                    wally_head = wally_head.rotate(rotation, expand=1)
                    wally_head = wally_head.resize((int(scale * bg_w), int(scale * bg_h)))
                    wy_w, wy_h = wally_head.size

                    offset = (
                        random.randrange(bg_w - wy_w),
                        random.randrange(bg_h - wy_h))

                    combined = Image.new('RGB', (bg_w, bg_h), (0,)*3)
                    combined.paste(cropped)
                    combined.paste(wally_head, offset, wally_head)
                    combined.save(os.path.join(output_wally_directory, filename))


if __name__ == '__main__':
    build_training_set()
