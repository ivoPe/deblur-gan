import numpy as np
from PIL import Image
import click
import os
from model import generator_model
from utils import load_images, deprocess_image

BASE_DIR = 'weights/'
IMAGES_FOLDER = '/notebooks/data/images'
TEST_FOLDER = os.path.join(IMAGES_FOLDER, 'test')
SAVE_MODEL_PATH = '/notebooks/deblur-gan/weights/89/generator_3_659.h5'
# SAVE_MODEL_PATH = 'generator.h5'


def test(batch_size):
    data = load_images(TEST_FOLDER, batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights(SAVE_MODEL_PATH)
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))


@click.command()
@click.option('--batch_size', default=4, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
