from playground.auxiliary_models import GroundingDINOSAMSegmentor
from PIL import Image

segmentor = GroundingDINOSAMSegmentor()

img = Image.open('test.jpg')

keys_list = ['man', 'woman']

for key in keys_list:
    man_mask = segmentor(img, [key])
    print(man_mask)

    Image.fromarray(man_mask).save(f'mask_{key}.png')
