from src.utils.preprocess import CropAndExtract
from src.utils.init_path import init_path
import os
# 初始化路径
checkpoint_path = 'checkpoints'
config_path = 'src/config'
os.environ['TORCH_HOME']= checkpoint_path

sadtalker_paths = init_path(checkpoint_path, config_path, 256, False, 'crop')
device = 'cuda'
# 初始化模型
processor = CropAndExtract(sadtalker_paths, device)
# 进行测试
# first_coeff_path, crop_pic_path, crop_info = sadtalker.get_processed_pic('crop',256,'girl.png','nervld_test_firstframe')
input_path = 'girl.png'
save_dir = 'nervld_test_firstframe'

processor.generate(input_path, save_dir)
