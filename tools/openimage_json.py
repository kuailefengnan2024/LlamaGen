"""
该脚本用于处理OpenImages数据集，生成用于模型训练的图像路径JSON文件。
在LlamaGen项目中，此脚本主要负责：
1. 扫描OpenImages数据集文件夹，收集所有图像文件的路径
2. 验证每个图像文件的完整性和可读性
3. 利用多进程并行处理，提高大规模数据集的处理效率
4. 将有效的图像路径保存为JSON格式，便于训练过程中加载

此脚本是数据预处理流程的重要组成部分，确保模型训练中使用的图像数据是有效和可访问的。
"""

import argparse
import os
import json
from PIL import Image
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')


def check_image(image_path):
    try:
        Image.open(image_path)
        return True
    except Exception as e:
        print(f"Error details: {str(e)}")
        return False


def check_image_path(image_info):
    data_path, image_path_list = image_info  # Unpack the info
    valid_image_paths = []
    for image_path in image_path_list:
        if check_image(os.path.join(data_path, image_path)):
            valid_image_paths.append(image_path)
    return valid_image_paths


def load_image_path(image_info):
    folder_name, data_path, image_extensions = image_info  # Unpack the info
    print(folder_name)

    folder_path = os.path.join(data_path, folder_name)
    local_image_paths = []
    for image_path in os.listdir(folder_path):
        _, file_extension = os.path.splitext(image_path)
        if file_extension.lower() in image_extensions:
            image_path_full = os.path.join(folder_name, image_path)
            local_image_paths.append(image_path_full)
    return local_image_paths



def main(args):
    data_path = args.data_path
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    num_processes = 47
    work_list = [('openimages_{:0>4}'.format(idx), data_path, image_extensions) for idx in range(1, 48)]
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(load_image_path, work_list)
    image_paths = [image_path for sublist in results for image_path in sublist]
    print('image_paths is loaded')


    num_processes = max(mp.cpu_count() // 2, 4)
    unit = len(image_paths) // num_processes
    work_list = [(data_path, image_paths[idx*unit:(idx+1)*unit]) for idx in range(num_processes)]
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(check_image_path, work_list)
    valid_image_paths = [image_path for sublist in results for image_path in sublist]
    print('image_paths is checked')


    output_json_file_path = os.path.join(data_path, 'image_paths.json')
    with open(output_json_file_path, 'w') as outfile:
        json.dump(valid_image_paths, outfile, indent=4)
    print(f"Image paths have been saved to {output_json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()
    main(args)