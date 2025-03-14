import os
import sys
import ctypes
import tensorflow as tf
import numpy as np

# 添加DLL搜索路径
def add_dll_directory():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义所有可能的DLL路径
    dll_paths = [
        os.path.join(script_dir, 'cuda_dlls'),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin',
        os.environ.get('CUDA_PATH', ''),
    ]
    
    # 添加所有存在的路径到DLL搜索路径
    for path in dll_paths:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
                print(f"已添加DLL目录: {path}")
            except Exception as e:
                print(f"添加DLL目录失败 {path}: {str(e)}")

# 在导入TensorFlow之前添加DLL目录
add_dll_directory()

def check_gpu_details():
    print("\n" + "="*50)
    print("系统信息：")
    print(f"Python 版本: {sys.version}")
    print(f"TensorFlow 版本: {tf.__version__}")
    
    print("\n" + "="*50)
    print("CUDA环境变量：")
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    print(f"CUDA_PATH: {cuda_path}")
    
    print("\n" + "="*50)
    print("GPU 设备信息：")
    physical_devices = tf.config.list_physical_devices()
    print("\n所有物理设备：")
    for device in physical_devices:
        print(f"  {device.device_type}: {device.name}")
    
    print("\nGPU 设备：")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"  发现 GPU: {gpu}")
            try:
                # 获取GPU详细信息
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"  GPU 详细信息: {gpu_details}")
            except:
                print("  无法获取GPU详细信息")
    else:
        print("  未找到可用的 GPU 设备")
    
    print("\n" + "="*50)
    print("TensorFlow 设备策略：")
    print(tf.config.get_visible_devices())
    
    print("\n" + "="*50)
    print("CUDA 可用性测试：")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("矩阵乘法测试成功：")
            print(c)
    except Exception as e:
        print(f"GPU 测试失败: {str(e)}")

    print("\n" + "="*50)
    print("cuDNN 版本：")
    try:
        print(f"cuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    except:
        print("无法获取 cuDNN 版本信息")
    
    print("\n" + "="*50)
    print("TensorFlow 编译信息：")
    try:
        build_info = tf.sysconfig.get_build_info()
        for key, value in build_info.items():
            print(f"  {key}: {value}")
    except:
        print("无法获取 TensorFlow 编译信息")

if __name__ == "__main__":
    check_gpu_details()



