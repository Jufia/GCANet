import os
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split


from params import args
import logging

logging.basicConfig(
    filename='./checkpoint/log/' + args.log_name,
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)

class DIRGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def read_mat_files_from_dir(directory):
    """
    读取指定目录下所有的.mat文件
    
    Args:
        directory: .mat文件所在的目录路径
        
    Returns:
        dict: 包含所有.mat文件数据的字典
    """
    data_dict = {}
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            file_path = os.path.join(directory, filename)
            try:
                # 使用scipy.io.loadmat读取.mat文件
                mat_data = loadmat(file_path)
                
                # 移除mat文件中的系统变量
                mat_data = {k: v for k, v in mat_data.items() 
                           if not k.startswith('__')}
                
                # 将数据添加到字典中
                data_dict[filename] = mat_data
                
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
                

    return data_dict

class DIRGDataProcessor:
    def __init__(self, data_path, window_size=512, batch_size=32):
        """
        初始化DIRG数据处理器
        
        Args:
            data_path: 数据文件路径
            window_size: 滑动窗口大小
            batch_size: 批次大小
        """
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = args.batch_size
        self.scaler = StandardScaler()
        
    def _extract_label(self, filename):
        """从文件名中提取标签"""
        # 处理C0A到C06的命名格式
        if filename.startswith('C0'):
            if filename[2] == 'A':
                return 0
            else:
                return int(filename[2])
        return None
    
    def process_data(self):
        """处理数据并返回数据集"""
        # 读取数据
        data_dict = read_mat_files_from_dir(self.data_path)
        
        # 存储所有数据和标签
        all_windows = []
        all_labels = []
        
        # 处理每个文件的数据
        for filename, mat_data in data_dict.items():
            logging.info(f"处理文件: {filename}")
            label = self._extract_label(filename)
            if label is None:
                continue
                
            # 获取数据矩阵
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray):
                    # 创建不重叠的窗口
                    n_windows = value.shape[0] // self.window_size
                    for i in range(n_windows):
                        start_idx = i * self.window_size
                        end_idx = start_idx + self.window_size
                        window = value[start_idx:end_idx]
                        all_windows.append(window.transpose(1, 0))
                        all_labels.append(label)
        
        if not all_windows:
            raise ValueError("没有找到有效的数据")
            
        # 转换为numpy数组
        all_windows = np.array(all_windows)
        
        # 数据标准化
        n_samples, n_timesteps, n_features = all_windows.shape
        all_windows = all_windows.reshape(-1, n_features)
        all_windows = self.scaler.fit_transform(all_windows)
        all_windows = all_windows.reshape(n_samples, n_timesteps, n_features)

        # 保存处理后的数据
        save_dir = os.path.join(self.data_path, 'processed')
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数据
        np.save(os.path.join(save_dir, '512x.npy'), all_windows)
        np.save(os.path.join(save_dir, '512y.npy'), np.array(all_labels))
        
        return all_windows, np.array(all_labels)
    
    def create_data_loaders(self, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3):
        """创建训练、验证和测试数据加载器"""
        # 处理数据
        # x, y = self.process_data()
        
        # 加载.npy数据
        x = np.load('./data/dirg/processed/512x.npy')
        y = np.load('./data/dirg/processed/512y.npy')
               
        # 创建数据集
        dataset = DIRGDataset(x, y)
        
        # 计算数据集大小
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # 随机分割数据集
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


def Loador():
    # 设置数据路径
    data_path = args.path
    processor = DIRGDataProcessor(
        data_path=data_path,
        window_size=args.windows,
        batch_size=args.batch_size
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_data_loaders()
    # 打印数据集信息
    logging.info(f"训练集批次数量: {len(train_loader)}")
    logging.info(f"验证集批次数量: {len(val_loader)}")
    logging.info(f"测试集批次数量: {len(test_loader)}")
    
    # 打印一个批次的数据形状
    for batch_data, batch_labels in train_loader:
        logging.info(f"数据形状: {batch_data.shape}")
        logging.info(f"标签形状: {batch_labels.shape}")
        break

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = Loador()
    