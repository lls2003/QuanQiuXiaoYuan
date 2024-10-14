import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='电动汽车调度 DQN 模型')

    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='./dataset/', help='数据集路径')
    parser.add_argument('--n_cars', type=int, default=50, help='电动汽车数量')

    # 模型相关参数
    parser.add_argument('--state_size_per_car', type=int, default=4, help='每辆车的状态大小')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')

    # 训练相关参数
    parser.add_argument('--episodes', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epsilon', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='探索率衰减')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='最小探索率')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')

    # 其他参数
    parser.add_argument('--save_model', type=str, default='best_dqn_model.pth', help='保存模型的文件名')
    parser.add_argument('--save_rewards', type=str, default='rewards.csv', help='保存奖励的文件名')
    parser.add_argument('--save_results', type=str, default='results.xlsx', help='保存结果的文件名')

    return parser.parse_args()


# 如果需要默认参数，可以定义一个函数来获取
def get_default_args():
    return parse_args([])  # 传入空列表，使用所有默认值


# 如果需要将参数保存为字典形式，可以添加这个函数
def args_to_dict(args):
    return vars(args)


# 如果需要从字典中加载参数，可以添加这个函数
def dict_to_args(args_dict):
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args