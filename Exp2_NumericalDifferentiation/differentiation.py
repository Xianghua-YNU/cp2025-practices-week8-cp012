import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图


def f(x):
    """定义测试函数 f(x) = x(x-1)

    参数:
        x (float): 输入值

    返回:
        float: 函数计算结果
    """
    return x * (x - 1)  # 计算并返回 x(x-1)，即 x^2 - x


def forward_diff(f, x, delta):
    """前向差分法计算导数

    参数:
        f (function): 要求导的函数
        x (float): 求导点
        delta (float): 步长

    返回:
        float: 导数的近似值
    """
    return (f(x + delta) - f(x)) / delta  # 使用前向差分公式 (f(x + delta) - f(x)) / delta 计算导数


def central_diff(f, x, delta):
    """中心差分法计算导数

    参数:
        f (function): 要求导的函数
        x (float): 求导点
        delta (float): 步长

    返回:
        float: 导数的近似值
    """
    return (f(x + delta) - f(x - delta)) / (2 * delta)  # 使用中心差分公式 (f(x + delta) - f(x - delta)) / (2 * delta) 计算导数


def analytical_derivative(x):
    """解析导数 f'(x) = 2x - 1

    参数:
        x (float): 求导点

    返回:
        float: 导数的精确值
    """
    return 2 * x - 1  # 根据 f(x) = x(x-1) = x^2 - x，计算解析导数 2x - 1


def calculate_errors(x_point=1.0):
    """计算不同步长下的误差

    参数:
        x_point (float): 求导点，默认为1.0

    返回:
        tuple: (deltas, forward_errors, central_errors)
            deltas: 步长数组
            forward_errors: 前向差分误差数组
            central_errors: 中心差分误差数组
    """
    deltas = np.logspace(-2, -14, num=13, base=10.0)  # 生成步长序列，从 10^-2 到 10^-14，共13个点
    exact = analytical_derivative(x_point)  # 计算在 x_point 处的解析导数值，作为精确值
    forward_errors = []  # 初始化前向差分误差列表
    central_errors = []  # 初始化中心差分误差列表

    for delta in deltas:  # 遍历每个步长
        forward_approx = forward_diff(f, x_point, delta)  # 计算前向差分近似导数值
        central_approx = central_diff(f, x_point, delta)  # 计算中心差分近似导数值
        forward_error = abs(forward_approx - exact) / abs(exact)  # 计算前向差分的相对误差
        central_error = abs(central_approx - exact) / abs(exact)  # 计算中心差分的相对误差
        forward_errors.append(forward_error)  # 将前向差分误差添加到列表
        central_errors.append(central_error)  # 将中心差分误差添加到列表

    return deltas, forward_errors, central_errors  # 返回步长数组和两种方法的误差数组


def plot_errors(deltas, forward_errors, central_errors):
    """绘制误差-步长关系图

    参数:
        deltas (array): 步长数组
        forward_errors (array): 前向差分误差数组
        central_errors (array): 中心差分误差数组
    """
    plt.figure(figsize=(10, 6))  # 创建一个大小为10x6英寸的图形窗口
    plt.loglog(deltas, forward_errors, 'o-', label='Forward Difference')  # 绘制前向差分误差的双对数图，带圆点和实线
    plt.loglog(deltas, central_errors, 's-', label='Central Difference')  # 绘制中心差分误差的双对数图，带方点和实线

    ref_deltas = np.logspace(-2, -14, num=100)  # 生成用于参考线的步长序列，100个点
    ref_forward = ref_deltas  # 前向差分理论误差随步长线性变化，斜率1
    ref_central = ref_deltas ** 2  # 中心差分理论误差随步长平方变化，斜率2
    plt.loglog(ref_deltas, ref_forward, '--', label='Slope=1 (Forward)')  # 绘制斜率1的参考线，虚线
    plt.loglog(ref_deltas, ref_central, '--', label='Slope=2 (Central)')  # 绘制斜率2的参考线，虚线

    plt.xlabel('Step Size (delta)')  # 设置x轴标签为“步长”
    plt.ylabel('Relative Error')  # 设置y轴标签为“相对误差”
    plt.title('Error vs Step Size for Numerical Differentiation')  # 设置图形标题
    plt.legend()  # 显示图例
    plt.grid(True, which="both", ls="--")  # 添加对数网格线，虚线样式
    plt.show()  # 显示图形


def print_results(deltas, forward_errors, central_errors):
    """打印计算结果表格

    参数:
        deltas (array): 步长数组
        forward_errors (array): 前向差分误差数组
        central_errors (array): 中心差分误差数组
    """
    print(f"{'Step Size':<15} {'Forward Error':<20} {'Central Error':<20}")  # 打印表头，设置列宽对齐
    for delta, fwd_err, ctr_err in zip(deltas, forward_errors, central_errors):  # 遍历步长和误差数据
        print(f"{delta:<15.2e} {fwd_err:<20.6e} {ctr_err:<20.6e}")  # 格式化输出步长和误差，保留科学计数法


def main():
    """主函数"""
    x_point = 1.0  # 设置求导点为 x = 1.0

    deltas, forward_errors, central_errors = calculate_errors(x_point)  # 调用函数计算不同步长下的误差

    print(f"函数 f(x) = x(x-1) 在 x = {x_point} 处的解析导数值: {analytical_derivative(x_point)}")  # 打印解析导数值
    print_results(deltas, forward_errors, central_errors)  # 打印误差结果表格

    plot_errors(deltas, forward_errors, central_errors)  # 绘制误差-步长关系图

    forward_best_idx = np.argmin(forward_errors)  # 找到前向差分误差最小的索引
    central_best_idx = np.argmin(central_errors)  # 找到中心差分误差最小的索引

    print("\n最优步长分析:")  # 打印最优步长分析标题
    print(
        f"前向差分最优步长: {deltas[forward_best_idx]:.2e}, 相对误差: {forward_errors[forward_best_idx]:.6e}")  # 打印前向差分最优步长和误差
    print(
        f"中心差分最优步长: {deltas[central_best_idx]:.2e}, 相对误差: {central_errors[central_best_idx]:.6e}")  # 打印中心差分最优步长和误差

    mid_idx = len(deltas) // 2  # 选择步长数组中间索引，用于计算收敛阶数
    forward_slope = np.log(forward_errors[mid_idx] / forward_errors[mid_idx - 2]) / np.log(
        deltas[mid_idx] / deltas[mid_idx - 2])  # 计算前向差分收敛阶数
    central_slope = np.log(central_errors[mid_idx] / central_errors[mid_idx - 2]) / np.log(
        deltas[mid_idx] / deltas[mid_idx - 2])  # 计算中心差分收敛阶数

    print("\n收敛阶数分析:")  # 打印收敛阶数分析标题
    print(f"前向差分收敛阶数约为: {forward_slope:.2f}")  # 打印前向差分收敛阶数
    print(f"中心差分收敛阶数约为: {central_slope:.2f}")  # 打印中心差分收敛阶数


if __name__ == "__main__":
    main()  # 运行主函数
