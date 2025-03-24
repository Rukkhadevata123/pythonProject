import sys
import time


class ProgressBar:
    """进度条显示类"""

    def __init__(self, total, prefix="", suffix="", decimals=1, length=50, fill="█"):
        """
        初始化进度条

        参数:
        total: 总迭代次数
        prefix: 前缀字符串
        suffix: 后缀字符串
        decimals: 百分比的小数位数
        length: 进度条长度
        fill: 填充字符
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.iteration = 0
        self.start_time = time.time()

    def update(self, iteration=None):
        """
        更新进度条

        参数:
        iteration: 当前迭代次数，如果为None则自动递增
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1

        self._print()

    def finish(self):
        """完成进度条"""
        self.iteration = self.total
        self._print()
        print()  # 添加一个换行

    def _print(self):
        """打印进度条"""
        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (self.iteration / float(self.total))
        )
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + "-" * (self.length - filled_length)

        # 计算已运行时间
        elapsed = time.time() - self.start_time
        time_str = f"{elapsed:.1f}s"

        # 如果未完成，估计剩余时间
        if self.iteration < self.total:
            time_per_iter = elapsed / self.iteration if self.iteration > 0 else 0
            eta = time_per_iter * (self.total - self.iteration)
            time_str += f" | ETA: {eta:.1f}s"

        sys.stdout.write(
            f"\r{self.prefix} |{bar}| {percent}% {self.suffix} ({time_str})"
        )
        sys.stdout.flush()

    def simulate(self, delay=0.05):
        """
        模拟进度（用于异步操作已完成但需要显示进度的情况）

        参数:
        delay: 每次更新之间的延迟时间
        """
        for i in range(self.total):
            self.update(i + 1)
            time.sleep(delay)
        print()  # 添加一个换行


def create_progress_bar(total, description="Processing", **kwargs):
    """
    创建并返回一个进度条实例

    参数:
    total: 总迭代次数
    description: 进度条描述
    **kwargs: 传递给ProgressBar的额外参数

    返回值:
    ProgressBar实例
    """
    return ProgressBar(total, prefix=f"{description}:", suffix="完成", **kwargs)
