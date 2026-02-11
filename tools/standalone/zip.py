"""ZIP 口令并行暴力尝试脚本（中文注释版）。

说明：
- 本脚本用于学习/恢复场景下的口令尝试示例；
- 使用多进程并结合 tqdm 展示实时进度；
- 支持 UTF-8 / GBK 口令编码尝试。
"""

import zipfile
import multiprocessing
import itertools
import os
import time
from typing import Optional
from tqdm import tqdm  # 引入可视化进度条库

# 定义密码字符集：大写字母、小写字母、数字
CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# 全局统计：已尝试的密码总数（用于可视化展示）
total_attempted = 0

def is_zip_file_valid(zip_path: str) -> bool:
    """
    验证zip文件是否存在且可正常读取
    :param zip_path: zip文件路径
    :return: 有效返回True，无效返回False
    """
    if not os.path.exists(zip_path):
        print(f"错误：zip文件 '{zip_path}' 不存在！")
        return False
    if not zipfile.is_zipfile(zip_path):
        print(f"错误：'{zip_path}' 不是一个有效的zip文件！")
        return False
    # 额外验证：zip文件非空
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if len(zf.namelist()) == 0:
            print(f"错误：'{zip_path}' 是一个空zip文件，无需破解！")
            return False
    return True

def try_zip_password(zip_path: str, password: tuple) -> Optional[str]:
    """
    尝试单个密码破解zip文件（注意：password是元组，需要拼接成字符串）
    :param zip_path: zip文件路径
    :param password: 密码元组（来自itertools.product的输出）
    :return: 破解成功返回密码字符串，失败返回None
    """
    # 拼接密码元组为字符串（itertools.product返回的是元组，如('a','1','B')）
    password_str = ''.join(password)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # 设置密码：兼容传统zip密码，同时支持UTF-8和GBK编码（避免密码编码问题）
            try:
                # 优先尝试UTF-8编码
                zip_file.setpassword(password_str.encode('utf-8'))
            except:
                # 备用尝试GBK编码（Windows环境下常见）
                zip_file.setpassword(password_str.encode('gbk', errors='ignore'))
            # 尝试读取第一个文件来验证密码是否正确（忽略解码错误，只验证密码有效性）
            first_file = zip_file.namelist()[0]
            # 读取文件字节流，不进行解码（避免文件内容/文件名解码错误）
            zip_file.read(first_file)
            return password_str  # 破解成功，返回密码
    except (RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile, UnicodeDecodeError):
        # 新增捕获UnicodeDecodeError，抑制无关解码错误
        return None
    except Exception as e:
        # 只打印严重未知错误，避免刷屏
        if "invalid continuation byte" not in str(e):
            print(f"\n⚠️  尝试密码 '{password_str}' 时出现严重未知错误：{str(e)}")
        return None

# ===== 顶层辅助函数（替换原来的lambda，可被pickle序列化）=====
def task_wrapper(task_tuple: tuple) -> Optional[str]:
    """
    任务包装器，接收任务元组，调用try_zip_password
    :param task_tuple: 元组格式 (zip_path, password_tuple)
    :return: 破解成功返回密码，失败返回None
    """
    zip_path, password_tuple = task_tuple
    return try_zip_password(zip_path, password_tuple)

def parallel_zip_crack(zip_path: str, process_num: Optional[int] = None) -> None:
    """
    并行暴力破解zip文件主函数（带可视化进度）
    :param zip_path: zip文件路径
    :param process_num: 并行进程数，默认使用CPU核心数
    """
    global total_attempted
    total_attempted = 0  # 初始化已尝试密码数
    
    # 先验证zip文件有效性
    if not is_zip_file_valid(zip_path):
        return

    # 设置并行进程数（默认CPU核心数，避免进程过多导致系统卡顿）
    if process_num is None or process_num <= 0:
        process_num = multiprocessing.cpu_count()
    # 额外限制：进程数不超过32（避免过多进程抢占资源）
    process_num = min(process_num, 32)
    
    # 记录破解开始时间（用于统计耗时）
    start_time = time.time()

    # 打印初始化信息
    print("=" * 60)
    print(f"开始破解zip文件：{os.path.basename(zip_path)}")
    print(f"并行进程数：{process_num}")
    print(f"密码范围：大小写字母+数字，长度1-8位")
    print(f"开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("=" * 60)

    # 遍历密码长度（1到8位）
    for password_length in range(1, 9):
        # 计算当前密码长度的总组合数
        total_combinations = len(CHAR_SET) ** password_length
        print(f"\n【第 {password_length} 位密码】总组合数：{total_combinations:,} 个")
        
        # 生成当前长度的所有可能密码（迭代器，不占用大量内存）
        password_generator = itertools.product(CHAR_SET, repeat=password_length)
        
        # 创建进程池并并行执行密码尝试
        with multiprocessing.Pool(processes=process_num) as pool:
            # 批量提交任务，每次传入(zip_path, password)元组
            tasks = [(zip_path, pwd) for pwd in password_generator]
            
            # 可视化进度条：tqdm包装任务迭代，显示当前进度、耗时、已尝试速度
            pbar = tqdm(
                tasks,
                desc=f"尝试 {password_length} 位密码",
                unit="个密码",
                dynamic_ncols=True,  # 自适应控制台宽度
                leave=True  # 进度条完成后保留，方便查看历史记录
            )
            
            # 替换lambda，使用顶层辅助函数task_wrapper
            for result in pool.imap_unordered(task_wrapper, pbar):
                # 更新总尝试密码数
                total_attempted += 1
                # 更新进度条的附加信息（显示耗时、总尝试数）
                elapsed_time = time.time() - start_time
                pbar.set_postfix({
                    "总尝试": f"{total_attempted:,}",
                    "耗时": f"{elapsed_time:.2f}s",
                    "速度": f"{total_attempted/elapsed_time:.2f} 个/秒" if elapsed_time > 0 else "0 个/秒"
                })
                
                if result is not None:
                    # 破解成功，立即关闭进度条和进程池
                    pbar.close()
                    pool.terminate()
                    
                    # 计算总耗时
                    total_elapsed = time.time() - start_time
                    print("\n" + "=" * 60)
                    print(f"✅ 破解成功！")
                    print(f"zip文件密码：{result}")
                    print(f"总尝试密码数：{total_attempted:,} 个")
                    print(f"总耗时：{total_elapsed:.2f} 秒（约 {total_elapsed/60:.2f} 分钟）")
                    print("=" * 60)
                    return
    
    # 所有长度尝试完毕仍未破解
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"❌ 破解失败！未在1-8位大小写字母+数字组合中找到有效密码。")
    print(f"总尝试密码数：{total_attempted:,} 个")
    print(f"总耗时：{total_elapsed:.2f} 秒（约 {total_elapsed/60:.2f} 分钟）")
    print("=" * 60)

if __name__ == "__main__":
    # 接收用户输入zip文件路径
    zip_file_path = input("请输入zip文件的完整路径（例如：D:/test/encrypted.zip）：").strip()
    
    # 可选：让用户自定义并行进程数（默认按CPU核心数）
    try:
        process_count = int(input("请输入并行进程数（默认按CPU核心数，直接回车即可）：").strip() or 0)
    except ValueError:
        process_count = 0
    
    # 安装提醒（如果用户未安装tqdm）
    try:
        from tqdm import tqdm
    except ImportError:
        print("\n未检测到tqdm库，正在尝试自动安装...")
        os.system("pip install tqdm")
        from tqdm import tqdm
    
    # 启动破解（带可视化）
    parallel_zip_crack(zip_file_path, process_count)
