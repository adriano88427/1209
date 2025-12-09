import baostock as bs
import pandas as pd
import datetime
import time
import os
import sys
import logging

# 配置日志系统
def setup_logging():
    # 创建日志文件路径
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(log_dir, f"stock_signal_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 重置root logger配置
    root_logger = logging.getLogger('')
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 配置logging - 设置为DEBUG级别以捕获所有日志
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台输出 - 保持INFO级别避免过多调试信息
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # 记录日志系统初始化信息
    logging.info(f"日志系统初始化完成")
    logging.info(f"日志级别设置 - 文件: DEBUG, 控制台: INFO")
    
    return log_filename

# ================================================
# 配置设置区域 - 用户可以在这里修改参数
# ================================================
# 运行模式设置
# "是" = 使用所有股票进行查询（全量模式）
# "否" = 只使用前20支股票进行测试（测试模式）
USE_ALL_STOCKS = "是"  # 默认全量模式

# 日期范围设置
START_DATE = "2025-04-20"  # 分析开始日期
END_DATE = "2025-05-01"  # 分析结束日期

# 信号检测参数
DROP_THRESHOLD = -0.14  # 跌幅阈值（-14%）
DETECTION_DAYS = 1  # 检测连续下跌天数
# ================================================

class TradingSignalDetector:
    def __init__(self, use_all_stocks):
        self.stock_list = []
        self.signal_stocks = []
        # 日期范围参数设置（从全局配置获取）
        self.start_date = START_DATE  # 分析开始日期
        self.end_date = END_DATE  # 分析结束日期
        self.stock_name_map = {}
        self.use_all_stocks = use_all_stocks  # 是否使用所有股票进行查询
        self.specified_stock = None  # 指定的股票代码
        self.specified_stock_name = None  # 指定的股票名称
        
        # 检查是否指定了具体股票
        if use_all_stocks.startswith("STOCK:"):
            stock_info = use_all_stocks.split(":")
            if len(stock_info) >= 3:
                self.specified_stock = stock_info[1]
                self.specified_stock_name = stock_info[2]
        
    def initialize(self):
        """初始化BAOSTOCK连接"""
        logging.info("正在初始化BAOSTOCK连接...")
        lg = bs.login()
        
        if lg.error_code != '0':
            logging.error(f"BAOSTOCK登录失败: 错误代码={lg.error_code}, 错误信息={lg.error_msg}")
            print(f"BAOSTOCK登录失败: 错误代码={lg.error_code}, 错误信息={lg.error_msg}")
            return False
        
        logging.info("BAOSTOCK登录成功")
        return True
    
    def load_stock_list(self):
        """加载股票列表（从创业板股票列表.xlsx文件）"""
        logging.info("开始加载股票列表...")
        
        # 如果指定了具体股票，直接使用该股票信息
        if self.specified_stock:
            logging.info(f"使用指定的股票: {self.specified_stock}({self.specified_stock_name})")
            
            # 格式化股票代码（添加市场前缀）
            code_str = str(self.specified_stock).strip()
            if code_str and (code_str.isdigit() or code_str.replace('.', '').isdigit()):
                # 根据代码前缀添加市场标识
                if '.' not in code_str:  # 如果没有市场前缀
                    if code_str.startswith('6'):
                        formatted_code = 'sh.' + code_str
                    else:
                        formatted_code = 'sz.' + code_str
                else:
                    formatted_code = code_str
                
                # 设置股票列表和名称映射
                self.stock_list = [formatted_code]
                self.stock_name_map[formatted_code] = self.specified_stock_name
                self.stock_name_map[code_str] = self.specified_stock_name
                
                logging.info(f"成功设置指定股票: {formatted_code} - {self.specified_stock_name}")
                return True
            else:
                error_msg = f"错误: 指定的股票代码 '{code_str}' 无效"
                logging.error(error_msg)
                print(error_msg)
                return False
        
        # 尝试从Excel文件加载创业板股票列表
        try:
            # 检查是否安装了必要的Excel读取库
            try:
                import openpyxl
                logging.info("已成功导入openpyxl库")
            except ImportError:
                logging.warning("未找到openpyxl库，将使用pandas默认方式读取Excel")
            
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(script_dir, '创业板股票列表.xlsx')
            logging.info(f"Excel文件路径: {excel_path}")
            
            # 检查文件是否存在
            if not os.path.exists(excel_path):
                raise FileNotFoundError(f"Excel文件不存在: {excel_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(excel_path)
            logging.info(f"尝试从文件 {excel_path} 读取创业板股票列表...")
            logging.info(f"文件大小: {file_size} 字节")
            print(f"尝试从文件 {excel_path} 读取创业板股票列表...")
            print(f"文件大小: {file_size} 字节")
            
            # 读取Excel文件
            try:
                stock_df = pd.read_excel(excel_path, engine='openpyxl')
                logging.info("已使用openpyxl引擎读取Excel文件")
            except Exception as e:
                logging.warning(f"使用openpyxl读取失败，将尝试默认引擎: {e}")
                stock_df = pd.read_excel(excel_path)  # 使用pandas默认引擎读取
                logging.info("已使用pandas默认引擎读取Excel文件")
            
            # 清空股票代码和名称映射字典
            self.stock_name_map = {}
            logging.info("已清空股票代码和名称映射字典")
            
            # 方案1: 尝试直接读取列名为'股票代码'和'股票名称'的格式
            if '股票代码' in stock_df.columns and '股票名称' in stock_df.columns:
                logging.info("检测到标准格式Excel，包含'股票代码'和'股票名称'列")
                # 获取股票代码和名称列表
                stock_codes = stock_df['股票代码'].tolist()
                stock_names = stock_df['股票名称'].tolist()
                logging.info(f"获取到{len(stock_codes)}条股票记录")
                
                # 格式化股票代码（添加市场前缀）并建立映射
                self.stock_list = []
                success_count = 0
                error_count = 0
                
                for i, (code, name) in enumerate(zip(stock_codes, stock_names)):
                    try:
                        # 确保代码是字符串
                        code_str = str(code).strip()
                        name_str = str(name).strip() if pd.notna(name) else ""
                        
                        # 验证是否为有效股票代码
                        if code_str and (code_str.isdigit() or code_str.replace('.', '').isdigit()):
                            # 根据代码前缀添加市场标识
                            if code_str.startswith('6'):
                                formatted_code = 'sh.' + code_str
                            else:
                                formatted_code = 'sz.' + code_str
                            
                            self.stock_list.append(formatted_code)
                            # 保存股票代码和名称的映射关系
                            self.stock_name_map[formatted_code] = name_str
                            self.stock_name_map[code_str] = name_str
                            success_count += 1
                        else:
                            warning_msg = f"警告: 第{i+1}行股票代码 '{code_str}' 无效"
                            logging.warning(warning_msg)
                            print(warning_msg)
                            error_count += 1
                    except Exception as e:
                        error_msg = f"处理第{i+1}行时出错: {e}"
                        logging.error(error_msg)
                        print(error_msg)
                        error_count += 1
                
                log_msg = f"成功加载 {success_count} 只股票，跳过 {error_count} 条无效记录"
                logging.info(log_msg)
                # 保留logging.info，print语句可保留用于控制台输出
                logging.info(f"前5支股票列表示例: {self.stock_list[:5]}...")
                print(f"前5支股票列表示例: {self.stock_list[:5]}...")
                
                # 验证是否成功加载了股票
                if self.stock_list:
                    logging.info(f"股票列表加载成功，共{len(self.stock_list)}只股票")
                    return True
                else:
                    error_msg = "错误: 虽然Excel格式正确，但未能加载到有效股票数据"
                    logging.error(error_msg)
                    print(error_msg)
                    return False
            
            # 方案2: 尝试处理垂直排列的键值对格式（股票代码和股票名称交替出现）
            elif len(stock_df.columns) == 1:
                info_msg = "检测到单列格式，尝试解析为股票代码和名称对"
                logging.info(info_msg)
                print(info_msg)
                # 获取单列数据
                column_data = stock_df.iloc[:, 0].tolist()
                logging.info(f"单列数据包含{len(column_data)}条记录")
                
                # 假设数据格式为：股票代码, 股票名称, 代码值, 名称值, 代码值, 名称值...
                # 跳过表头（如果存在）
                i = 0
                if len(column_data) >= 2 and (str(column_data[0]).strip() == '股票代码' or str(column_data[1]).strip() == '股票名称'):
                    i = 2  # 跳过前两行表头
                    info_msg = "跳过表头，从第3行开始读取数据"
                    logging.info(info_msg)
                    print(info_msg)
                
                # 读取股票代码和名称对
                self.stock_list = []
                pair_count = 0
                
                while i + 1 < len(column_data):
                    try:
                        # 获取股票代码和名称
                        code_str = str(column_data[i]).strip()
                        name = str(column_data[i+1]).strip()
                        
                        # 验证是否为有效股票代码（只保留数字）
                        if code_str.isdigit() and len(code_str) >= 6:
                            # 根据代码前缀添加市场标识
                            if code_str.startswith('6'):
                                formatted_code = 'sh.' + code_str
                            else:
                                formatted_code = 'sz.' + code_str
                            
                            self.stock_list.append(formatted_code)
                            # 保存股票代码和名称的映射关系
                            self.stock_name_map[formatted_code] = name
                            self.stock_name_map[code_str] = name
                            pair_count += 1
                    except Exception as e:
                        error_msg = f"处理数据行{i}和{i+1}时出错: {e}"
                        logging.error(error_msg)
                        print(error_msg)
                    
                    i += 2  # 前进到下一对
                
                if self.stock_list:
                    success_msg = f"成功从Excel文件加载创业板股票列表，格式为垂直键值对，共加载{pair_count}只股票"
                    logging.info(success_msg)
                    print(success_msg)
                    return True
                else:
                    warning_msg = "无法从垂直键值对格式中识别有效股票数据"
                    logging.warning(warning_msg)
                    print(warning_msg)
            
            # 如果都不成功，打印更详细的调试信息
            error_msg = "Excel文件格式不符合预期"
            logging.error(error_msg)
            print(error_msg)
            logging.error(f"Excel文件中的列名: {list(stock_df.columns)}")
            print(f"Excel文件中的列名: {list(stock_df.columns)}")
            logging.error(f"数据格式示例（前10行）: {stock_df.head(10).values.tolist()}")
            print(f"数据格式示例（前10行）: {stock_df.head(10).values.tolist()}")
            raise ValueError(f"无法识别股票代码列和股票名称列，当前列名: {list(stock_df.columns)}")
                
        except ImportError as e:
            error_msg = f"错误: 缺少必要的库 - {e}"
            logging.error(error_msg)
            print(error_msg)
            logging.error("请安装openpyxl库: pip install openpyxl")
            print("请安装openpyxl库: pip install openpyxl")
            logging.warning("尝试加载默认股票列表")
            self._load_default_stock_list()
            return False
            
        except FileNotFoundError as e:
            error_msg = f"错误: {e}"
            logging.error(error_msg)
            print(error_msg)
            logging.error("请确保'创业板股票列表.xlsx'文件存在于程序目录下")
            print("请确保'创业板股票列表.xlsx'文件存在于程序目录下")
            logging.warning("尝试加载默认股票列表")
            self._load_default_stock_list()
            return False
            
        except Exception as e:
            error_msg = f"从Excel文件加载失败: {type(e).__name__}: {e}"
            logging.error(error_msg)
            print(error_msg)
            logging.warning("尝试加载默认股票列表")
            self._load_default_stock_list()
            return False
            
    def _load_default_stock_list(self):
        """原默认股票池已移除"""
        logging.warning("警告: 无法加载股票列表，且已不再提供默认股票池")
        print("警告: 无法加载股票列表，且已不再提供默认股票池")
        self.stock_list = []
        self.stock_name_map = {}
    

            
    def check_drop_signal(self, stock_code, stock_name=None, days=None, threshold=None):
        # 使用全局配置的默认值
        if days is None:
            days = DETECTION_DAYS
        if threshold is None:
            threshold = DROP_THRESHOLD
        """
        检查股票是否满足单日大幅下跌信号
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称（可选）
            days: 连续下跌天数，默认1天
            threshold: 跌幅阈值，默认-0.14（-14%）
            
        Returns:
            list: 满足条件的信号列表
        """
        try:
            # 确保股票代码是字符串类型
            stock_code = str(stock_code)
            
            # 格式化股票代码（确保包含市场前缀）
            if not (stock_code.startswith('sz.') or stock_code.startswith('sh.')):
                if stock_code.startswith('6'):
                    stock_code = 'sh.' + stock_code
                else:
                    stock_code = 'sz.' + stock_code
            
            # 从映射字典中获取股票名称
            stock_name = self.stock_name_map.get(stock_code, "未知名称")
            if stock_name == "未知名称":
                # 尝试不带前缀的股票代码
                code_without_prefix = stock_code.split('.')[-1]
                stock_name = self.stock_name_map.get(code_without_prefix, "未知名称")
            
            # 获取股票的上市日期
            listing_date = self.get_stock_listing_date(stock_code)
            
            print(f"{stock_code}的股票名称: {stock_name}, 上市日期: {listing_date if listing_date else '未知'}")
            
            # 使用初始化时设置的日期范围
            start_date = self.start_date
            
            # 获取历史K线数据（包含开盘价、收盘价、最低价）- 使用前复权价格
            print(f"正在获取{stock_code}的前复权K线数据...")
            # 使用正确的前复权参数
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,open,close,low,high",
                start_date=start_date,
                end_date=self.end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            # 打印BAOSTOCK的响应状态
            print(f"BAOSTOCK响应 - 股票: {stock_code}, 错误代码: {rs.error_code}, 错误信息: {rs.error_msg}")
            
            if rs.error_code != '0':
                print(f"获取{stock_code}历史数据失败: {rs.error_msg}")
                return []
            
            # 解析数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            # 打印获取到的数据条数
            print(f"{stock_code}获取到{len(data_list)}条K线数据")
            
            # 检查数据量
            if len(data_list) == 0:
                print(f"警告: {stock_code}获取到0条数据，可能原因:")
                print(f"1. 股票代码可能不存在或已退市")
                print(f"2. 2024年该股票没有交易数据")
                print(f"3. BAOSTOCK数据库中没有该股票的历史数据")
                return []
            
            # 对于单日跌幅信号，只要有数据就可以进行检测
            if len(data_list) < 2:  # 需要至少两天数据才能计算单日涨跌幅
                print(f"{stock_code}数据不足2天，无法计算涨跌幅，跳过")
                # 如果获取到数据但不足2天，打印看看
                if data_list:
                    print(f"{stock_code}的部分数据示例: {data_list[:3]}")
                return []
            
            # 创建DataFrame，所有价格数据均为前复权价格
            df = pd.DataFrame(data_list, columns=['date', 'open', 'close', 'low', 'high'])
            
            # 数据验证和类型转换
            print("前复权数据验证和转换...")
            # 确保价格列可以正确转换为浮点数
            try:
                df['open'] = pd.to_numeric(df['open'], errors='coerce')  # 转换为浮点数 - 前复权开盘价
                df['close'] = pd.to_numeric(df['close'], errors='coerce')  # 转换为浮点数 - 前复权收盘价
                df['low'] = pd.to_numeric(df['low'], errors='coerce')  # 转换为浮点数 - 前复权最低价
                df['high'] = pd.to_numeric(df['high'], errors='coerce')  # 转换为浮点数 - 前复权最高价
                
                # 检查是否有转换失败的数据
                if df['open'].isna().any() or df['close'].isna().any() or df['low'].isna().any() or df['high'].isna().any():
                    print(f"警告: 部分价格数据转换失败，存在NaN值")
                    print(f"open列NaN值数量: {df['open'].isna().sum()}")
                    print(f"close列NaN值数量: {df['close'].isna().sum()}")
                    print(f"low列NaN值数量: {df['low'].isna().sum()}")
                    print(f"high列NaN值数量: {df['high'].isna().sum()}")
                
                # 数据框已创建并验证
                print(f"数据日期范围: {df['date'].min()} 至 {df['date'].max()}")
            except Exception as e:
                print(f"数据转换错误: {e}")
                return []
            
            # 存储所有满足条件的信号
            signals = []
            
            # 遍历所有交易日，计算单日跌幅
            for i in range(1, len(df)):  # 从第2条数据开始，因为需要前一天收盘价
                # 获取当前交易日和前一交易日的数据
                current_day = df.iloc[i]
                previous_day = df.iloc[i-1]
                
                # 计算当日收盘涨跌幅 - 使用（当日收盘价 - 前日收盘价）/ 前日收盘价
                daily_return = (current_day['close'] - previous_day['close']) / previous_day['close']
                
                signal_date = current_day['date']
                
                # 判断股票上市是否超过11个交易日
                is_listed_enough_days = True
                if listing_date:
                    try:
                        # 再次验证上市日期格式
                        if len(listing_date) == 10 and listing_date[4] == '-' and listing_date[7] == '-':
                            signal_datetime = datetime.datetime.strptime(signal_date, '%Y-%m-%d')
                            listing_datetime = datetime.datetime.strptime(listing_date, '%Y-%m-%d')
                            days_since_listing = (signal_datetime - listing_datetime).days
                            
                            # 计算实际交易天数（约为日历天数的0.7倍，考虑周末和节假日）
                            trading_days_since_listing = days_since_listing * 0.7
                            
                            if trading_days_since_listing < 11:
                                is_listed_enough_days = False
                                print(f"{stock_code}在{signal_date}上市不足11个交易日，跳过此信号")
                        else:
                            print(f"{stock_code}的上市日期格式无效: {listing_date}，跳过上市时间检查")
                    except Exception as e:
                        print(f"计算上市天数时出错: {e}")
                        # 出错时默认认为已满足上市时间要求，不影响其他信号判断
                        pass
                
                # 判断是否满足条件：当日收盘涨跌幅小于-14% 且 上市超过11个交易日
                if daily_return <= threshold and is_listed_enough_days:
                    
                    # 获取信号日的价格信息
                    signal_day_open = current_day['open']
                    signal_day_close = current_day['close']
                    signal_day_low = current_day['low']
                    signal_day_high = current_day['high']
                    previous_day_close = previous_day['close']
                    
                    # 调试信息已移除
                    
                    # 获取信号发出后1,3,4,5,6,7,8个交易日的价格
                    after_prices = {}  # 存储信号发出后的价格
                    
                    for day_offset in [1, 2, 3]:
                        # 计算目标索引
                        target_index = i + days + day_offset - 1  # -1是因为信号发出后第1天就是下一天
                        
                        if target_index < len(df):
                            if day_offset == 1:  # 信号发出后第1天同时取开盘价和收盘价
                                after_prices[f"day_{day_offset}_open"] = df['open'].iloc[target_index]  # 前复权开盘价
                                after_prices[f"day_{day_offset}_close"] = df['close'].iloc[target_index]  # 前复权收盘价
                            else:  # 其他天数取前复权收盘价
                                after_prices[f"day_{day_offset}_close"] = df['close'].iloc[target_index]  # 前复权收盘价
                        else:
                            if day_offset == 1:
                                after_prices[f"day_{day_offset}_open"] = None
                                after_prices[f"day_{day_offset}_close"] = None
                            else:
                                after_prices[f"day_{day_offset}_close"] = None  # 数据不足时设为None
                    
                    # 确保stock_name有值
                    if stock_name is None:
                        stock_name = '未知名称'
                    
                    # 信号触发信息已移除
                    
                    # 计算-14%阈值价格（基于前一日收盘价计算）
                    threshold_price = previous_day_close * (1 + threshold)  # 1 + (-0.14) = 0.86
                    
                    # 计算当日回调 = (当日最高价 - 当日收盘价) / 当日最高价
                    daily_callback = (signal_day_high - signal_day_close) / signal_day_high if signal_day_high != 0 else None
                    
                    # 计算当日最大跌幅 = (当日最低价 - 前日收盘价) / 前日收盘价
                    max_daily_drop = (signal_day_low - previous_day_close) / previous_day_close if previous_day_close != 0 else None
                    
                    # 存储信号信息
                    signal_info = {
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'signal_end_date': signal_date,  # 信号日期
                        'drop_percent': max_daily_drop,  # 当日最大跌幅（使用当日最低价计算）
                        'daily_return': daily_return,  # 当日收盘涨跌幅
                        'first_day_open': signal_day_open,  # 信号日开盘价
                        'last_day_close': signal_day_close,  # 信号日收盘价
                        'signal_day_low': signal_day_low,  # 信号日最低价
                        'signal_day_high': signal_day_high,  # 信号日最高价
                        'daily_callback': daily_callback,  # 当日回调
                        'previous_day_close': previous_day_close,  # 信号发出前一日收盘价
                        'after_day_1_open': after_prices.get('day_1_open'),  # 信号后第1天开盘价
                        'after_day_1_close': after_prices.get('day_1_close'),  # 信号后第1天收盘价
                        'after_day_2_close': after_prices.get('day_2_close'),  # 信号后第2天收盘价
                        'threshold_price': threshold_price,  # -14%阈值价格
                        'listing_date': listing_date,  # 上市日期
                    }
                    
                    signals.append(signal_info)
            
            return signals
        
        except Exception as e:
            print(f"检查{stock_code}信号时出错: {e}")
            return []
    
    def scan_all_stocks(self):
        """扫描所有股票，检测交易信号（获取所有符合条件的信号）"""
        # 记录当前use_all_stocks值，确认是否正确设置
        logging.info(f"scan_all_stocks方法中的use_all_stocks值: {self.use_all_stocks}")
        logging.info(f"当前加载的股票总数: {len(self.stock_list)}")
        # 保留logging但移除重复的print语句，使用logging.info确保日志完整记录
        
        # 根据参数决定使用所有股票还是测试股票
        if self.use_all_stocks == "是":
            stocks_to_scan = self.stock_list
            logging.info(f"全量模式 - 开始扫描所有{len(stocks_to_scan)}支股票...")
            # 保留logging.info，print语句可保留用于控制台输出
        else:
            stocks_to_scan = self.stock_list[:20]
            logging.info(f"测试模式 - 开始扫描前20支股票（共{len(self.stock_list)}支）...")
            # 保留logging.info，print语句可保留用于控制台输出
        
        # 使用列表存储所有信号信息字典
        self.all_signals = []
        
        for i, stock in enumerate(stocks_to_scan):
            # 显示进度并记录到日志
            logging.info(f"扫描进度: {i+1}/{len(stocks_to_scan)}")
            print(f"扫描进度: {i+1}/{len(stocks_to_scan)}", end='\r')
            # 每50支股票休眠并记录日志
            if (i+1) % 50 == 0:
                sleep_info = f"处理了{i+1}支股票，休眠1秒以避免API限流..."
                logging.info(sleep_info)
                print(f"\n{sleep_info}")
                time.sleep(1)
            
            # 每10支股票记录一次进度到日志
            if i % 10 == 0:
                logging.info(f"扫描进度: {i+1}/{len(stocks_to_scan)}")
            
            # 检查信号，获取所有满足条件的信号列表
            stock_signals = self.check_drop_signal(stock)
            
            if stock_signals:
                # 对每个信号计算前10个交易日的最大涨幅
                for signal in stock_signals:
                    # 调用方法获取前10个交易日的最大涨幅（只取第一个返回值，即最大涨幅百分比）
                    max_gain_result = self.get_max_gain_percentage(signal['stock_code'], signal['signal_end_date'])
                    # 将最大涨幅信息添加到信号字典中（只使用第一个返回值）
                    signal['max_gain_percentage'] = max_gain_result if max_gain_result else None
                    # 所有信号都添加到总列表（不再过滤）
                    self.all_signals.append(signal)
                    # 记录信号发现到日志
                    stock_name = signal['stock_name'] or '未知名称'
                    logging.info(f"发现信号: {signal['stock_code']}({stock_name}) 信号结束日: {signal['signal_end_date']} 日最大跌幅: {signal['drop_percent']:.2%}")
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        logging.info(f"扫描完成！发现{len(self.all_signals)}个满足条件的信号")
        # 保留logging.info，print语句可保留用于控制台输出
        
        if self.all_signals:
            logging.info("满足条件的信号列表:")
            # 保留logging.info，print语句可保留用于控制台输出
            for signal in self.all_signals[:10]:  # 只显示前10个信号，避免输出过多
                stock_name = signal['stock_name'] or '未知名称'
                log_msg = f"  - {signal['stock_code']}({stock_name}) 信号结束日: {signal['signal_end_date']} 日最大跌幅: {signal['drop_percent']:.2%} 前10日最大涨幅: {signal['max_gain_percentage']:.2f}%" if signal['max_gain_percentage'] is not None else f"  - {signal['stock_code']}({stock_name}) 信号结束日: {signal['signal_end_date']} 日最大跌幅: {signal['drop_percent']:.2%} 前10日最大涨幅: 无法计算"
                logging.info(log_msg)
                print(log_msg)
            if len(self.all_signals) > 10:
                remaining_count = len(self.all_signals) - 10
                logging.info(f"  ... 以及其他{remaining_count}个信号")
        
        # 为了保持与原有接口兼容，返回有信号的股票列表（去重）
        self.signal_stocks = list(set([signal['stock_code'] for signal in self.all_signals]))
        logging.info(f"信号股票总数（去重）: {len(self.signal_stocks)}")
        return self.signal_stocks
    

    
    def save_signal_results(self):
        """保存所有信号结果到CSV文件（中文列名，包含更多信息）"""
        logging.info("准备保存信号结果...")
        print("\n准备保存信号结果...")
        
        # 检查是否有所有信号数据
        if hasattr(self, 'all_signals') and self.all_signals:
            logging.info(f"总信号数量: {len(self.all_signals)}")
            print(f"总信号数量: {len(self.all_signals)}")
            
            # 手动创建并保存CSV文件
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"all_signal_stocks_{timestamp}.csv"
            
            # 使用pandas保存CSV文件，确保表头完整性
            import pandas as pd
            
            # 准备数据
            data = []
            for signal in self.all_signals:
                # 删除股票代码中的市场标识符（sz.或sh.）
                stock_code_clean = signal['stock_code'].split('.')[-1]
                
                # 直接从signal中获取预先计算好的当日收盘涨跌幅
                daily_return = signal.get('daily_return')
                
                # 计算信号后一日开盘涨跌幅 = (信号后1日开盘价 - 信号日收盘价) / 信号日收盘价
                next_day_open_return = None
                if signal['after_day_1_open'] is not None and signal['last_day_close'] is not None and signal['last_day_close'] != 0:
                    next_day_open_return = (signal['after_day_1_open'] - signal['last_day_close']) / signal['last_day_close']
                
                # 计算次日开盘后总体下跌幅度 = 信号当日收盘涨跌幅 + 信号后一日开盘涨跌幅
                total_drop_after_open = None
                if daily_return is not None and next_day_open_return is not None:
                    total_drop_after_open = daily_return + next_day_open_return
                
                # 计算持股2天收益率
                two_day_return = None
                if signal['after_day_1_open'] is not None and signal['after_day_2_close'] is not None and signal['after_day_1_open'] != 0:
                    two_day_return = (signal['after_day_2_close'] - signal['after_day_1_open']) / signal['after_day_1_open']
                
                # 计算上市天数
                listing_days = '未知'
                try:
                    listing_date = signal.get('listing_date')
                    signal_date = signal['signal_end_date']
                    if listing_date and listing_date != '未知' and len(listing_date) == 10:
                        # 解析日期
                        listing_dt = datetime.datetime.strptime(listing_date, '%Y-%m-%d')
                        signal_dt = datetime.datetime.strptime(signal_date, '%Y-%m-%d')
                        # 计算天数差
                        listing_days = (signal_dt - listing_dt).days
                except Exception as e:
                    error_msg = f"计算{stock_code_clean}上市天数时出错: {str(e)}"
                    logging.error(error_msg)
                    print(error_msg)
                    listing_days = '未知'
                
                # 格式化价格数据（保留2位小数）
                def format_price(price):
                    if price is not None and isinstance(price, (int, float)):
                        return round(price, 2)
                    return price
                
                # 格式化百分比数据（根据数据来源正确处理）
                def format_percent(value, source=None):
                    if value is not None and isinstance(value, (int, float)):
                        # 区分处理不同来源的数据：
                        # 1. get_max_gain_percentage返回的已经是百分比形式（已乘以100）
                        # 2. 其他涨跌幅数据需要先乘以100再格式化
                        if source == 'max_gain' or value > 100:  # 如果值很大，可能已经是百分比形式
                            # 已经是百分比形式，直接格式化
                            result = f"{round(value, 1)}%"
                            
                        else:
                            # 非百分比形式，先乘以100再格式化
                            result = f"{round(value * 100, 1)}%"
                            
                        return result
                    return value
                
                # 准备数据字典
                data.append({
                    '股票代码': stock_code_clean,
                    '股票名称': signal['stock_name'],
                    '上市日期': signal.get('listing_date', '未知'),
                    '信号日期': signal['signal_end_date'],
                    '信号发出时上市天数': listing_days,
                    '日最大跌幅百分比': format_percent(signal['drop_percent']),
                    '信号当日收盘涨跌幅': format_percent(daily_return),
                    '信号后一日开盘涨跌幅': format_percent(next_day_open_return),
                    '次日开盘后总体下跌幅度': format_percent(total_drop_after_open),
                    '前10日最大涨幅': format_percent(signal.get('max_gain_percentage'), source='max_gain') if signal.get('max_gain_percentage') != '数据不足' else '数据不足',
                    '信号发出前一日收盘价': format_price(signal['previous_day_close']),
                    '信号日收盘价': format_price(signal['last_day_close']),
                    '信号日最低价': format_price(signal.get('signal_day_low')),
                    '信号日最高价': format_price(signal.get('signal_day_high')),
                    '当日回调': format_percent(signal.get('daily_callback')),
                    '信号后1日开盘价': format_price(signal['after_day_1_open']),
                    '信号后1日收盘价': format_price(signal['after_day_1_close']),
                    '信号后2日收盘价': format_price(signal['after_day_2_close']),
                    '-14%阈值价格': format_price(signal.get('threshold_price')),
                    '持股2天收益率': format_percent(two_day_return)
                })
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 保存到CSV文件 - 不使用全局float_format，因为包含字符串值
            try:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                logging.info(f"所有信号结果已保存到: {filename}")
                print(f"\n所有信号结果已保存到: {filename}")
                logging.info(f"文件内容预览:")
                print(f"文件内容预览:")
            except Exception as e:
                error_msg = f"保存CSV文件时出错: {str(e)}"
                logging.error(error_msg)
                print(error_msg)
                import traceback
                logging.error(f"错误详情: {traceback.format_exc()}")
            
            # 显示DataFrame的前5行，这样能更好地格式化显示
            print(df.head().to_string(index=False))
        else:
            print("\n未找到所有信号数据，尝试保存基本信号信息")
            # 兼容原有的保存逻辑
            if hasattr(self, 'signal_stocks') and self.signal_stocks:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"signal_stocks_{timestamp}.csv"
                
                df = pd.DataFrame({
                    '股票代码': [code.split('.')[-1] for code in self.signal_stocks],
                    '股票名称': ['未知'] * len(self.signal_stocks),
                    '信号日期': ['查看详细信号文件'] * len(self.signal_stocks)
                })
                
                try:
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    logging.info(f"基本信号信息已保存到: {filename}")
                    print(f"基本信号信息已保存到: {filename}")
                except Exception as e:
                    error_msg = f"保存基本信号信息时出错: {str(e)}"
                    logging.error(error_msg)
                    print(error_msg)
                    import traceback
                    logging.error(f"错误详情: {traceback.format_exc()}")
            else:
                logging.info("没有找到任何信号数据可保存")
                print("没有找到任何信号数据可保存")
    
    def get_stock_listing_date(self, stock_code):
        """
        获取股票的上市日期
        使用BAOSTOCK API的query_stock_basic接口，增加容错机制
        
        Args:
            stock_code: 股票代码
            
        Returns:
            str: 上市日期字符串 (YYYY-MM-DD)，获取失败或无效时返回None
        """
        try:
            # 确保股票代码是字符串类型
            stock_code = str(stock_code)
            
            # 格式化股票代码（确保包含市场前缀）
            if not (stock_code.startswith('sz.') or stock_code.startswith('sh.')):
                if stock_code.startswith('6'):
                    stock_code = 'sh.' + stock_code
                else:
                    stock_code = 'sz.' + stock_code
            
            logging.info(f"正在获取{stock_code}的上市日期...")
            
            # 尝试多种方式获取上市日期
            # 方法1: 使用query_stock_basic直接查询
            rs = bs.query_stock_basic(code=stock_code)
            
            if rs.error_code != '0':
                error_msg = f"获取{stock_code}上市日期失败: {rs.error_msg}"
                logging.error(error_msg)
                print(error_msg)
                # 尝试方法2: 查询更广泛的数据
                retry_msg = f"尝试使用更广泛的查询方式获取{stock_code}的上市日期..."
                logging.info(retry_msg)
                print(retry_msg)
                rs = bs.query_stock_basic(code=stock_code.split('.')[-1])
                
                if rs.error_code != '0':
                    error_msg = f"第二种查询方式也失败: {rs.error_msg}"
                    logging.error(error_msg)
                    print(error_msg)
                    return None
            
            # 解析数据
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            if not data_list:
                print(f"未获取到{stock_code}的基本信息")
                # 尝试方法3: 不带市场前缀查询
                code_without_prefix = stock_code.split('.')[-1]
                print(f"尝试不带市场前缀查询: {code_without_prefix}")
                rs2 = bs.query_stock_basic(code=code_without_prefix)
                
                if rs2.error_code == '0':
                    while rs2.next():
                        data_list.append(rs2.get_row_data())
                
                if not data_list:
                    return None
            
            # 首先尝试第4列（根据BAOSTOCK文档，上市日期在第4列）
            listing_date = None
            if len(data_list[0]) >= 5:
                listing_date = data_list[0][4]
            
            # 如果第4列无效，尝试检查其他可能包含日期的列
            valid_date_found = False
            if listing_date and len(listing_date) >= 8:
                # 检查是否为纯数字格式（如20201231）
                if listing_date.isdigit() and len(listing_date) == 8:
                    # 转换为YYYY-MM-DD格式
                    listing_date = f"{listing_date[:4]}-{listing_date[4:6]}-{listing_date[6:]}"
                    valid_date_found = True
                # 检查是否已经是YYYY-MM-DD格式
                elif len(listing_date) == 10 and listing_date[4] == '-' and listing_date[7] == '-':
                    valid_date_found = True
            
            # 如果第一列无效，尝试遍历其他列查找日期格式
            if not valid_date_found:
                print(f"尝试在其他列中查找{stock_code}的上市日期...")
                for col_index, value in enumerate(data_list[0]):
                    if value and isinstance(value, str):
                        # 检查YYYY-MM-DD格式
                        if len(value) == 10 and value[4] == '-' and value[7] == '-':
                            try:
                                # 验证是否为有效日期
                                datetime.datetime.strptime(value, '%Y-%m-%d')
                                listing_date = value
                                valid_date_found = True
                                print(f"在第{col_index+1}列找到上市日期: {listing_date}")
                                break
                            except:
                                continue
                        # 检查YYYYMMDD格式
                        elif len(value) == 8 and value.isdigit():
                            try:
                                # 转换为YYYY-MM-DD格式
                                listing_date = f"{value[:4]}-{value[4:6]}-{value[6:]}"
                                # 验证是否为有效日期
                                datetime.datetime.strptime(listing_date, '%Y-%m-%d')
                                valid_date_found = True
                                print(f"在第{col_index+1}列找到上市日期并转换: {listing_date}")
                                break
                            except:
                                continue
            
            # 最终验证日期格式
            if valid_date_found and listing_date:
                try:
                    # 再次验证日期有效性
                    datetime.datetime.strptime(listing_date, '%Y-%m-%d')
                    print(f"{stock_code}的上市日期: {listing_date}")
                    return listing_date
                except:
                    print(f"{stock_code}获取到的日期无法解析: {listing_date}")
                    return None
            else:
                print(f"{stock_code}未找到有效的上市日期")
                return None
            
        except Exception as e:
            print(f"获取{stock_code}上市日期时出错: {str(e)}")
            # 出错时返回None，但不影响程序继续运行
            return None
    
    def get_max_gain_percentage(self, stock_code, signal_date):
        logging.info(f"开始计算股票{stock_code}在{signal_date}的最大涨幅")
        """
        计算股票在信号发出前后K线数据中的最大涨幅
        当发现停牌数据（high == low）时，会自动扩大检索范围
        
        Args:
            stock_code: 股票代码
            signal_date: 信号发出日期
            
        Returns:
            float or None: 最大涨幅百分比，无法计算时返回None
        """
        try:
            # 确保股票代码是字符串类型
            stock_code = str(stock_code)
            
            # 格式化股票代码（确保包含市场前缀）
            if not (stock_code.startswith('sz.') or stock_code.startswith('sh.')):
                if stock_code.startswith('6'):
                    stock_code = 'sh.' + stock_code
                else:
                    stock_code = 'sz.' + stock_code
            
            # 将signal_date转换为datetime对象
            signal_datetime = datetime.datetime.strptime(signal_date, '%Y-%m-%d')
            
            print(f"正在获取{stock_code}的K线数据...")
            
            # 初始化变量
            search_days = 20
            max_retries = 5
            retry_count = 0
            valid_data_found = False
            lows = []
            highs = []
            dates = []
            is_before_signal = []
            valid_low_count = 0
            invalid_low_count = 0
            
            while retry_count < max_retries and not valid_data_found:
                retry_count += 1
                logging.info(f"[最大涨幅计算] 第{retry_count}次尝试获取数据，搜索范围：{search_days}天")
                
                # 获取信号前后的K线数据
                start_datetime = signal_datetime - datetime.timedelta(days=search_days)
                start_date = start_datetime.strftime('%Y-%m-%d')
                
                # 向后扩展一点，确保有足够的数据
                end_datetime = signal_datetime + datetime.timedelta(days=5)
                end_date = end_datetime.strftime('%Y-%m-%d')
            
                # 获取历史K线数据，包含最高价、收盘价和最低价
                rs = bs.query_history_k_data_plus(
                    stock_code,
                    "date,high,low,close",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2"  # 前复权
                )
                
                if rs.error_code != '0':
                    logging.warning(f"[最大涨幅计算] 获取数据失败: {rs.error_msg}")
                    # 继续重试
                    search_days += 20
                    continue
                
                # 解析数据
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    logging.warning(f"[最大涨幅计算] 未获取到任何数据")
                    # 继续重试
                    search_days += 20
                    continue
                
                # 创建DataFrame
                df = pd.DataFrame(data_list, columns=['date', 'high', 'low', 'close'])
                
                # 数据类型转换
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                
                # 检查数据质量
                null_high_count = df['high'].isnull().sum()
                null_low_count = df['low'].isnull().sum()
                null_close_count = df['close'].isnull().sum()
                
                logging.info(f"[最大涨幅计算] 数据质量检查 - 高价格为NaN的行数: {null_high_count}/{len(df)}")
                logging.info(f"[最大涨幅计算] 数据质量检查 - 低价格为NaN的行数: {null_low_count}/{len(df)}")
                logging.info(f"[最大涨幅计算] 数据质量检查 - 收盘价为NaN的行数: {null_close_count}/{len(df)}")
                
                if df['high'].isnull().all() or df['close'].isnull().all():
                    logging.warning(f"[最大涨幅计算] 未找到有效的价格数据 - 原因: 所有high或close价格数据为NaN")
                    # 继续重试
                    search_days += 20
                    continue
                
                # 检查信号日期是否在数据中
                signal_datetime = pd.to_datetime(signal_date)
                if signal_datetime not in df['date'].values:
                    logging.warning(f"[最大涨幅计算] 信号日期{signal_date}不在数据范围内")
                    # 继续重试
                    search_days += 20
                    continue
                
                # 找到信号日期在数据框中的索引
                signal_index = df[df['date'] == signal_datetime].index[0]
                
                # 获取信号发出前的所有可用数据数量（不包含信号当天）
                k_lines_available = signal_index  # 不包括信号当天
                
                if k_lines_available < 1:  # 至少需要1个K线
                    logging.warning(f"[最大涨幅计算] 信号前可用K线数量不足: {k_lines_available}")
                    # 继续重试
                    search_days += 20
                    continue
                
                # 从信号日期向前搜索10根非停牌的K线数据，采用拼接方式处理停牌数据
                logging.info(f"[最大涨幅计算] 开始从信号日期向前搜索10根非停牌K线数据（拼接处理停牌数据）")
                
                # 获取信号前的所有K线数据
                all_before_signal_data = df.iloc[:signal_index].copy()
                
                # 过滤出所有非停牌的K线数据（high != low）
                non_suspended_all_before = all_before_signal_data[all_before_signal_data['high'] != all_before_signal_data['low']]
                
                logging.info(f"[最大涨幅计算] 信号前共有{len(non_suspended_all_before)}根非停牌K线数据")
                
                # 如果找到的非停牌K线数量少于10根，尝试扩大搜索范围
                if len(non_suspended_all_before) < 10:
                    logging.warning(f"[最大涨幅计算] 信号前找到的非停牌K线数量不足10根，仅有{len(non_suspended_all_before)}根")
                    search_days += 20
                    continue
                
                # 采用拼接方式：取最近的10根非停牌K线（这些K线可能不连续，但都是有效的交易数据）
                # 从信号日往前，跳过所有停牌数据，获取10根有效的交易K线
                before_signal_data = non_suspended_all_before.tail(10).copy()
                
                # 记录找到的10根非停牌K线的日期范围
                earliest_date = before_signal_data['date'].min().strftime('%Y-%m-%d')
                latest_date = before_signal_data['date'].max().strftime('%Y-%m-%d')
                logging.info(f"[最大涨幅计算] 成功获取10根非停牌K线，日期范围：{earliest_date} 至 {latest_date}")
                logging.info(f"[最大涨幅计算] 这些K线是从信号日向前跳过停牌数据后拼接获取的连续有效交易数据")
                
                # 获取信号后的数据（包含信号当天）
                after_signal_data = df.iloc[signal_index:min(len(df), signal_index+10)].copy()
                
                # 检查信号后的数据是否有停牌情况
                after_has_suspended = (after_signal_data['high'] == after_signal_data['low']).any()
                
                if after_has_suspended:
                    # 找出信号后的停牌数据
                    after_suspended = after_signal_data[after_signal_data['high'] == after_signal_data['low']]
                    logging.warning(f"[最大涨幅计算] 信号后数据中发现停牌数据: {len(after_suspended)}行")
                    for idx, row in after_suspended.iterrows():
                        logging.warning(f"[最大涨幅计算] 信号后停牌数据 - 日期: {row['date'].strftime('%Y-%m-%d')}, 价格: {row['high']:.2f}")
                    
                    # 过滤掉信号后的停牌数据
                    after_signal_data = after_signal_data[after_signal_data['high'] != after_signal_data['low']]
                    
                    # 如果过滤后的数据太少，继续扩大搜索范围
                    if len(after_signal_data) < 1:
                        logging.warning(f"[最大涨幅计算] 信号后过滤停牌数据后剩余太少")
                        search_days += 20
                        continue
                
                # 合并前后数据用于查找涨幅
                combined_data = pd.concat([before_signal_data, after_signal_data])
                
                # 再次验证合并后的数据中没有停牌记录（双重保障）
                combined_has_suspended = (combined_data['high'] == combined_data['low']).any()
                if combined_has_suspended:
                    logging.warning(f"[最大涨幅计算] 警告：合并后的数据中仍存在停牌记录")
                    # 再次过滤确保没有停牌数据
                    combined_data = combined_data[combined_data['high'] != combined_data['low']]
                    
                    # 如果过滤后数据太少，继续扩大搜索范围
                    if len(combined_data) < 11:  # 至少需要信号前10根+信号后1根
                        logging.warning(f"[最大涨幅计算] 过滤后数据不足，需要重新搜索")
                        search_days += 20
                        continue
                
                logging.info(f"[最大涨幅计算] 最终确认：所有使用的K线数据都为非停牌数据")
                
                # 获取K线价格数据和日期
                lows = combined_data['low'].tolist()
                highs = combined_data['high'].tolist()
                dates = combined_data['date'].tolist()
                
                # 标记哪些K线在信号前
                is_before_signal = [(idx < signal_index) for idx in combined_data.index]
                
                # 添加详细的价格数据日志
                logging.info(f"[最大涨幅计算] 获取到的{len(lows)}根K线数据 (无停牌记录):")
                for j in range(len(lows)):
                    signal_flag = "[信号前]" if is_before_signal[j] else "[信号后]"
                    logging.info(f"[最大涨幅计算] K线[{j+1}]: {signal_flag} 日期={dates[j].strftime('%Y-%m-%d')}, 最高价={highs[j]:.2f}, 最低价={lows[j]:.2f}")
                
                # 统计有效和无效价格数量
                valid_low_count = sum(1 for low in lows if low > 0)
                invalid_low_count = sum(1 for low in lows if low <= 0)
                
                logging.info(f"[最大涨幅计算] 价格统计 - 有效低价(low>0)数量: {valid_low_count}, 无效低价数量: {invalid_low_count}")
                
                if len(lows) < 1:
                    logging.warning(f"[最大涨幅计算] 未找到有效的价格数据 - 原因: 交易数据少于1条记录")
                    # 继续重试
                    search_days += 20
                    continue
                
                # 如果通过了所有检查，标记找到有效数据
                valid_data_found = True
                break  # 跳出while循环，继续处理找到的有效数据
            
            # 检查是否成功找到有效数据
            if not valid_data_found:
                logging.info(f"[最大涨幅计算] 未找到有效涨幅数据")
                logging.info(f"[最大涨幅计算] 股票代码: {stock_code}")
                logging.info(f"[最大涨幅计算] 信号日期: {signal_date}")
                logging.info(f"[最大涨幅计算] 使用了0根K线数据进行计算")
                logging.info(f"[最大涨幅计算] 搜索范围: {search_days}天")
                logging.info(f"[最大涨幅计算] 尝试次数: {retry_count}")
                return None, None, None, None
            
            # 初始化最大涨幅
            max_gain = 0.0
            best_low_idx = -1
            best_high_idx = -1
            
            # 遍历所有K线数据，找到最低点后计算后续K线中的最高点涨幅
            for i in range(len(lows)):
                if lows[i] <= 0:  # 跳过无效价格
                    continue
                
                # 计算从当前低点到所有后续K线（包括当前K线之后的所有K线）的最大涨幅
                current_low = lows[i]
                current_low_date = dates[i].strftime('%Y-%m-%d')
                low_flag = "[信号前]" if is_before_signal[i] else "[信号后]"
                
                # 查找i之后的所有K线中的最高价
                for j in range(i+1, len(highs)):
                    if highs[j] > 0:  # 只考虑有效最高价
                        gain = (highs[j] - current_low) / current_low * 100
                        high_flag = "[信号前]" if is_before_signal[j] else "[信号后]"
                        logging.info(f"[最大涨幅计算] 低点[{i+1}]{low_flag}({current_low_date}, {current_low:.2f}) 到 高点[{j+1}]{high_flag}({dates[j].strftime('%Y-%m-%d')}, {highs[j]:.2f}) 的涨幅: {gain:.2f}%")
                        if gain > max_gain:
                            max_gain = gain
                            best_low_idx = i
                            best_high_idx = j
            
            # 添加最佳涨幅信息的日志
            if best_low_idx >= 0 and best_high_idx >= 0:
                best_low_flag = "[信号前]" if is_before_signal[best_low_idx] else "[信号后]"
                best_high_flag = "[信号前]" if is_before_signal[best_high_idx] else "[信号后]"
                logging.info(f"[最大涨幅计算] 找到最大涨幅: {max_gain:.2f}% (低点[{best_low_idx+1}]{best_low_flag}: {dates[best_low_idx].strftime('%Y-%m-%d')} {lows[best_low_idx]:.2f}, 高点[{best_high_idx+1}]{best_high_flag}: {dates[best_high_idx].strftime('%Y-%m-%d')} {highs[best_high_idx]:.2f})")
            else:
                logging.info(f"[最大涨幅计算] 未找到有效的涨幅数据")
            
            # 增加详细的日志输出
            logging.info(f"[最大涨幅计算] 股票代码: {stock_code}")
            logging.info(f"[最大涨幅计算] 信号日期: {signal_date}")
            logging.info(f"[最大涨幅计算] 使用了{len(lows)}根K线数据进行计算")
            logging.info(f"[最大涨幅计算] 搜索范围: {search_days}天")
            logging.info(f"[最大涨幅计算] 重试次数: {retry_count}")
            logging.info(f"[最大涨幅计算] 有效低价(low>0)数量: {valid_low_count}, 无效低价数量: {invalid_low_count}")
            logging.info(f"[最大涨幅计算] 最终最高涨幅计算结果: {max_gain:.2f}%")
            
            # 同时保留打印输出以便控制台查看
            print(f"\n股票代码: {stock_code}")
            print(f"信号日期: {signal_date}")
            print(f"使用了{len(lows)}根K线数据进行计算")
            print(f"搜索范围: {search_days}天")
            print(f"重试次数: {retry_count}")
            print(f"价格统计 - 有效低价(low>0)数量: {valid_low_count}, 无效低价数量: {invalid_low_count}")
            if best_low_idx >= 0 and best_high_idx >= 0:
                best_low_flag = "[信号前]" if is_before_signal[best_low_idx] else "[信号后]"
                best_high_flag = "[信号前]" if is_before_signal[best_high_idx] else "[信号后]"
                print(f"最大涨幅: {max_gain:.2f}% (低点{best_low_flag}: {dates[best_low_idx].strftime('%Y-%m-%d')} {lows[best_low_idx]:.2f}, 高点{best_high_flag}: {dates[best_high_idx].strftime('%Y-%m-%d')} {highs[best_high_idx]:.2f})")
            else:
                print(f"未找到有效的涨幅数据")
            print(f"最终最高涨幅计算结果: {max_gain:.2f}%")
            print("-" * 50)
            
            # 返回最大涨幅百分比，保留两位小数
            return round(max_gain, 2)
                
        except Exception as e:
            logging.error(f"计算股票{stock_code}最大涨幅时发生错误: {str(e)}")
            print(f"计算股票{stock_code}最大涨幅时发生错误: {str(e)}")
            return None
    
    def run(self):
        """运行整个检测流程"""
        try:
            logging.info("开始运行股票检测流程...")
            
            # 初始化
            logging.info("初始化BAOSTOCK连接...")
            if not self.initialize():
                logging.error("BAOSTOCK初始化失败，程序退出")
                return
            logging.info("BAOSTOCK连接初始化成功")
            
            # 加载股票列表
            logging.info("开始加载股票列表...")
            success = self.load_stock_list()
            if success:
                logging.info(f"股票列表加载成功，共加载{len(self.stock_list)}支股票")
            else:
                logging.warning("股票列表加载可能不完整")
            
            # 扫描股票
            logging.info("开始扫描股票检测交易信号...")
            self.scan_all_stocks()
            
            # 保存结果
            logging.info("开始保存检测结果...")
            self.save_signal_results()
            logging.info("检测结果保存完成")
            
        except Exception as e:
            logging.error(f"运行过程中出错: {e}")
            import traceback
            logging.error(f"错误详情: {traceback.format_exc()}")
        finally:
            # 登出
            bs.logout()
            logging.info("\n已断开BAOSTOCK连接")



# 主程序入口
if __name__ == "__main__":
    # 初始化日志系统
    log_file = setup_logging()
    logging.info("=== 股票当日涨跌幅信号检测系统 ===")
    logging.info("本系统使用BAOSTOCK获取数据，检测股票当日涨跌幅是否小于-14%")
    logging.info(f"日志已保存到: {log_file}")
    
    # 检查命令行参数（可覆盖全局配置）
    logging.info(f"全局配置USE_ALL_STOCKS值: {USE_ALL_STOCKS}")
    use_all_stocks = USE_ALL_STOCKS  # 默认使用全局配置
    specified_stock = None
    specified_stock_name = None
    
    # 首先检查全局配置是否包含股票代码（数字）
    if any(char.isdigit() for char in use_all_stocks) and use_all_stocks not in ["是", "否"]:
        # 尝试从全局配置中提取股票代码和名称
        parts = use_all_stocks.strip().split()
        if parts:
            # 第一个部分作为股票代码
            code_part = parts[0]
            # 剩余部分作为股票名称
            name_part = ' '.join(parts[1:]) if len(parts) > 1 else "未知"
            
            logging.info(f"从全局配置检测到股票信息: 代码={code_part}, 名称={name_part}")
            use_all_stocks = f"STOCK:{code_part}:{name_part}"
    
    # 命令行参数可以覆盖全局配置
    if len(sys.argv) > 1:
        user_input = sys.argv[1].strip()
        logging.info(f"命令行输入: {user_input}")
        
        # 检查是否为测试模式
        if user_input.lower() == "否" or user_input.lower() == "n" or user_input.lower() == "test":
            use_all_stocks = "否"
        # 检查是否为股票代码格式（包含数字和可能的点）
        elif any(char.isdigit() for char in user_input):
            # 处理股票代码格式
            specified_stock = user_input
            specified_stock_name = sys.argv[2].strip() if len(sys.argv) > 2 else "未知"
            logging.info(f"指定了股票代码: {specified_stock}, 股票名称: {specified_stock_name}")
            use_all_stocks = f"STOCK:{specified_stock}:{specified_stock_name}"  # 使用特殊格式标记
    
    logging.info(f"最终使用的use_all_stocks值: {use_all_stocks}")
    
    # 确定运行模式
    if use_all_stocks.startswith("STOCK:"):
        stock_info = use_all_stocks.split(":")
        logging.info(f"运行模式: 单股票模式 - 股票代码: {stock_info[1]}, 股票名称: {stock_info[2]}")
    else:
        logging.info(f"运行模式: {'全量模式' if use_all_stocks == '是' else '测试模式（前20支股票）'}")
    
    # 检查BAOSTOCK是否已安装
    try:
        import baostock
        
        # 创建并运行检测器，传入模式参数
        detector = TradingSignalDetector(use_all_stocks=use_all_stocks)
        detector.run()
        
        logging.info("\n使用说明：")
        logging.info("  - 全量模式（默认）: python BAOSTOCK单日下跌14%.py")
        logging.info("  - 测试模式: python BAOSTOCK单日下跌14%.py 否")
        logging.info("  - 测试模式: python BAOSTOCK单日下跌14%.py n")
        logging.info("  - 测试模式: python BAOSTOCK单日下跌14%.py test")
        logging.info("  - 单股票模式: python BAOSTOCK单日下跌14%.py 股票代码 股票名称")
        
    except ImportError:
        logging.error("错误: 未安装BAOSTOCK库")
        logging.error("请先安装: pip install baostock")
        logging.info("\n使用说明：")
        logging.info("  - 全量模式（默认）: python BAOSTOCK单日下跌14%.py")
        logging.info("  - 测试模式: python BAOSTOCK单日下跌14%.py 否")
        logging.info("  - 测试模式: python BAOSTOCK单日下跌14%.py n")
        logging.info("  - 测试模式: python BAOSTOCK单日下跌14%.py test")
        logging.info("  - 单股票模式: python BAOSTOCK单日下跌14%.py 股票代码 股票名称")