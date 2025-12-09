import sys
import subprocess
import time
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd

# 查询开关：将对应值改为 False 可跳过该模块的所有请求
def _flag(value: str) -> bool:
    return value in {"是", "true", "True", "1", True}


FEATURE_FLAGS = {
    "获取价格": _flag("是"),
    "获取流通股本": _flag("是"),
    "获取高管变动": _flag("否"),
    "获取一致行动人": _flag("否"),
    "获取上市信息": _flag("否"),
    "获取历史十大股东": _flag("否"),
    "获取十大流通股东": _flag("否"),
    "获取机构持仓": _flag("否"),
}

# 自动定位股票数据文件（默认使用同目录下的数据）
script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STOCK_DATA_FILE = "股票列表.xlsx"
excel2_path = os.path.join(script_dir, DEFAULT_STOCK_DATA_FILE)

if not os.path.exists(excel2_path):
    print(f"股票数据文件不存在: {excel2_path}")
    sys.exit(1)

try:
    df_stocks = pd.read_excel(excel2_path, sheet_name='Sheet1')
except Exception as e:
    print(f"❌❌❌❌ 读取股票列表失败: {str(e)}")
    sys.exit(1)

if '信号日期' not in df_stocks.columns:
    print("❌❌❌❌ 股票数据文件缺少“信号日期”列，无法推导目标日期")
    sys.exit(1)

signal_dates = pd.to_datetime(df_stocks['信号日期'], errors='coerce').dropna()
if signal_dates.empty:
    print("❌❌❌❌ “信号日期”列没有可用日期，无法推导目标日期")
    sys.exit(1)

reference_signal_date = signal_dates.max()
target_year = reference_signal_date.year - 1
if target_year < 1900:
    print("❌❌❌❌ 推导出的目标年份无效")
    sys.exit(1)

target_date = f"{target_year}1231"
system_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(script_dir, f"股票综合分析_{system_timestamp}.xlsx")

print(f"股票数据文件: {excel2_path}")
print(f"参考信号日期: {reference_signal_date.strftime('%Y-%m-%d')}")
print(f"目标日期: {target_date} (信号日期前一年末)")
print(f"结果文件: {save_path}")

try:
    import akshare as ak
    print("akshare 已成功导入")
except ImportError:
    print("未找到 akshare 模块，正在安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "akshare"])
        import akshare as ak
        print("akshare 安装成功并导入")
    except Exception as e:
        print(f"安装 akshare 失败: {str(e)}")
        sys.exit(1)

from akshare_client import AkShareClient
from akshare_config import DEFAULT_CLIENT_CONFIG

client_config = DEFAULT_CLIENT_CONFIG
client = AkShareClient(client_config)
stock_info_dict_cache = {}
stock_info_lock = threading.Lock()
price_main_disabled = False
CNINFO_SHARE_START_DATE = "19900101"
cninfo_share_cache = {}
cninfo_share_failures = set()
cninfo_share_lock = threading.Lock()


def get_stock_info_dict(stock_code: str):
    stock_code = str(stock_code).zfill(6)
    with stock_info_lock:
        cached = stock_info_dict_cache.get(stock_code)
    if cached is not None:
        return cached
    info_value = {}
    try:
        info_df = client.stock_individual_info(stock_code)
        if isinstance(info_df, pd.DataFrame) and not info_df.empty:
            info_value = dict(zip(info_df["item"], info_df["value"]))
    except Exception as exc:
        print(f"⚠️ 获取股票信息失败: {stock_code} - {exc}")
    with stock_info_lock:
        stock_info_dict_cache[stock_code] = info_value
    return info_value


def prefetch_stock_info(codes):
    unique_codes = list({str(code).zfill(6) for code in codes})
    if not unique_codes:
        return
    print(f"🧵 正在并行预热 {len(unique_codes)} 只股票的基础信息...")

    def worker(code: str):
        get_stock_info_dict(code)

    max_workers = min(8, len(unique_codes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, code) for code in unique_codes]
        for future in as_completed(futures):
            future.result()


def ensure_cninfo_share_data(stock_code: str):
    stock_code = str(stock_code).zfill(6)
    with cninfo_share_lock:
        cached = cninfo_share_cache.get(stock_code)
    if cached is not None:
        return cached
    with cninfo_share_lock:
        if stock_code in cninfo_share_failures:
            return None
    start_date = CNINFO_SHARE_START_DATE
    end_date = datetime.now().strftime("%Y%m%d")
    try:
        print(f"  ⏳ 获取CNInfo股本数据: {stock_code} ({start_date}-{end_date})")
        df = client.stock_share_change(stock_code, start_date, end_date)
    except Exception as exc:
        print(f"⚠️ CNInfo股本数据请求失败 ({stock_code}): {exc}")
        with cninfo_share_lock:
            cninfo_share_failures.add(stock_code)
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"⚠️ CNInfo股本数据为空 ({stock_code})")
        with cninfo_share_lock:
            cninfo_share_failures.add(stock_code)
        return None
    df = df.copy()
    df["变动日期"] = pd.to_datetime(df.get("变动日期"), errors="coerce")
    df["已流通股份"] = pd.to_numeric(df.get("已流通股份"), errors="coerce")
    df.sort_values("变动日期", inplace=True)
    with cninfo_share_lock:
        cninfo_share_cache[stock_code] = df
    print(f"✅ CNInfo股本数据缓存完成: {stock_code} 共 {len(df)} 条记录")
    return df


def get_cninfo_circulating_shares(stock_code: str, reference_date: str):
    df = ensure_cninfo_share_data(stock_code)
    if df is None or df.empty:
        return None
    try:
        ref_datetime = pd.to_datetime(reference_date, errors="coerce")
    except Exception:
        ref_datetime = None
    if pd.isna(ref_datetime):
        print(f"⚠️ 无法解析参考日期 {reference_date}，跳过CNInfo历史匹配")
        return None
    df_valid = df.dropna(subset=["已流通股份"])
    if df_valid.empty:
        print(f"⚠️ CNInfo记录缺少已流通股份字段 ({stock_code})")
        return None
    df_with_dates = df_valid[df_valid["变动日期"].notna()]
    df_before = df_with_dates[df_with_dates["变动日期"] <= ref_datetime]
    if not df_before.empty:
        row = df_before.iloc[-1]
        out_of_range = False
    else:
        if df_with_dates.empty:
            print(f"⚠️ CNInfo记录缺少有效变动日期 ({stock_code})")
            return None
        row = df_with_dates.iloc[0]
        out_of_range = True
        print(f"⚠️ 信号日 {reference_date} 早于CNInfo首条记录 ({stock_code})，使用最早一次变动数据 {row['变动日期'].date()}")
    shares = row["已流通股份"]
    if pd.isna(shares):
        return None
    # CNInfo 返回单位为“万股”，换算成股
    shares = float(shares) * 10000
    change_date = row["变动日期"]
    change_date_str = change_date.strftime("%Y%m%d") if pd.notna(change_date) else ""
    if out_of_range:
        print(f"   ⮕ CNInfo最近一次变动日期: {change_date_str}")
    return float(shares), change_date_str


def get_circulating_shares_from_stock_info(stock_code: str):
    info_dict = get_stock_info_dict(stock_code)
    for key, value in info_dict.items():
        if "流通股" in str(key):
            value_str = str(value).replace(",", "")
            try:
                if "亿" in value_str:
                    return float(value_str.replace("亿", "")) * 100000000
                return float(value_str)
            except ValueError:
                continue
    return None


def get_circulating_shares(stock_code: str, reference_date: str):
    if not FEATURE_FLAGS["获取流通股本"]:
        return None
    cninfo_result = get_cninfo_circulating_shares(stock_code, reference_date)
    if cninfo_result:
        shares, change_date_str = cninfo_result
        return shares, change_date_str, "CNInfo"
    fallback = get_circulating_shares_from_stock_info(stock_code)
    if fallback is not None:
        return fallback, "", "东方财富"
    return None


def build_shared_maps(target_dates, hold_types):
    yzxdr_map = {}
    if FEATURE_FLAGS["获取一致行动人"]:
        for target_date in target_dates:
            temp_map = defaultdict(set)
            try:
                yzxdr_df = client.stock_yzxdr(target_date)
                if isinstance(yzxdr_df, pd.DataFrame) and not yzxdr_df.empty:
                    for _, row in yzxdr_df.iterrows():
                        code = row.get("股票代码")
                        names = row.get("一致行动人")
                        if code and isinstance(names, str):
                            for name in names.split(","):
                                cleaned = name.strip()
                                if cleaned:
                                    temp_map[code].add(cleaned)
            except Exception as exc:
                print(f"⚠️ 预获取一致行动人失败 (日期: {target_date}): {exc}")
            yzxdr_map[target_date] = temp_map
    else:
        for target_date in target_dates:
            yzxdr_map[target_date] = defaultdict(set)

    fund_hold_map = {}
    if FEATURE_FLAGS["获取机构持仓"]:
        for target_date in target_dates:
            fund_hold_map[target_date] = {}
            for hold_type in hold_types:
                fund_hold_map[target_date][hold_type] = {}
                try:
                    df_hold = client.stock_report_fund_hold(hold_type, target_date)
                    if (
                        isinstance(df_hold, pd.DataFrame)
                        and not df_hold.empty
                        and "股票代码" in df_hold.columns
                    ):
                        for _, row in df_hold.iterrows():
                            code = str(row.get("股票代码", "")).strip().zfill(6)
                            if code:
                                fund_hold_map[target_date][hold_type][code] = row
                except Exception as exc:
                    print(f"⚠️ 预获取 {hold_type} 数据失败 (日期: {target_date}): {exc}")
    else:
        for target_date in target_dates:
            fund_hold_map[target_date] = {hold: {} for hold in hold_types}
    return yzxdr_map, fund_hold_map

def fetch_price_from_backup(stock_code_str: str, price_date: str):
    """使用备用接口（新浪）获取指定日期的收盘价"""
    try:
        prefix = "sh" if stock_code_str.startswith("6") else "sz"
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{stock_code_str}")
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        target_datetime = pd.to_datetime(price_date)
        available = df[df["date"] <= target_datetime]
        if available.empty:
            return None
        row = available.iloc[-1]
        return float(row["close"]), row["date"]
    except Exception as exc:
        print(f"⚠️ 备用价格接口失败 ({stock_code_str}): {exc}")
        return None


def get_circ_mv_on_date(stock_code, price_date):
    """获取指定股票在信号当日（或该日最近交易日）收盘价对应的流通市值"""
    stock_code_str = str(stock_code).zfill(6)

    closing_price = None
    used_date = None

    global price_main_disabled

    main_price_available = FEATURE_FLAGS["获取价格"] and not price_main_disabled

    fallback_due_to_connection = False

    try:
        if main_price_available:
            start_date = price_date[:6] + "01"
            hist_data = client.stock_hist(symbol=stock_code_str, start_date=start_date, end_date=price_date)

            if hist_data.empty:
                raise ValueError("主行情接口无数据")

            hist_data['日期'] = pd.to_datetime(hist_data['日期'])
            hist_data.set_index('日期', inplace=True)

            target_datetime = pd.to_datetime(price_date)
            if target_datetime not in hist_data.index:
                print(f"⚠️ 在历史数据中未找到日期 {price_date} 的记录，尝试获取前一个交易日数据 (股票: {stock_code_str})")
                prev_trading_day = hist_data.index[hist_data.index < target_datetime].max()
                closing_price = hist_data.loc[prev_trading_day]['收盘']
                used_date = prev_trading_day
                print(f"✅ 使用前一个交易日 {prev_trading_day.strftime('%Y%m%d')} 的收盘价: {closing_price:.2f} 元")
            else:
                closing_price = hist_data.loc[target_datetime]['收盘']
                used_date = target_datetime
                print(f"✅ 获取收盘价成功: {stock_code_str} 在 {price_date} 的收盘价为 {closing_price:.2f} 元")
        else:
            raise RuntimeError("主行情接口已禁用")

    except Exception as e:
        if "RemoteDisconnected" in str(e) or "Connection aborted" in str(e):
            price_main_disabled = True
            print("🛑 检测到东方财富行情接口被封，后续将直接使用备用接口。")
            fallback_due_to_connection = True
        else:
            print(f"⚠️ 主行情接口失败 (股票: {stock_code_str}): {str(e)}，尝试备用接口")
        backup = fetch_price_from_backup(stock_code_str, price_date)
        if backup is None:
            print(f"⚠️ 备用接口也无法获取价格 (股票: {stock_code_str})")
            return None
        closing_price, backup_date = backup
        used_date = pd.to_datetime(price_date) if fallback_due_to_connection else backup_date
        backup_msg_date = backup_date.strftime('%Y%m%d')
        print(f"✅ 备用接口获取收盘价成功: {stock_code_str} 在 {backup_msg_date} 的收盘价为 {closing_price:.2f} 元")
        if fallback_due_to_connection and backup_msg_date != price_date:
            print(f"   ⮕ 为保持信号日期一致，结果中仍使用信号日 {price_date}")

    circ_info = get_circulating_shares(stock_code_str, price_date)
    if circ_info is None:
        print(f"⚠️ 未找到流通股本数据 (股票: {stock_code_str})")
        return None

    circ_shares, share_date_str, share_source = circ_info
    if share_source == "CNInfo":
        date_display = share_date_str or "未知日期"
        print(f"✅ CNInfo流通股本匹配成功: {stock_code_str} 在 {date_display} 的流通股本为 {circ_shares:,.0f} 股")
    else:
        print(f"✅ 获取流通股本成功(东方财富): {stock_code_str} 的流通股本为 {circ_shares:,.0f} 股")
    circulating_mv = circ_shares * closing_price
    used_date_str = used_date.strftime("%Y%m%d")
    print(f"✅ 计算流通市值成功: {stock_code_str} 在 {used_date_str} 的流通市值为 {circulating_mv:,.2f} 元")
    return circulating_mv, closing_price, used_date_str

# 从EXCEL读取股票数据
try:
    print(f"✅ 成功读取股票列表: 共 {len(df_stocks)} 只股票")
    
    stock_entries = []
    for _, row in df_stocks.iterrows():
        name = str(row['股票名称']).strip()
        code = str(row['股票代码']).strip().zfill(6)
        code_with_prefix = f"sz{code}" if code.startswith('0') or code.startswith('3') else f"sh{code}"
        signal_date_val = row.get('信号日期')
        signal_dt = pd.to_datetime(signal_date_val, errors='coerce')
        if pd.isna(signal_dt):
            continue
        target_year_for_stock = signal_dt.year - 1
        if target_year_for_stock < 1900:
            entry_target_date = target_date
        else:
            entry_target_date = f"{target_year_for_stock}1231"
        stock_entries.append({
            "name": name,
            "code": code,
            "code_with_prefix": code_with_prefix,
            "signal_date": signal_dt.strftime("%Y%m%d"),
            "target_date": entry_target_date,
        })

    if not stock_entries:
        print("❌❌❌❌ 输入表内无有效信号记录")
        sys.exit(1)

    sample_names = [entry['name'] for entry in stock_entries[:3]]
    print(f"✅ 已加载股票列表: {sample_names}...等 {len(stock_entries)} 条记录")

    if client_config.enable_async_prefetch:
        prefetch_stock_info([entry["code"] for entry in stock_entries])
    
except Exception as e:
    print(f"❌❌❌❌ 处理股票列表失败: {str(e)}")
    sys.exit(1)

categories = ["其它", "投资公司", "私募基金", "集合理财计划", "其他理财产品", "员工持股计划"]
hold_types = ["信托持仓", "社保持仓", "QFII持仓", "保险持仓", "基金持仓", "券商持仓"]

default_target_date = target_date
target_dates_for_fetch = sorted({entry["target_date"] for entry in stock_entries} | {default_target_date})

result_list = []
cache = {}
start_time = datetime.now()
total_stocks = len(stock_entries)
print(f"⏱⏱⏱⏱⏱⏱⏱⏱⏱️ 开始执行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊📊📊📊 共需处理 {total_stocks} 条股票-信号记录")

print("🔄 正在预获取共享数据...")
shared_yzxdr_map, fund_hold_map = build_shared_maps(target_dates_for_fetch, hold_types)
print("✅ 共享数据预获取完成")

processed_keys = set()
if os.path.exists(save_path):
    try:
        existing_df = pd.read_excel(save_path)
        if '数据日期' in existing_df.columns:
            names = existing_df['股票名称'].astype(str)
            dates = existing_df['数据日期'].astype(str).fillna(default_target_date)
            if '原始信号日期' in existing_df.columns:
                signal_series = existing_df['原始信号日期'].astype(str).fillna("")
            elif '信号当日日期' in existing_df.columns:
                signal_series = existing_df['信号当日日期'].astype(str).fillna("")
            else:
                signal_series = pd.Series([''] * len(existing_df))
            processed_keys = set(zip(names, dates, signal_series))
        else:
            processed_keys = {(str(name), default_target_date, "") for name in existing_df['股票名称'].tolist()}
        print(f"✅ 发现已有结果文件，已处理 {len(processed_keys)} 条记录")
    except Exception as e:
        print(f"⚠️ 读取现有结果文件失败: {str(e)}")

stocks_to_process = [
    entry
    for entry in stock_entries
    if (entry["name"], entry["target_date"], entry["signal_date"]) not in processed_keys
]
print(f"📊📊📊📊 需要处理 {len(stocks_to_process)} 条新记录")

for idx, entry in enumerate(stocks_to_process, 1):
    stock_name = entry["name"]
    stock_code_pure = entry["code"]
    stock_code_with_prefix = entry["code_with_prefix"]
    entry_signal_date = entry.get("signal_date")
    print(f"\n🔍🔍🔍🔍 [{idx}/{len(stocks_to_process)}] 正在处理股票: {stock_name}")
    
    cache_key = (stock_name, entry["target_date"], entry_signal_date)
    if cache_key in cache:
        cached_row = cache[cache_key].copy()
        if entry_signal_date:
            cached_row['原始信号日期'] = entry_signal_date
        result_list.append(cached_row)
        print(f"✅ 使用缓存数据: {stock_name} - {entry['target_date']} - {entry_signal_date}")
        continue

    current_target_date = entry.get("target_date") or default_target_date
    if current_target_date not in shared_yzxdr_map:
        current_target_date = default_target_date

    # 获取流通市值
    price_date = entry_signal_date or current_target_date
    signal_price = None
    signal_price_date = None
    if FEATURE_FLAGS["获取价格"]:
        price_info = get_circ_mv_on_date(stock_code_pure, price_date)
        if price_info is None:
            print(f"⚠️ 无法获取流通市值，使用默认值0 (股票: {stock_name})")
            circulating_mv = 0
        else:
            circulating_mv, signal_price, signal_price_date = price_info
    else:
        print(f"⚠️ 已关闭流通市值查询，使用默认值0 (股票: {stock_name})")
        circulating_mv = 0

    # 高管持股变动查询
    insider_names = set()
    if FEATURE_FLAGS["获取高管变动"]:
        try:
            print("  正在查询高管持股变动...")
            exec_df = client.stock_management_change(stock_code_with_prefix[2:])
            if isinstance(exec_df, pd.DataFrame) and not exec_df.empty:
                insider_names = set(exec_df["变动人"].tolist())
                print(f"✅ 高管持股变动数据查询成功: {stock_name}")
            else:
                print(f"⚠️ 未查询到高管持股变动数据: {stock_name}")
        except Exception as e:
            print(f"⚠️ 高管持股变动查询失败 (股票: {stock_name}): {str(e)}")
    else:
        print("⚠️ 已关闭高管持股变动查询")

    # 一致行动人查询
    if FEATURE_FLAGS["获取一致行动人"]:
        yzxdr_names = shared_yzxdr_map.get(current_target_date, {}).get(stock_code_with_prefix, set())
        if yzxdr_names:
            print(f"✅ 一致行动人数据命中缓存: {stock_name}")
            insider_names.update(yzxdr_names)
        else:
            print(f"⚠️ 未查询到一致行动人数据 (股票: {stock_name})")
    else:
        yzxdr_names = set()
        print("⚠️ 已关闭一致行动人查询")

    # 获取上市时间
    OLDDATE = "20191231"
    if FEATURE_FLAGS["获取上市信息"]:
        try:
            print("  正在获取上市时间...")
            info_dict = get_stock_info_dict(stock_code_pure)
            list_date = str(info_dict.get("上市时间", "")).strip()
            if list_date:
                try:
                    list_date_obj = datetime.strptime(list_date, "%Y-%m-%d")
                except ValueError:
                    try:
                        list_date_obj = datetime.strptime(list_date, "%Y%m%d")
                    except ValueError:
                        list_date_obj = None
                        print(f"⚠️ 无法解析上市时间格式: {list_date}，使用默认值 {OLDDATE}")
                if list_date_obj:
                    query_date_obj = datetime.strptime(current_target_date, "%Y%m%d")
                    time_difference = (query_date_obj - list_date_obj).days / 365.25
                    OLDDATE = "20191231" if time_difference > 5 else f"{list_date_obj.year}1231"
                    print(f"✅ 获取上市时间成功: {stock_name} 上市于 {list_date}，使用历史日期 {OLDDATE}")
            else:
                print(f"⚠️ 未找到上市时间字段，使用默认值: {OLDDATE}")
        except Exception as e:
            print(f"⚠️ 获取上市时间失败: {str(e)}，使用默认值: {OLDDATE}")
    else:
        print("⚠️ 已关闭上市时间查询，使用默认历史日期")

    # 历史十大股东查询
    historical_insiders = set()
    if FEATURE_FLAGS["获取历史十大股东"]:
        try:
            print("  正在查询历史十大股东...")
            historical_df = client.stock_top10_history(stock_code_with_prefix, OLDDATE)
            if isinstance(historical_df, pd.DataFrame) and not historical_df.empty:
                historical_insiders = set(historical_df["股东名称"].tolist())
                print(f"✅ 历史十大股东查询成功: {stock_name} (日期: {OLDDATE})")
            else:
                print(f"⚠️ 未查询到历史十大股东数据 (股票: {stock_name})")
        except Exception as e:
            print(f"⚠️ 历史十大股东查询失败 (股票: {stock_name}): {str(e)}")
    else:
        print("⚠️ 已关闭历史十大股东查询")

    # 十大流通股东处理
    category_ratios = {category: 0.0 for category in categories}
    hk_ratio = enterprise_ratio = insider_ratio = retail_ratio = 0.0
    small_non_ratio = small_non_enterprise_ratio = 0.0
    
    # 代码2中的新增指标
    top10_total_ratio = 0.0
    top10_small_non_ratio = 0.0
    top10_large_non_ratio = 0.0
    institutional_large_non_ratio = 0.0
    institutional_small_non_ratio = 0.0

    if FEATURE_FLAGS["获取十大流通股东"]:
        try:
            print("  正在查询十大流通股东...")
            df = client.stock_top10_free(stock_code_with_prefix, current_target_date)
            required_cols = {'股东名称', '股东性质', '占总流通股本持股比例'}
            if df.empty:
                print(f"⚠️ 未获取到十大流通股东数据 (股票: {stock_name})")
            elif not required_cols.issubset(set(df.columns)):
                print(f"⚠️ 十大流通股东数据列缺失 (股票: {stock_name})，列: {list(df.columns)}")
            else:
                for _, row in df.iterrows():
                    holder_name = row['股东名称']
                    holder_type = row.get('股东性质', '')
                    holding_ratio = row['占总流通股本持股比例']
                    
                    # 代码1原有逻辑
                    if holder_type == '个人':
                        if holding_ratio > 10 or holder_name in historical_insiders or holder_name in insider_names:
                            if holding_ratio < 5:
                                small_non_ratio += holding_ratio
                            else:
                                insider_ratio += holding_ratio
                        else:
                            retail_ratio += holding_ratio
                    elif holder_type == '投资公司':
                        if holding_ratio > 20.0 or any(keyword in holder_name for keyword in ['国有', '国资']) or (stock_name[:2] in holder_name) or (stock_name in holder_name):
                            enterprise_ratio += holding_ratio
                        else:
                            category_ratios['投资公司'] += holding_ratio
                    elif holder_type == '其它':
                        if '香港中央结算' in holder_name:
                            hk_ratio += holding_ratio
                        elif holding_ratio > 20.0:
                            enterprise_ratio += holding_ratio
                        elif any(keyword in holder_name for keyword in ['公司', '国有']) or (stock_name[:2] in holder_name) or (stock_name in holder_name):
                            if holding_ratio < 5:
                                small_non_enterprise_ratio += holding_ratio
                            else:
                                enterprise_ratio += holding_ratio
                        else:
                            category_ratios['其它'] += holding_ratio
                    elif holder_type in categories:
                        category_ratios[holder_type] += holding_ratio
                    
                    # 代码2新增逻辑
                    top10_total_ratio += holding_ratio
                    
                    if holding_ratio < 5.0:
                        top10_small_non_ratio += holding_ratio
                    else:
                        top10_large_non_ratio += holding_ratio
                    
                    is_individual = "个人" in str(holder_type) or "自然人" in str(holder_name)
                    is_hk_central = "香港中央结算" in str(holder_name)
                    
                    if not is_individual and not is_hk_central:
                        if holding_ratio >= 5.0:
                            institutional_large_non_ratio += holding_ratio
                        else:
                            institutional_small_non_ratio += holding_ratio
                
                print(f"✅ 十大流通股东数据处理完成: {stock_name}")
                print(f"✅ 新增指标 - 前10大流通股东持股比例合计: {top10_total_ratio:.1f}%")
                print(f"✅ 新增指标 - 十大流通股东小非合计: {top10_small_non_ratio:.1f}%")
                print(f"✅ 新增指标 - 十大流通股东大非合计: {top10_large_non_ratio:.1f}%")
                print(f"✅ 新增指标 - 十大流通机构大非: {institutional_large_non_ratio:.1f}%")
                print(f"✅ 新增指标 - 十大流通机构小非: {institutional_small_non_ratio:.1f}%")
        except Exception as e:
            print(f"❌❌❌❌ 十大流通股东数据处理失败 (股票: {stock_name}): {str(e)}")
    else:
        print("⚠️ 已关闭十大流通股东查询")

    # 结果格式化
    result_data = {
        '股票名称': stock_name,
        '数据日期': current_target_date,
        '信号当日流通市值(元)': f'{circulating_mv:,.0f}',
        '信号当日价格': f'{signal_price:.2f}' if signal_price is not None else 'N/A',
        '信号当日日期': signal_price_date or price_date,
        '原始信号日期': entry.get("signal_date"),
        '高管/大股东持股比例': f'{insider_ratio:.1f}%',
        '高管/大股东持股比例（小非）': f'{small_non_ratio:.1f}%',
        '普通散户持股比例': f'{retail_ratio:.1f}%',
        '香港中央结算': f'{hk_ratio:.1f}%',
        '企业大股东（包含国资）': f'{enterprise_ratio:.1f}%',
        '企业大股东（包含国资）（小非）': f'{small_non_enterprise_ratio:.1f}%',
        **{k: f'{v:.1f}%' for k, v in category_ratios.items()},
        # 代码2新增字段
        '前10大流通股东持股比例合计': f'{top10_total_ratio:.1f}%',
        '十大流通股东小非合计': f'{top10_small_non_ratio:.1f}%',
        '十大流通股东大非合计': f'{top10_large_non_ratio:.1f}%',
        '十大流通机构大非': f'{institutional_large_non_ratio:.1f}%',
        '十大流通机构小非': f'{institutional_small_non_ratio:.1f}%'
    }
    
    # 查询持仓数据
    if FEATURE_FLAGS["获取机构持仓"]:
        print("  正在查询持仓数据...")
        for hold_type in hold_types:
            result_data[hold_type + "占比"] = "N/A"
        result_data["持有基金家数"] = 0

        for hold_type in hold_types:
            stock_row = (
                fund_hold_map.get(current_target_date, {})
                .get(hold_type, {})
                .get(stock_code_pure)
            )
            if stock_row is None or (isinstance(stock_row, pd.Series) and stock_row.empty):
                result_data[hold_type + "占比"] = "0.0%"
                continue

            market_value = stock_row.get("持股市值", 0)
            if market_value and circulating_mv > 0:
                hold_ratio = (market_value / circulating_mv) * 100
                result_data[hold_type + "占比"] = f"{hold_ratio:.1f}%"
                print(f"    ✅ {hold_type}占比: {hold_ratio:.1f}%")
            else:
                result_data[hold_type + "占比"] = "N/A"
                print(f"    ⚠⚠⚠⚠⚠️ 无法计算{hold_type}占比 (股票: {stock_name})")

            if hold_type == "基金持仓":
                fund_hold_count = stock_row.get("持有基金家数", 0)
                result_data["持有基金家数"] = int(fund_hold_count) if pd.notna(fund_hold_count) else 0
    else:
        print("⚠️ 已关闭持仓数据查询")
        for hold_type in hold_types:
            result_data[hold_type + "占比"] = "N/A"
        result_data["持有基金家数"] = 0
    
    result_list.append(result_data)
    cache[cache_key] = result_data.copy()
    print(f"✅ 股票处理完成: {stock_name}")
    
    # 每处理10个股票自动保存一次（降低频率以减少中断损失）
    if idx % 10 == 0:
        print(f"\n💾💾💾💾 已处理 {idx} 只股票，正在保存中间结果...")
        try:
            # 读取现有结果（如果有）
            if os.path.exists(save_path):
                existing_df = pd.read_excel(save_path)
                new_df = pd.concat([existing_df, pd.DataFrame(result_list)], ignore_index=True)
            else:
                new_df = pd.DataFrame(result_list)
                
            new_df.to_excel(save_path, index=False)
            print(f"✅ 中间结果已保存至: {save_path}")
            result_list = []  # 清空结果列表
            print(f"📈 当前请求统计: {client.metrics()}")
        except Exception as e:
            print(f"❌❌❌❌ 保存中间结果失败: {str(e)}")

# 保存最终结果
print("\n💾💾💾💾 正在保存最终结果...")
try:
    # 读取现有结果（如果有）
    if os.path.exists(save_path) and len(result_list) > 0:
        existing_df = pd.read_excel(save_path)
        final_df = pd.concat([existing_df, pd.DataFrame(result_list)], ignore_index=True)
    elif len(result_list) > 0:
        final_df = pd.DataFrame(result_list)
    else:
        print("⚠️ 没有新数据需要保存")
        sys.exit(0)
        
    final_df.to_excel(save_path, index=False)
    print(f"✅ 最终结果已保存至: {save_path}")
    print(f"📈 最终请求统计: {client.metrics()}")
except Exception as e:
    print(f"❌❌❌❌ 保存最终结果失败: {str(e)}")
    sys.exit(1)

# 输出执行时间
end_time = datetime.now()
total_time = (end_time - start_time).seconds
minutes, seconds = divmod(total_time, 60)
print(f"⏱⏱⏱⏱⏱⏱⏱⏱⏱️ 结束执行时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱⏱⏱⏱⏱⏱⏱⏱⏱️ 总耗时: {minutes}分{seconds}秒")
print(f"📊📊📊📊 成功处理 {len(stocks_to_process)} 条记录")
