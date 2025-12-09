# yinzifenxi1119.py 因子分析代码结构说明

## 1. 文件整体定位
- 用途: 针对创业板单日大跌樣本, 對多個因子做表現評估與策略報告
- 輸入: 默認 Excel 文件 'DEFAULT_DATA_FILE' (創業板單日下跌 4% 數據清理後)
- 輸出: 日誌 txt, 因子分析匯總 csv, 詳細分析 txt, 帶參數因子分析 txt 及多個 csv
- 主要角色: FactorAnalysis 負責因子層分析; ParameterizedFactorAnalyzer 負責參數區間層分析
- 入口: main 函數, 串起完整分析流程

## 2. 頂層配置與工具
- 模塊導入
  - 標準庫: sys, os, warnings, datetime
  - 數據與統計: pandas, numpy, scipy
  - 畫圖: matplotlib, 以及按需導入 seaborn
- 全局常量與開關
  - DEFAULT_DATA_FILE: 默認數據文件路徑
  - 啟動時檢查文件是否存在, 不存在則直接退出
  - HAS_SCIPY, HAS_PLOT: 根據庫是否可用控制後續統計與畫圖功能
- Logger 類
  - 作用: 將所有標準輸出重定向到帶時間戳的日誌文件 '因子分析日志_YYYYMMDD_HHMMSS.txt'
  - 核心屬性: log_file, 	erminal (保存原始 stdout)
  - 核心方法:
    - write: 同時寫入終端與文件
    - lush: 刷新輸出
    - close: 在文件尾部寫入結束時間, 並恢復系統 stdout
- 安全工具函數
  - ensure_list(obj, obj_name): 保證對象為 list, 否則打印警告並返回空列表
  - safe_len(obj, obj_name): 對 list, tuple, ndarray 安全求長度, 其他類型返回 0
  - 主要用在按日期循環計算 IC 的過程中, 避免出現 len of unsized object 錯誤

## 3. 全局統計與穩健性輔助函數
- 相關與檢驗
  - kendall_tau_corr(x, y): 自行實現 Kendall Tau 相關系數
  - obust_correlation(x, y, method): 基於中位數與 MAD 等方法的穩健相關
  - mann_whitney_u_test(x, y): 兩獨立樣本的非參數檢驗, 返回 U 值與 p 值
  - custom_spearman_corr(x, y): 不依賴 scipy 的 Spearman 相關計算, 使用 ank_with_ties 處理並列秩
- 置信區間與多重檢驗
  - ootstrap_confidence_interval(x, y, statistic, n_bootstrap, confidence_level):
    - 通過自助抽樣計算相關或平均差的置信區間
  - alse_discovery_control(p_values, method, alpha):
    - 實現 BH 與 BY 兩種 FDR 控制, 返回校正後 p 值與顯著結果數
- 異常值與敏感性分析
  - detect_outliers(x, method):
    - 支持 IQR, zscore, modified zscore 三種方法, 返回異常值掩碼與閾值
  - sensitivity_analysis(x, y, outlier_methods, include_outliers):
    - 對比包含與剔除異常值時的相關變化, 得到敏感性指標
  - olling_window_analysis(df, factor_col, return_col, window_sizes, compute_ic_decay, save_plots):
    - 按日期排序後做滾動窗口 IC 計算, 得到 IC 時序與穩定性指標
  - 	emporal_stability_analysis(factor_results):
    - 對 IC 序列做自相關與趨勢分析, 評估時序穩定性
  - sample_sensitivity_analysis(df, factor_col, return_col, sample_sizes, n_iterations):
    - 多次隨機抽樣不同樣本比例, 觀察 IC 分佈與跨樣本穩健性
- 年化收益相關
  - calculate_standard_annual_return(total_return_rate, observation_years, method):
    - 使用標準復利或對數方法計算年化收益, 並做數值穩定性檢查
  - safe_calculate_annual_return(total_return, years, method):
    - 包裝年化計算, 防止異常輸入導致報錯
  - alidate_annual_return_calculation(annual_return, observation_years, original_total_return, tolerance):
    - 反推總收益率並與原值對比, 給出絕對誤差與相對誤差等驗證信息

## 4. FactorAnalysis 類
### 4.1 核心屬性與初始化
- ile_path: 數據文件路徑, 默認為 DEFAULT_DATA_FILE
- data: 原始 DataFrame, 可由外部直接傳入
- actors: 預設要分析的因子列表, 包含
  - 信號發出時上市天數
  - 日最大跌幅百分比
  - 信號當日收盤漲跌
  - 信號後一日開盤漲跌幅
  - 次日開盤後總體下跌幅度
  - N 日最大漲幅
  - 當日回調
- eturn_col: 目標收益列, 固定為 '持股2日收益率'
- nalysis_results: 存放每個因子的 IC, IR, 多空收益等結果
- nomaly_stats: 統計缺失值, 異常值, 重複行, IC 計算中跳過樣本的原因等
- 初始化時:
  - 若 data 為空且 ile_path 非空, 自動調用 load_data 讀取數據

### 4.2 數據讀取與預處理
- load_data(self)
  - 根據後綴從 Excel 或 CSV 讀取數據
- pply_factor_processing(self, df, factor_col, method, winsorize, winsorize_limits)
  - 對單個因子做標準化, 歸一化或秩轉換, 可選縮尾處理
- preprocess_data(self, process_factors, factor_method, winsorize, winsorize_limits)
  - 功能: 對所有因子及目標收益做統一預處理
  - 核心步驟:
    - 檢查必要列是否存在: 股票代碼, 股票名稱, 信號日期, 回報列以及所有因子
    - 處理百分比字符串列: 去掉百分號, 轉為數值並除以 100
    - 將 '信號日期' 轉為日期類型
    - 對每個因子:
      - 嘗試轉為數值型並記錄原始類型
      - 統計缺失值並寫入 nomaly_stats
      - 若 process_factors 為真, 調用 pply_factor_processing 做標準化與縮尾
    - 最終得到 self.processed_data, 作為後續分析的主數據集

### 4.3 IC 計算
- calculate_ic(self, factor_col, use_pearson, use_robust_corr, use_kendall, use_nonparam_test, compute_bootstrap_ci, n_bootstrap)
- 數據來源: 優先使用 processed_data, 否則使用原始 data
- 主要流程:
  - 按日期統計每日有效樣本數, 根據平均每日樣本量選擇高, 中, 低樣本模式, 對應最小日樣本門檻 5, 3, 2
  - 以 '信號日期' 分組, 對每個交易日計算因子與收益的 Spearman 或 Pearson 相關
    - 使用 ensure_list 保證每日 IC 列表類型穩定
    - 對缺乏變異性, 標準差為零等情況做防護, 盡量避免數學錯誤
  - 匯總每日 IC, 得到整體 IC 均值與理論標準差
  - 使用 t 分佈近似計算 t 統計量與 p 值, 無 scipy 時退化為正態近似
  - 按需附加:
    - Kendall Tau
    - 穩健相關
    - Mann Whitney U 檢驗
    - Bootstrap 置信區間
  - 全程更新 nomaly_stats['ic_calculation'][factor_col], 記錄處理日期數, 跳過原因等

### 4.4 分組收益計算
- calculate_group_returns(self, factor_col, n_groups)
- 功能: 將樣本按因子值從小到大排序後平均分成若干組, 計算每組表現與多空收益
- 主要步驟:
  - 使用 processed_data, 先剔除因子值為空的行
  - 按因子排序, 等量切分為 n 組, 確保每組樣本數盡量接近
  - 檢查各組樣本數的均衡性, 差異過大時打印警告
  - 分組聚合計算:
    - 組內平均收益, 收益標準差, 樣本數
    - 組內因子最小值與最大值, 組成參數區間描述
  - 構造 vg_returns 簡表與完整 group_stats 表
  - 計算各組收益的 t 統計量與 p 值, 並求出最高組減最低組的多空收益 long_short_return

### 4.5 年化收益與算法選擇
- _analyze_data_characteristics(self)
  - 統計觀測期年數, 總交易次數, 持股周期, 年化交易頻率
- _select_optimal_annualization_method(self, characteristics)
  - 根據數據質量與頻率穩定性, 選擇年化算法
  - 目前已統一為標準復利為主, CAGR 為輔的方案
- _calculate_adaptive_annual_returns(self, avg_returns, characteristics, method_info)
  - 對每組收益使用 safe_calculate_annual_return 做年化
  - 同時調用 alidate_annual_return_calculation 檢查結果合理性
- _print_annualization_analysis(self, characteristics, method_info, results)
  - 打印數據特徵, 算法選擇理由以及最終年化統計

### 4.6 评分與説明
- _get_new_scoring_weights 及一系列 _score_xxx 方法
  - 根據因子屬性與表現, 給 IC 均值, IR, 顯著性, 多空收益, 穩定性等指標打分
- _identify_factor_type(self, ic_mean, long_short_return)
  - 判斷因子是正向 alpha 因子, 反向因子還是無效因子
- _generate_detailed_reason 與 _generate_improved_detailed_reason
  - 將打分結果與統計指標轉成自然語言說明, 用於報告

### 4.7 報告, 匯總與可視化
- calculate_factor_stats(self, factor_col)
  - 計算單因子的描述統計與數據質量指標
- un_factor_analysis(self, use_pearson)
  - 對 actors 中所有因子依次調用 IC 計算, 分組收益, 年化分析等, 並將結果寫入 nalysis_results
- plot_factor_distribution(self) 與 plot_group_returns(self)
  - 在 HAS_PLOT 為真時, 畫出因子分佈與分組收益圖
- generate_summary_report(self)
  - 從 nalysis_results 構造匯總 DataFrame, 包含因子名, IC 均值, IC 標準差, IR, t 統計量, p 值, 多空收益
  - 打印表格並保存到 '因子分析匯總_時間戳.csv'
- generate_factor_analysis_report(self, summary_df, process_factors, factor_method, winsorize)
  - 生成主 TXT 報告, 包含管理層概要, 每個因子的詳細評級與說明, 正負因子分析, 策略建議與風險提示
- un_full_analysis(self)
  - 一站式調用完整流程: un_factor_analysis → 可視化 → generate_summary_report

## 5. ParameterizedFactorAnalyzer 類
### 5.1 角色與屬性
- 作用: 在已有樣本上, 將每個因子再劃分為多個參數區間 (通常為 10 等分), 對每個區間計算收益與風險指標並打分
- 主要屬性:
  - data: 原始或經初步處理的數據
  - actors 與 actor_list: 與 FactorAnalysis 相同的因子列表
  - eturn_col: 仍為 '持股2日收益率'
  - sqrt_annualization_factor, nnualization_factor: 用於年化波動與收益

### 5.2 數據預處理
- preprocess_data(self)
  - 處理百分比字符串列為小數
  - 將收益列轉為數值
  - 將 '信號日期' 轉為日期類型
  - 丟棄在收益與任一因子上存在缺失的行
  - 結果存入 processed_data

### 5.3 分組與綜合指標
- calculate_comprehensive_metrics(self, factor_col)
  - 對指定因子:
    - 使用 processed_data
    - 將樣本按因子值劃為若干等量參數區間
    - 對每個區間計算:
      - 平均收益, 收益標準差, 勝率
      - 最大回撤, 年化收益, 年化收益標準差
      - 年化夏普比與索提諾比
    - 匯總出因子整體的收益與風險特徵
- calculate_ic(self, factor_col, use_pearson)
  - 在帶參數分析場景下計算因子 IC, 用於補充評估

### 5.4 打分與報告
- score_factors(self, factor_results)
  - 對 {因子名: 綜合指標} 字典中的每個因子與參數區間打分
  - 綜合考慮勝率, 年化收益, 夏普, 回撤等多個維度, 得到 0 至 10 的綜合得分
  - 同時標註因子方向 (正向或反向)
- generate_parameterized_report(self)
  - 主要步驟:
    - 調用 calculate_comprehensive_metrics 生成功能因子與參數區間的統計結果
    - 調用 score_factors 得到 scores_df
    - 生成帶參數因子綜合分析 TXT 報告, 內容包括:
      - 正向與反向因子參數區間排行榜
      - 最優正向與負向參數區間推薦
      - 每個參數區間的詳細表現指標
      - 組合配置與權重建議
      - 風險提示
    - 輸出兩類 CSV:
      - '帶參數因子分析數據_時間戳.csv': 各因子參數區間的綜合得分
      - '帶參數因子詳細分析_因子名稱_時間戳.csv': 每個因子各參數區間的詳細指標

## 6. main 函數流程總結
- 入口: if __name__ == '__main__': main()
- 執行步驟:
  1. 創建 Logger, 將 sys.stdout 重定向到日誌文件
  2. 實例化 FactorAnalysis, 並調用 load_data
  3. 設置分析參數:
     - 默認使用 Spearman IC (use_pearson 為 False)
     - 開啟因子標準化與縮尾處理
  4. 調用 preprocess_data, 失敗則退出
  5. 打印可用因子列表
  6. 調用 un_factor_analysis 完成對所有因子的 IC, 分組收益與年化分析
  7. 調用 generate_summary_report 與 generate_factor_analysis_report 生成匯總與詳細 TXT 報告
  8. 再對每個因子單獨做一次 10 等分分組與 IC 計算, 打印結果
  9. 創建 ParameterizedFactorAnalyzer, 使用 nalyzer.data.copy() 作為輸入
 10. 調用 preprocess_data 與 generate_parameterized_report, 生成帶參數因子綜合分析報告及相關 CSV
 11. 打印完成信息並關閉 Logger

## 7. 使用建議
- 若需要在代碼中重用分析邏輯:
  - 直接使用 FactorAnalysis: load_data → preprocess_data → calculate_ic 與 calculate_group_returns
- 若需要分析某個因子在不同參數區間上的表現:
  - 使用 ParameterizedFactorAnalyzer: preprocess_data → calculate_comprehensive_metrics 或 generate_parameterized_report
- 所有匯總結果均會輸出為 CSV 與 TXT, 便於後續檢查與複盤
