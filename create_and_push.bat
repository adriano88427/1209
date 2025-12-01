@echo off
echo 请按照以下步骤手动创建GitHub仓库并推送代码：
echo.
echo 1. 打开浏览器，访问 https://github.com/adriano88427
echo 2. 点击 "New" 按钮创建新仓库
echo 3. 仓库名称输入: yinzifenxi1129
echo 4. 描述输入: 因子分析项目 - 1129版本
echo 5. 选择 Public (公开)
echo 6. 不要勾选 "Initialize this repository with a README"
echo 7. 点击 "Create repository"
echo.
echo 仓库创建完成后，按任意键继续推送代码...
pause > nul
echo.
echo 正在推送代码到新仓库...
git push origin1129 main
echo.
echo 代码已成功推送到 https://github.com/adriano88427/yinzifenxi1129
pause