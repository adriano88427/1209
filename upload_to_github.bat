@echo off
echo 正在创建GitHub仓库并上传项目...
echo.
echo 请按照以下步骤操作：
echo 1. 访问 https://github.com/adriano88427/1206
echo 2. 如果仓库不存在，点击"Create repository"按钮创建一个新的公共仓库
echo 3. 仓库创建完成后，按任意键继续上传代码...
echo.
pause
echo.
echo 正在上传代码到GitHub...
git push -u origin main
echo.
echo 上传完成！
pause