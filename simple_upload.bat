@echo off
echo ========================================
echo    项目上传到GitHub脚本
echo ========================================
echo.

echo 检查Git状态...
git status
echo.

echo 当前远程仓库:
git remote -v
echo.

echo 正在尝试上传代码到GitHub...
echo 如果出现错误，请确保已在GitHub上创建了仓库 https://github.com/adriano88427/1206
echo.

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo    上传成功！
    echo ========================================
    echo 您可以在 https://github.com/adriano88427/1206 查看您的项目
) else (
    echo.
    echo ========================================
    echo    上传失败！
    echo ========================================
    echo 请检查以下几点：
    echo 1. 是否已在GitHub上创建了仓库 https://github.com/adriano88427/1206
    echo 2. 是否有权限访问该仓库
    echo 3. 网络连接是否正常
    echo.
    echo 如果问题仍然存在，请尝试手动运行以下命令：
    echo git push -u origin main
)

echo.
pause