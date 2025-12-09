# GitHub仓库创建和上传脚本
Write-Host "正在准备上传项目到GitHub..." -ForegroundColor Green

# 检查Git状态
Write-Host "检查Git状态..." -ForegroundColor Yellow
git status

# 显示当前远程仓库
Write-Host "当前远程仓库:" -ForegroundColor Yellow
git remote -v

# 提示用户手动创建仓库
Write-Host "`n请按照以下步骤操作:" -ForegroundColor Cyan
Write-Host "1. 访问 https://github.com/adriano88427/1206" -ForegroundColor White
Write-Host "2. 如果仓库不存在，点击'Create repository'按钮创建一个新的公共仓库" -ForegroundColor White
Write-Host "3. 仓库创建完成后，按任意键继续上传代码..." -ForegroundColor White

# 等待用户确认
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# 尝试推送代码
Write-Host "`n正在上传代码到GitHub..." -ForegroundColor Yellow
try {
    git push -u origin main
    Write-Host "上传成功完成！" -ForegroundColor Green
} catch {
    Write-Host "上传失败，请检查网络连接和仓库权限。" -ForegroundColor Red
    Write-Host "错误信息: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n按任意键退出..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")