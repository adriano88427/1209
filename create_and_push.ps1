# 创建GitHub仓库并推送代码的指导脚本

Write-Host "请按照以下步骤手动创建GitHub仓库并推送代码：" -ForegroundColor Green
Write-Host ""

Write-Host "1. 打开浏览器，访问 https://github.com/adriano88427" -ForegroundColor Yellow
Start-Process "https://github.com/adriano88427"

Write-Host "2. 点击页面右上角的 '+' 按钮，然后选择 'New repository'" -ForegroundColor Yellow
Write-Host "3. 在 'Repository name' 中输入: yinzifenxi1129" -ForegroundColor Yellow
Write-Host "4. 在 'Description' 中输入: 因子分析项目 - 1129版本" -ForegroundColor Yellow
Write-Host "5. 选择 'Public' (公开)" -ForegroundColor Yellow
Write-Host "6. 不要勾选 'Initialize this repository with a README'" -ForegroundColor Yellow
Write-Host "7. 点击 'Create repository' 按钮" -ForegroundColor Yellow

Write-Host ""
Read-Host "仓库创建完成后，按 Enter 键继续推送代码"

Write-Host ""
Write-Host "正在推送代码到新仓库..." -ForegroundColor Green

# 尝试推送代码
$pushResult = git push origin1129 main

if ($LASTEXITCODE -eq 0) {
    Write-Host "代码已成功推送到 https://github.com/adriano88427/yinzifenxi1129" -ForegroundColor Green
} else {
    Write-Host "推送失败，请检查仓库URL是否正确" -ForegroundColor Red
}

Read-Host "按 Enter 键退出"