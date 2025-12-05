# 使用GitHub API创建仓库并上传代码
param(
    [Parameter(Mandatory=$false)]
    [string]$GitHubToken = $env:GITHUB_TOKEN
)

Write-Host "正在使用GitHub API创建仓库..." -ForegroundColor Green

# 如果没有提供令牌，提示用户输入
if ([string]::IsNullOrEmpty($GitHubToken)) {
    Write-Host "请输入您的GitHub Personal Access Token:" -ForegroundColor Yellow
    $GitHubToken = Read-Host -AsSecureString
    $GitHubToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($GitHubToken))
}

# 仓库信息
$owner = "adriano88427"
$repoName = "1206"
$description = "因子分析项目"
$private = $false

# 创建仓库的API端点
$apiUrl = "https://api.github.com/user/repos"

# 请求体
$body = @{
    name = $repoName
    description = $description
    private = $private
}
$jsonBody = $body | ConvertTo-Json

# 请求头
$headers = @{
    Authorization = "token $GitHubToken"
    Accept = "application/vnd.github.v3+json"
}

try {
    Write-Host "正在创建仓库..." -ForegroundColor Yellow
    $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Body $jsonBody -Headers $headers -ContentType "application/json"
    Write-Host "仓库创建成功！" -ForegroundColor Green
    Write-Host "仓库URL: $($response.html_url)" -ForegroundColor White
    
    # 等待几秒钟让仓库完全初始化
    Write-Host "等待仓库初始化..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # 推送代码
    Write-Host "正在推送代码到GitHub..." -ForegroundColor Yellow
    git push -u origin main
    Write-Host "代码上传成功完成！" -ForegroundColor Green
    
} catch {
    Write-Host "操作失败: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = [System.IO.StreamReader]::new($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $errorBody = $reader.ReadToEnd()
        Write-Host "错误详情: $errorBody" -ForegroundColor Red
    }
}

Write-Host "`n按任意键退出..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")