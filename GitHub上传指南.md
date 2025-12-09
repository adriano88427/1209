# GitHub仓库创建和项目上传指南

## 步骤1：创建GitHub仓库

1. 访问 [GitHub](https://github.com) 并登录您的账户
2. 点击右上角的 "+" 按钮，然后选择 "New repository"
3. 在 "Repository name" 字段中输入：`1206`
4. 在 "Description" 字段中输入：`因子分析项目`
5. 选择 "Public" (公开) 或 "Private" (私有)
6. **不要**勾选 "Add a README file"、"Add .gitignore" 或 "Choose a license"（因为我们已经有了本地仓库）
7. 点击 "Create repository" 按钮

## 步骤2：推送本地代码到GitHub

仓库创建后，GitHub会显示一些命令。由于我们已经有本地仓库，我们需要：

1. 打开命令提示符或PowerShell
2. 导航到项目目录：
   ```
   cd c:\Users\lenovo\Documents\yinzifenxi\1205
   ```
3. 确保远程仓库指向正确的URL：
   ```
   git remote set-url origin https://github.com/adriano88427/1206.git
   ```
4. 推送代码到GitHub：
   ```
   git push -u origin main
   ```

## 步骤3：验证上传

1. 访问 [https://github.com/adriano88427/1206](https://github.com/adriano88427/1206)
2. 确认所有文件都已上传

## 常见问题

### 如果遇到"repository not found"错误：
- 确保仓库已正确创建
- 检查仓库名称是否正确（1206）
- 确认您有权限访问该仓库

### 如果遇到认证错误：
- 确保您已登录GitHub
- 如果使用HTTPS，可能需要配置个人访问令牌

### 如果遇到推送被拒绝：
- 可能需要先拉取远程更改：`git pull origin main --allow-unrelated-histories`
- 然后再尝试推送：`git push -u origin main`

## 自动化脚本

您也可以运行以下批处理文件自动完成推送（在手动创建仓库后）：
```
upload_to_github.bat
```

或者运行PowerShell脚本：
```
powershell -ExecutionPolicy Bypass -File upload_to_github.ps1
```