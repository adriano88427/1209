# GitHub文件上传方案

## 方案概述

基于实际操作经验，本文档提供完整的GitHub文件上传解决方案，包括网络连接问题的解决方法和多个备选方案。

## 方案一：标准HTTPS连接（推荐）

### 适用场景
- 网络环境良好，能正常访问GitHub
- 需要完整的Git操作权限（push、pull、merge等）

### 操作步骤
```bash
# 1. 克隆仓库
git clone https://github.com/user/repo.git

# 2. 进入目录
cd repo

# 3. 创建.gitignore文件（排除不需要的文件）
echo "# 排除目录
baogao/
shuju/
*.log
*.tmp
__pycache__/" > .gitignore

# 4. 添加文件
git add .

# 5. 提交更改
git commit -m "Add project files"

# 6. 推送到远程仓库
git push origin main
```

## 方案二：gitclone.com代理（网络受限场景）

### 适用场景
- 直接连接GitHub超时或失败
- 需要通过代理访问GitHub

### 操作步骤
```bash
# 1. 使用代理URL克隆仓库
git clone https://gitclone.com/github.com/user/repo.git

# 2. 进入目录
cd repo

# 3. 修改远程仓库地址为原始GitHub地址
git remote set-url origin https://github.com/user/repo.git

# 4. 后续操作同标准流程
git add .
git commit -m "Add project files"
git push origin main
```

### 注意事项
- 代理URL仅用于克隆，后续推送需要正常网络连接
- 如果推送仍然失败，可临时将远程地址改为代理地址进行只读操作

## 方案三：github.com.cnpmjs.org镜像（只读场景）

### 适用场景
- 仅需要读取/下载GitHub仓库内容
- 不需要push操作

### 使用方法
```bash
# 直接将GitHub URL中的github.com替换为github.com.cnpmjs.org
git clone https://github.com.cnpmjs.org/user/repo.git
```

### 限制说明
- **不需要注册**：任何人都可直接使用，无需注册账号
- **不支持登录**：仅提供代码只读访问功能
- **只读操作**：无法进行push、pull request等写入操作
- **内容延迟**：镜像内容可能有几分钟到几小时的延迟

### 注意事项
- 如果需要完整的GitHub操作（提交代码、创建PR等），建议使用SSH连接或配置本地代理
- 镜像站点内容更新可能不及时，如需最新代码建议直接连接GitHub

## 常见问题解决

### 1. HTTPS连接超时
```bash
# 尝试禁用SSL验证（不推荐，仅临时使用）
git config --global http.sslVerify false

# 或使用代理
git config --global http.proxy http://proxy-server:port
```

### 2. 仓库不存在错误
```bash
# 验证仓库是否存在
curl -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/user/repo
```

### 3. 合并冲突
```bash
# 拉取远程更改
git pull origin main --allow-unrelated-histories

# 解决冲突后提交
git add .
git commit -m "Resolve merge conflict"
git push origin main
```

### 4. 权限问题
```bash
# 配置Git用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 使用Personal Access Token进行认证
# 在推送时使用token作为密码
```

## 项目文件管理最佳实践

### .gitignore配置示例
```gitignore
# 排除大文件目录
baogao/
shuju/

# 排除缓存和临时文件
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
.venv/

# 排除系统文件
.DS_Store
Thumbs.db
*.log
*.tmp

# 排除IDE配置
.vscode/
.idea/
*.swp
*.swo
```

### 提交信息规范
```bash
# 使用清晰描述性提交信息
git commit -m "feat: Add new factor analysis module

- Implemented single-factor analysis
- Added dual-factor analysis support
- Updated configuration files"
```

### 分支管理
```bash
# 创建功能分支
git checkout -b feature/new-analysis

# 开发完成后合并到main
git checkout main
git merge feature/new-analysis
git push origin main
```

## 网络环境配置

### Windows环境
```powershell
# 检查网络连接
ping github.com

# 配置代理（如需要）
git config --global http.proxy http://proxy-server:port
git config --global https.proxy https://proxy-server:port
```

### WSL环境
```bash
# 在WSL中调用Windows脚本
pwsh.exe -c "git config --global http.sslVerify false"
```

## 总结

根据不同的网络环境和需求，选择合适的方案：

1. **网络正常** → 使用方案一（标准HTTPS）
2. **网络受限** → 使用方案二（gitclone.com代理）
3. **仅需读取** → 使用方案三（cnpmjs.org镜像）

**推荐流程**：
1. 先尝试标准HTTPS连接
2. 如果失败，使用gitclone.com代理克隆
3. 解决.gitignore配置，排除不需要的文件
4. 分步骤添加文件，避免大文件问题
5. 提交前检查状态，解决冲突
6. 使用清晰的提交信息

## 附录：快速命令参考

```bash
# 快速克隆（多种方式）
git clone https://github.com/user/repo.git                    # 标准方式
git clone https://gitclone.com/github.com/user/repo.git       # 代理方式
git clone https://github.com.cnpmjs.org/user/repo.git         # 镜像方式

# 常用操作
git status                    # 查看状态
git add .                     # 添加所有文件
git commit -m "message"       # 提交
git push origin main          # 推送到主分支
git pull origin main          # 拉取最新更改

# 解决冲突
git pull origin main --allow-unrelated-histories  # 允许无关历史合并
```

---
*创建时间：2025-12-09*
*基于实际GitHub文件上传操作经验编写*
