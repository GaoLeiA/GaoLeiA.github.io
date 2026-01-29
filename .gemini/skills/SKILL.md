---
name: Windows 开发环境命令执行指南
description: 在 Windows 环境下选择正确的工具执行命令，避免编码问题
---

# Windows 开发环境命令执行指南

本项目在 Windows 环境下开发，有多种命令执行方式可选。根据任务类型选择最合适的工具。

## 工具选择原则

### 1. Python（推荐用于文件操作）

**适用场景：**
- 📄 **文件读写**：特别是涉及中文或 UTF-8 编码的文件
- 🔄 **批量文本替换**：使用 `str.replace()` 或 `re.sub()`
- 📊 **数据处理**：JSON、YAML、Markdown 解析
- 🔧 **复杂脚本逻辑**：需要条件判断、循环、错误处理

**示例：**
```python
# 正确处理 UTF-8 编码的文件操作
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('old', 'new')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
```

**⚠️ 重要：** 始终指定 `encoding='utf-8'`，否则可能使用系统默认编码导致乱码。

---

### 2. PowerShell（用于系统命令和 Git）

**适用场景：**
- 🔀 **Git 操作**：`git add`, `git commit`, `git push`, `git status`
- 📁 **目录操作**：创建、删除、移动目录
- 🔍 **文件查找**：`Get-ChildItem`, `Test-Path`
- 🖥️ **系统管理**：环境变量、进程管理

**注意事项：**
- ❌ **不要使用 `Set-Content` 处理中文文件**：默认编码可能不是 UTF-8
- ❌ **不支持 `&&` 链接命令**：使用 `;` 或分开执行
- ✅ **使用 `-Encoding UTF8` 参数**：如果必须用 PowerShell 读写文件

**示例：**
```powershell
# Git 操作（推荐）
git add .
git commit -m "提交信息"
git push origin master

# 目录操作（推荐）
New-Item -ItemType Directory -Path "new_folder"
Move-Item -Path "source" -Destination "dest"

# ❌ 避免（会导致编码问题）
(Get-Content file.md) -replace 'old', 'new' | Set-Content file.md
```

---

### 3. WSL（用于 Linux 特定工具）

**适用场景：**
- 🔧 **Linux 命令行工具**：`sed`, `awk`, `grep`, `find`, `xargs`
- 📦 **需要 bash 脚本**：复杂的 shell 脚本
- 🐧 **Linux 环境测试**：验证跨平台兼容性

**路径转换：**
```
Windows: c:\projects\GaoLeiA.github.io
WSL:     /mnt/c/projects/GaoLeiA.github.io
```

**示例：**
```powershell
# 在 WSL 中执行 sed 替换
wsl sed -i 's/old/new/g' /mnt/c/projects/file.txt

# 在 WSL 中执行 find
wsl find /mnt/c/projects -name "*.md" -type f
```

**⚠️ 注意：** WSL 可能有代理警告，通常不影响使用。

---

## 常见任务推荐方案

| 任务类型 | 推荐工具 | 原因 |
|---------|---------|------|
| 修改 Markdown/中文文件 | **Python** | 编码可控 |
| Git 操作 | **PowerShell** | 原生支持好 |
| 批量重命名文件 | **Python** | 跨平台、可控 |
| 文本搜索替换 | **Python** | 编码安全 |
| 运行 Node.js/npm | **PowerShell** | 直接支持 |
| 复杂 shell 脚本 | **WSL** | bash 语法 |
| 安装系统依赖 | **PowerShell** | 管理员权限 |

---

## 编码问题排查

如果出现乱码：

1. **检查文件编码**：
   ```powershell
   # 查看文件前几行（指定编码）
   Get-Content file.md -First 5 -Encoding UTF8
   ```

2. **从 Git 恢复**：
   ```powershell
   git checkout HEAD -- path/to/file.md
   ```

3. **使用 Python 修复**：
   ```python
   # 读取并重新保存为正确编码
   with open('file.md', 'r', encoding='utf-8', errors='ignore') as f:
       content = f.read()
   with open('file.md', 'w', encoding='utf-8') as f:
       f.write(content)
   ```

---

## 项目特定信息

- **博客框架**: Next.js
- **内容目录**: `content/posts/`
- **静态资源**: `public/`
- **图片目录**: `public/posts-images/`
- **构建命令**: `npm run build`
- **开发服务器**: `npm run dev`
