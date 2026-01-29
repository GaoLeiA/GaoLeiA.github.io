#!/usr/bin/env python3
"""
修复所有 Markdown 文件中的图片路径
将 ./images/xxx 和 images/xxx 统一转换为 /posts-images/xxx
"""
import os
import re

posts_dir = r"c:\projects\GaoLeiA.github.io\content\posts"
image_dir = r"c:\projects\GaoLeiA.github.io\public\posts-images"

# 获取所有可用图片
available_images = set(os.listdir(image_dir))

def fix_image_paths():
    count = 0
    for filename in os.listdir(posts_dir):
        if not filename.endswith('.md'):
            continue
        
        filepath = os.path.join(posts_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 修复 ./images/ -> /posts-images/
        new_content = content.replace('(./images/', '(/posts-images/')
        
        # 2. 修复 images/ -> /posts-images/ (防止之前的脚本有遗漏)
        new_content = new_content.replace('(images/', '(/posts-images/')
        
        # 3. 检查是否有未链接的图片引用 (比如文件名匹配)
        # 这是一个简单的启发式：如果文章标题或内容包含某些关键词，尝试插入对应图片
        # 但为了安全，我们只修复现有链接。
        
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ 已修复路径: {filename}")
            count += 1
        else:
            # 检查是否有断链
            links = re.findall(r'\((/posts-images/[^)]+)\)', new_content)
            for link in links:
                img_name = os.path.basename(link)
                if img_name not in available_images:
                    print(f"⚠️  {filename}: 图片不存在 {img_name}")
    
    print(f"\n🎉 总共修复了 {count} 个文件")

if __name__ == "__main__":
    fix_image_paths()
