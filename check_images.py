#!/usr/bin/env python3
import os
import re

posts_dir = r"c:\projects\GaoLeiA.github.io\content\posts"

def check_missing_images():
    for filename in os.listdir(posts_dir):
        if not filename.endswith('.md'):
            continue
            
        filepath = os.path.join(posts_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        has_image = '/posts-images/' in content
        if not has_image:
            print(f"📄 {filename} 没有图片链接")
            
            # 简单的关键词匹配建议
            if "scheduler" in filename:
                print(f"   Suggest: scheduler_architecture.png, scheduler_algorithm_flow.png")
            elif "worker" in filename:
                print(f"   Suggest: worker_architecture.png")
            elif "flash" in filename:
                print(f"   Suggest: flashattention_architecture.png")
            elif "moe" in filename:
                print(f"   Suggest: (No specific MoE image found)")
        else:
             # Count images
             count = content.count('/posts-images/')
             print(f"🖼️  {filename}: {count} 张图片")

if __name__ == "__main__":
    check_missing_images()
