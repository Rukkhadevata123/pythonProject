#!/bin/bash
# filepath: collect_utils.sh

# 获取当前文件夹下的 utils 目录中的所有 .py 文件
files=$(find utils -name "*.py")

# 为每个文件创建 Markdown 代码块并添加到报告末尾
for file in $files; do
    echo -e "\n### ${file##*/}\n" >> 3230102179_黎学圣_RayTracing作业报告.md
    echo '```python' >> 3230102179_黎学圣_RayTracing作业报告.md
    echo "# filepath: /home/yoimiya/zju_cg/ray_tracing/${file}" >> 3230102179_黎学圣_RayTracing作业报告.md
    cat "$file" >> 3230102179_黎学圣_RayTracing作业报告.md
    echo '```' >> 3230102179_黎学圣_RayTracing作业报告.md
done