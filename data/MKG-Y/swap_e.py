# 输入文件名和输出文件名
input_file = "entity2id1.txt"  # 原始文件路径
output_file = "entity2id.txt"  # 输出文件路径

# 打开文件并处理
with open(input_file, "r") as infile, open(output_file, "w") as outfile:\
    
    for i, line in enumerate(infile):
        # 跳过第一行
        if i == 0:
            continue
        # 去掉行末换行符并分割
        parts = line.strip().split("\t")
        
        # 检查是否有两部分数据
        if len(parts) == 2:
            # 交换两个数字的位置
            parts[0], parts[1] = parts[1], parts[0]
            # 写入文件
            outfile.write("\t".join(parts) + "\n")
        else:
            # 如果不是两部分数据，可以忽略或报错，这里选择忽略
            continue

print(f"处理完成，结果已保存到 {output_file}")
