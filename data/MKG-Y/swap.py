# 打开原始文件读取，创建新文件写入
input_file = "valid1.txt"  # 原始文件名
output_file = "valid.txt"  # 输出文件名

# 打开文件并处理
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # 去掉行末的换行符并分割行
        parts = line.strip().split("\t")
        
        # 检查行是否符合预期格式
        if len(parts) == 3:
            # 交换最后两个数字的位置
            parts[-2], parts[-1] = parts[-1], parts[-2]
            # 将结果写入新文件
            outfile.write("\t".join(parts) + "\n")

print(f"处理完成，结果已保存到 {output_file}")
