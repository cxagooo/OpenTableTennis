# 导入必要的库
import re
def list_change(i, o):
    # 定义文件路径
    input_file_path = i  # 请替换为您的输入文件路径
    output_file_path = o  # 请替换为您想要保存输出文件的路径

    # 创建一个空列表来存储处理后的数据
    processed_data = []
    with open(input_file_path, 'r') as file:
        for line in file:
            # 使用正则表达式移除方括号
            clean_line = re.sub(r'\[|\]', '', line)
            # 分割字符串，得到每个元素
            elements = clean_line.split()
            # 只保留前两个元素
            first_two_elements = elements[:2]
            # 将字符串转换为浮点数，并添加到结果列表
            processed_data.append([float(element) for element in first_two_elements])

    # 获取前三行的数据
    first_three_rows = processed_data[:3]
    # 获取第四行的数据
    fourth_row = processed_data[3]

    # 减去第四行的值
    result_data = []
    for row in first_three_rows:
        new_row = [x - y for x, y in zip(row, fourth_row)]
        result_data.append(new_row)


    with open(output_file_path, 'w') as output_file:
        for row in result_data:
            output_file.write(' '.join(map(str, row)) + '\n')

    # 打印处理后的数据
    for row in result_data:
        print(row)