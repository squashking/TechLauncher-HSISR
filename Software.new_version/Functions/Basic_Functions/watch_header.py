def read_hdr_file(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines[0].strip() != "ENVI":
            raise ValueError("File does not appear to be an ENVI header (missing 'ENVI' at beginning of first line).")
        for line in lines[1:]:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                metadata[key.strip()] = value.strip()
    return metadata

def write_hdr_file(file_path, metadata):
    with open(file_path, 'w') as file:
        file.write("ENVI\n")
        for key, value in metadata.items():
            file.write(f"{key} = {value}\n")

def modify_hdr_file(file_path, new_metadata):
    metadata = read_hdr_file(file_path)
    metadata.update(new_metadata)
    write_hdr_file(file_path, metadata)

# 使用示例
hdr_file_path = "Result/SR result/result.hdr"
new_metadata = {
    'samples': '1000',  # 修改 samples 的值
    'lines': '770',     # 修改 lines 的值
    # 你可以根据需要添加更多的修改
}

modify_hdr_file(hdr_file_path, new_metadata)

# 读取并显示修改后的 HDR 文件内容
hdr_metadata = read_hdr_file(hdr_file_path)
print(hdr_metadata)
