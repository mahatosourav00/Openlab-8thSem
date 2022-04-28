def spe_file_read(file):
    list = []
    for line in file.readlines()[12:16396]:
        list.append(float(line))
    return list
