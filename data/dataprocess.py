import pandas as pd

def getPDB(pdbPath, position, CHAIN):
    text = []
    with open(pdbPath, 'r') as f:
        pdb_text = f.readlines()  # 读取该蛋白质pdb文件中的所有内容

    chain_pdb_txt = []
    for line in pdb_text:
        if line.startswith('ATOM') and line[21] == CHAIN:  # 只读取CHAIN链的数据
            chain_pdb_txt.append(line)
        if line.startswith('TER') and line[21] == CHAIN:  # 读到链的末尾 结束
            temp = line[23:26]
            temp = int(temp)
            seq_len = temp #通过pdb找到pdb序列真实长度
            break

    for line in chain_pdb_txt:
        temp = line[23:26]
        temp = int(temp)

        if position == 1:   #当位置为第一个原子时，就获取第一第二个第三原子的坐标值
            if temp == position or temp == position + 1 or temp == position + 2:
                text.append(line)
        else:
            if position == seq_len: #当位置为最后一个原子时，就获取倒数三个原子坐标值
                if temp == position-1 or temp == position or temp == position -2:
                    text.append(line)
            else:
                if temp == position + 1 or temp == position-1 or temp == position or temp == position + 2 or temp == position -2:
                    text.append(line)
    return text


#获取三行wild pdb
# data = pd.read_excel('testdata/OK_S350/OK_S350_1/OK_S350.xlsx')
data = pd.read_excel('testdata/OK_Ssym/OK_Ssym_1/OK_Ssym.xlsx')
# data = pd.read_excel('testdata/OK_p53/OK_p53_1/OK_p53.xlsx')
# data = pd.read_excel('testdata/OK_Myoglobin/OK_Myoglobin_1/OK_Myoglobin.xlsx')
# data = pd.read_excel('testdata/OK_S669/OK_S669_1/OK_S669.xlsx')
data = pd.read_excel('testdata/OK_S611/OK_S611_direct_1/OK_S611_direct.xlsx')
pdb_id = data['pdb_id']
wild = data['wild_type']
mutation = data['mutation']
position = data['pdb_seq_position']
CHAIN = data['CHAIN']
length = len(data)
dirpath = 'ome_wild_pdb/'
for i in range(len(pdb_id)):
    temp_pdbPath = dirpath + pdb_id[i] + str(CHAIN[i]) + '.pdb'
    temp_position = position[i]
    text = getPDB(temp_pdbPath, temp_position, str(CHAIN[i]))
    if len(text) == 0:
        print(pdb_id[i] + str(CHAIN[i]) + '_' + wild[i] + str(position[i]) + mutation[i] + '_wild')
    # out_file_name = 'ome_last_wild_pdb5/' + pdb_id[i] + CHAIN[i] +"_" + wild[i] + str(position[i]) + mutation[i] + '_wild' + ".pdb"
    out_file_name = 'testdata/ome_last_wild_pdb5/' + pdb_id[i] + str(CHAIN[i]) +"_" + wild[i] + str(position[i]) + mutation[i] + '_wild' + ".pdb"
    with open(out_file_name, 'w') as f:
        f.writelines(text)
        f.close()


#获取三行mutation pdb
# # data = pd.read_excel('testdata/OK_S350/OK_S350_1/OK_S350.xlsx')
# data = pd.read_excel('testdata/OK_Ssym/OK_Ssym_1/OK_Ssym.xlsx')
# data = pd.read_excel('testdata/OK_p53/OK_p53_1/OK_p53.xlsx')
# data = pd.read_excel('testdata/OK_Myoglobin/OK_Myoglobin_1/OK_Myoglobin.xlsx')
# data = pd.read_excel('testdata/OK_S669/OK_S669_1/OK_S669.xlsx')
# data = pd.read_excel('testdata/OK_S611/OK_S611_direct_1/OK_S611_direct.xlsx')
# pdb_id = data['pdb_id']
# wild = data['wild_type']
# mutation = data['mutation']
# position = data['pdb_seq_position']
# CHAIN = data['CHAIN']
# length = len(data)
# dirpath = 'ome_mutation_pdb/'
# for i in range(len(pdb_id)):
#     temp_pdbPath = dirpath + pdb_id[i] + str(CHAIN[i]) + "_" + wild[i] + str(position[i]) + mutation[i] + '_mutation' + ".pdb"
#     temp_position = position[i]
#     text = getPDB(temp_pdbPath, temp_position, str(CHAIN[i]))
#     if len(text) == 0:
#         print(pdb_id[i] + str(CHAIN[i]) + '_' + wild[i] + str(position[i]) + mutation[i] + '_wild')
#     # out_file_name = 'ome_last_mutation_pdb5/' + pdb_id[i] + str(CHAIN[i]) + "_" + wild[i] + str(position[i]) + mutation[i] + '_mutation' + ".pdb"
#     out_file_name = 'testdata/ome_last_mutation_pdb5/' + pdb_id[i] + str(CHAIN[i]) +"_" + wild[i] + str(position[i]) + mutation[i] + '_mutation' + ".pdb"
#     with open(out_file_name, 'w') as f:
#         f.writelines(text)
#         f.close()
