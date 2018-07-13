def database(dir, max_len):
	f = open(dir, 'r')
	decodelabel = {"0": [1, 0],"1": [0, 1]}
	decodex = {'N':[1,0,0,0,0,0],'A':[0,1,0,0,0,0],'T':[0,0,1,0,0,0],'C':[0,0,0,1,0,0],'G':[0,0,0,0,1,0],'0':[0,0,0,0,0,1]}
	x_one_hot = []
	y_one_hot = []
	print("#####读取data")
	data = f.readlines()
	print("#####over data")
	print("训练集的大小有：  ", len(data))

	for line in data:
		x = line.split()[:-1]
		array = [decodex['0'] * 64] * max_len
		if len(x) > max_len:
			continue
		for i in range(len(x)):
			site = x[i].upper()
			exon = []
			for code in list(site):
				exon.append(decodex[code])
			array[i] = sum(exon,[])
		y = line.split()[-1]
		x_one_hot.append(array)
		y_one_hot.append(decodelabel[y])