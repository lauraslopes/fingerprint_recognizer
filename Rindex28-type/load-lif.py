import json,os


files = [os.path.join('.',f) for f in os.listdir('.') 
         if f.endswith('.lif') and os.path.isfile(os.path.join('.',f))]


for f in files:
	print f

	file_data = open(f).read()
	data=json.loads(file_data)

	for es,s in enumerate(data["shapes"]):
		print es,s["label"]
		for p in s["points"]:
			print p

