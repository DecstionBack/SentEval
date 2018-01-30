import xml.etree.ElementTree as ET

def parse_file(input_file, output_file):
	data = []

	tree = ET.parse(input_file)
	reviews = tree.getroot()
	for review in reviews:
		for sentences in review:
			for sentence in sentences:
				for child in sentence:
					if child.tag=="text":
						text = child.text
					elif child.tag=="Opinions":
						polarities = []
						for opinion in child:
							opinion_data = opinion.attrib
							polarities.append(opinion_data["polarity"])
						if len(polarities)==1:
							data.append({"category": opinion_data["category"], "polarity": polarities[0], "text": text})
						elif all([p=="positive" for p in polarities]):
							data.append({"category": opinion_data["category"], "polarity": "positive", "text": text})
						elif all([p=="neutral" for p in polarities]):
							data.append({"category": opinion_data["category"], "polarity": "neutral", "text": text})
						elif not any([p=="negative" for p in polarities]):
							data.append({"category": opinion_data["category"], "polarity": "negative", "text": text})			

	w = open(output_file, "w")
	w.write("")
	w.close()

	w = open(output_file, "a")
	for d in data:
		d = {k: d[k] for k in ["category", "polarity", "text"]}
		if "\t".join(d.keys()) != "category\tpolarity\ttext":
			print d.keys()
		line = "\t".join(d.values()).encode("utf-8") + "\n"
		w.write(line)
	w.close()

parse_file('senteval_data/ABSA_SP/SP_REST_SB1_TEST.xml.gold', "senteval_data/ABSA_SP/ABSA_SP_test.tsv")
parse_file('senteval_data/ABSA_SP/SemEval-2016ABSA Restaurants-Spanish_Train_Subtask1.xml', "senteval_data/ABSA_SP/ABSA_SP_train.tsv")
