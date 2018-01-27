import xml.etree.ElementTree as ET

output_file = "senteval_data/ABSA_SP/ABSA_SP.tsv"

data = []

tree = ET.parse('senteval_data/ABSA_SP/SP_REST_SB1_TEST.xml.gold')
reviews = tree.getroot()
for review in reviews:
	for sentences in review:
		for sentence in sentences:
			for child in sentence:
				if child.tag=="text":
					text = child.text
				elif child.tag=="Opinions":
					for opinion in child:
						opinion_data = opinion.attrib
						opinion_data["text"] = text
						data.append(opinion_data)

w = open(output_file, "w")
w.write("")
w.close()

w = open(output_file, "a")
for d in data:
	line = "\t".join(d.values()).encode("utf-8") + "\n"
	w.write(line)
w.close()
