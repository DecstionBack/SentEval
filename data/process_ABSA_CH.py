import xml.etree.ElementTree as ET

subtasks = ["PHNS", "CAME"]

output_prefixes = {s: "senteval_data/ABSA_CH/ABSA_CH_{}".format(s) for s in subtasks}

input_files = {
	"test": {s: "senteval_data/ABSA_CH/CH_gold_labels_{}_SB1_TEST_.xml".format(s) for s in subtasks},
	"train": {
		"PHNS": "senteval_data/ABSA_CH/Chinese_phones_training.xml",
		"CAME": "senteval_data/ABSA_CH/camera_training.xml"
	}
}

for subtask in subtasks:
	for split in ["train", "test"]:
		input_file = input_files[split][subtask]
		output_file = "{}_{}.tsv".format(output_prefixes[subtask], split)

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
							for opinion in child:
								opinion_data = opinion.attrib
								opinion_data["text"] = text
								data.append(opinion_data)

		w = open(output_file, "w")
		w.write("")
		w.close()

		w = open(output_file, "a")
		for d in data:
			if "\t".join(d.keys()) != "category\tpolarity\ttext":
				print d.keys()
			line = "\t".join(d.values()).encode("utf-8") + "\n"
			w.write(line)
		w.close()

