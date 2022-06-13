import matplotlib.pyplot as plt
from os import walk
from cut import cut

def process(input_dirname, date, output_dirname):
	_, _, fs = next(walk(input_dirname), (None, None, []))
	input_filenames = sorted(fs)
	#print(input_filenames)
	group = 1
	for input_filename in sorted(input_filenames):
		image = plt.imread(f"{input_dirname}/{input_filename}")
		cut_images = cut(image)
		index = 1
		for cut_image in cut_images:
			output_filename = f"{date}_{group:02}_{index:02}.jpg"
			plt.imsave(f"{output_dirname}/{output_filename}", cut_image)
			index += 1
		group += 1

def main():
	'''
	_, ds, _ = next(walk("."), (None, None, []))
	dirnames = sorted(d for d in ds if d.startswith("AI_Bug"))
	print(dirnames)

	for dirname in dirnames:
		date = dirname.split("_")[-1]
		print(date)
		process(dirname, date, "data")
	'''
	dirname = "AI_Bug_0522_new"
	date = dirname.split("_")[-2]
	process(dirname, date, "new_data")

if __name__ == '__main__':
	main()
