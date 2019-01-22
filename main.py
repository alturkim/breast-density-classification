import utilities as util


output_path = util.dump_image_file_names("..\\IRMA_Patchers")
training_txt_file_path, testing_txt_file_path = util.split(output_path)
