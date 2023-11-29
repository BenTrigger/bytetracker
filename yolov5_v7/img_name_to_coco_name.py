import os

def serialize_file_names(imgfolder_path):
    count = 1
    lblfolder_path = imgfolder_path.replace('images','labels')
    for filename in os.listdir(imgfolder_path):
        old_file_path = os.path.join(imgfolder_path, filename)
        old_file_txt_path = old_file_path.replace('images','labels').replace('.png','.txt')
        # Extract the file extension
        file_name, _ = os.path.splitext(filename)
        img_ext, txt_ext = '.png','.txt'
        # Pad the count with leading zeros to make it a 10-digit number
        serialized_name = f"{count:010d}"

        # Construct the new file name
        new_fileimg_name = f"{serialized_name}{img_ext}"
        new_filetxt_name = f"{serialized_name}{txt_ext}"
        new_fileimg_path = os.path.join(imgfolder_path, new_fileimg_name)
        new_filetxt_path = os.path.join(lblfolder_path, new_filetxt_name)

        # Rename the file
        os.rename(old_file_path, new_fileimg_path)
        os.rename(old_file_txt_path, new_filetxt_path)

        count += 1

if __name__ == "__main__":
    folder_path  = "/home/user1/ariel/yolov5_quant_sample/b_data/images/"  # Replace with the path to the folder containing files to be serialize
    serialize_file_names(folder_path)
