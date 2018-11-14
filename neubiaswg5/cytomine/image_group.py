# -*- coding: utf-8 -*-
import re
import os
from os import path, makedirs
from cytomine.models import ImageGroupCollection, AttachedFileCollection


def download_attach_files_from_image_group(id_project, folder_path):
    # print(self._con.current_user)
    # ... Assume we are connected
    image_groups = ImageGroupCollection().fetch_with_filter("project", id_project)
    for image_group in image_groups:
        attached_collection = AttachedFileCollection(image_group).fetch()
        if len(attached_collection) > 0:
            print(image_group.name)
            for attached_file in attached_collection:
                print(attached_file.filename)
                attached_file.download(path.join(folder_path, attached_file.filename), True)


def download_all_image_group_from_project(id_project, output_folder_raw_image,
                                          output_folder_label_image=None, output_folder_path_attachment=None):
    """
    Usage: To download all the .ome.tif image stacks with their associated attachments
           The image stack will be saved using their uploaded name and the attachment
           will be saved using the basename of their associated image followed with their file extension
           In case multiple attachments are associatled with one group image, a number will be added
                                  Otherwise, no number will be added.
                                  e.g.: my_test.ome.tif the attachments will be my_test_1.swc my_test_2.swc
    Parameters
    ----------
    output_folder_raw_image:      The output folder path to save the raw images. If no other folder are mention,
                                  everything will be saved into this folder and the attachment.
    output_folder_label_image:    The output folder path to save the label images, optional.
    output_folder_path_attachment:The output folder path to save the attachment files, optional.
    """
    # ... Assume we are connected
    image_groups = ImageGroupCollection().fetch_with_filter("project", id_project)
    for image_group in image_groups:
        download_image_group(
            image_group, True,
            output_folder_raw_image,
            output_folder_label_image, output_folder_path_attachment
        )


def download_image_group(image_group, include_attachment, output_folder_raw_image,
                         output_folder_label_image=None, output_folder_path_attachment=None):
    """
    Usage: To download an .ome.tif image stack with its associated attachment
    Parameters
    ----------
    image_group: cytomine.ImageGroupThe
        image group object
    include_attachment: bool
       Whether or not to download the associated attachment
    output_folder_raw_image: str
        The output folder path to save the raw images. If no other folder are mention,
        everything will be saved into this folder and the attachment.
    output_folder_label_image: str|None
        The output folder path to save the label images.
    output_folder_path_attachment: str|None
        The output folder path to save the attachment files.
    """
    # If optional output directories parameters are None, use the raw directory
    if not output_folder_path_attachment:
        output_folder_path_attachment = output_folder_raw_image
    if not output_folder_label_image:
        output_folder_label_image = output_folder_raw_image
    # Create directory
    if not os.path.exists(output_folder_raw_image):
        makedirs(output_folder_raw_image)
    if output_folder_path_attachment and not os.path.exists(output_folder_path_attachment):
        makedirs(output_folder_path_attachment)
    if output_folder_label_image and not os.path.exists(output_folder_label_image):
        makedirs(output_folder_label_image)

    image_group.download(path.join(output_folder_raw_image, image_group.name), True)

    # If a file contain '_lbl.' inside its filename, it will be moved into the output_folder_label_image
    regexp = re.compile(r'(.*)_lbl.(.*)')
    files = os.listdir(output_folder_raw_image)
    for file in files:
        if os.path.isfile(path.join(output_folder_raw_image, file)) and regexp.search(file):
            print(file)
            os.rename(path.join(output_folder_raw_image, file), path.join(output_folder_label_image, file))
            continue
        else:
            continue

    if include_attachment:
        attached_collection = AttachedFileCollection(image_group).fetch()
        # We are supposed to have only 1 attachment per image group for NEUBIAS
        # In case there is one more, let's add a number
        if len(attached_collection) > 0:
            # print(image_group.name)
            padding_zero_size = len(str(len(attached_collection)))
            # attachment_number = 1
            # Sort the attachment by created date
            attached_collection = sorted(attached_collection, key=lambda x: x.created)

            # Always take the first attachement by default
            attached_file=attached_collection[0]
            # The attachment file will have the same basename as its associated image group
            # We suppose that the file format is .ome.tif
            base_image_group_name = image_group.name[:-8]
            attachment_file_extension = attached_file.filename[-3:]
            attachment_file_name = base_image_group_name + '.' \
                                        + attachment_file_extension
            attached_file.download(path.join(output_folder_path_attachment, attachment_file_name), True)

            '''
            for attached_file in attached_collection:
                # The attachment file will have the same basename as its associated image group
                # We suppose that the file format is .ome.tif
                base_image_group_name = image_group.name[:-8]
                attachment_file_extension = attached_file.filename[-3:]

                if len(attached_collection) > 1:
                    attachment_number_str = str(attachment_number).zfill(padding_zero_size)
                else:
                    attachment_number_str = ''

                attachment_file_name = base_image_group_name + '_' + attachment_number_str + '.' \
                                        + attachment_file_extension
                # print(attached_file.filename + ' ' + attached_file.created)
                attached_file.download(path.join(output_folder_path_attachment, attachment_file_name), True)
                attachment_number = attachment_number + 1
            '''