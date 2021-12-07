import os
import cv2
import scipy.io as sio
from shutil import copyfile,rmtree
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

gen_path = "VOC_hand_dataset"
if os.path.exists(gen_path):
    rmtree(gen_path)
gen_annot_path = os.path.join(gen_path,"VOC2007","Annotations")
gen_imageset_path = os.path.join(gen_path,"VOC2007","ImageSets","Main")
gen_jpeg_path = os.path.join(gen_path,"VOC2007","JPEGImages")
os.makedirs(gen_path)
os.makedirs(gen_annot_path)
os.makedirs(gen_imageset_path)
os.makedirs(gen_jpeg_path)

def make_xml(bbox_list,filename,tsize, difficult=0,labels=["hand"]):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = gen_path

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(bbox_list))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(tsize[0])  # tsize: (h, w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(tsize[1])  # tsize: (h, w)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(bbox_list)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(labels[0])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = str(difficult)
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = "unspecified"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = "0"

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(bbox_list[i][0]) + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(bbox_list[i][1]) + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(bbox_list[i][2] + 1))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(bbox_list[i][3] + 1))

    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)

    return dom

def generate_voc(set_name, root_path="hand_dataset"):
    images_dir = os.path.join(root_path,"images")
    annotations_dir = os.path.join(root_path,"annotations")

    assert os.path.isdir(images_dir)
    assert os.path.isdir(annotations_dir)

    f = open(os.path.join(gen_imageset_path,set_name+".txt"),"w")

    annotations_file = os.listdir(annotations_dir)
    for matfile in annotations_file:
        try:
            filename = matfile.split(".")[0]
            if matfile.split(".")[1] != "mat":
                continue
        except:
            continue
        image_path = os.path.join(images_dir,filename+".jpg")
        imagefile = cv2.imread(image_path)
        
        content = sio.loadmat(os.path.join(annotations_dir, matfile), matlab_compatible=False)
        boxes = content["boxes"]
        width = imagefile.shape[0]
        height = imagefile.shape[1]

        boxes_list = list()
        for box in boxes.T:
            a = box[0][0][0][0]
            b = box[0][0][0][1]
            c = box[0][0][0][2]
            d = box[0][0][0][3]

            aXY = (a[0][1], a[0][0])
            bXY = (b[0][1], b[0][0])
            cXY = (c[0][1], c[0][0])
            dXY = (d[0][1], d[0][0])

            maxX = max(aXY[0], bXY[0], cXY[0], dXY[0])
            minX = min(aXY[0], bXY[0], cXY[0], dXY[0])
            maxY = max(aXY[1], bXY[1], cXY[1], dXY[1])
            minY = min(aXY[1], bXY[1], cXY[1], dXY[1])

            # clip,防止超出边界
            maxX = min(maxX, width-1)
            minX = max(minX, 0)
            maxY = min(maxY, height-1)
            minY = max(minY, 0)
            boxes_list.append([minX, minY, maxX, maxY])
        
        dom = make_xml(boxes_list,filename,imagefile.shape)
        anno_xml = os.path.join(gen_annot_path, filename + '.xml')
        with open(anno_xml, 'w') as fx:
            fx.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))
        fx.close()
        copyfile(image_path,os.path.join(gen_jpeg_path,filename + '.jpg'))

        f.write(filename)
        f.write("\n")
    
    f.close()


if __name__ == "__main__":
    
    generate_voc("test","hand_dataset/test_dataset/test_data")
    generate_voc("training","hand_dataset/training_dataset/training_data")
    generate_voc("validation","hand_dataset/validation_dataset/validation_data")
