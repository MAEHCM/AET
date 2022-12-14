import argparse
import json
import os

from PIL import Image
from transformers import AutoTokenizer


def normalize_bbox(bbox, width, length):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ]


def string_box(box):
    return (
            str(box[0])
            + " "
            + str(box[1])
            + " "
            + str(box[2])
            + " "
            + str(box[3])
    )


def actual_bbox_string(box, width, length):
    return (
            str(box[0])
            + " "
            + str(box[1])
            + " "
            + str(box[2])
            + " "
            + str(box[3])
            + "\t"
            + str(width)
            + " "
            + str(length)
    )


def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box

def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox

a=set()

def convert(args):
    # 在外部data文件中，连开三个文件(如果不存在，就新建)
    with open(
            os.path.join(args.output_dir, args.data_split + ".txt.tmp"),
            "w",
            encoding="utf8",
    ) as fw, open(
        os.path.join(args.output_dir, args.data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(args.output_dir, args.data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw:
        for file in os.listdir(args.data_dir):
            # 0000971160.json
            file_path = os.path.join(args.data_dir, file)
            # ../data/FUNSD/training_data/annotations\0000971160.json
            with open(file_path, "r", encoding="utf8") as f:
                # 读入json中的数据
                data = json.load(f)
                # {'form': [{'box': [292, 91, 376, 175], 'text': 'R&D', 'label': 'other', 'words': [{'box': [292, 91, 376, 175], 'text': 'R&D'}], 'linking': [], 'id': 0}, {'box': [219, 316, 225, 327], 'text': ':', 'label': 'question', 'words': [{'box': [219, 316, 225, 327], 'text': ':'}], 'linking': [], 'id': 1}, {'box': [95, 355, 169, 370], 'text': 'Suggestion:', 'label': 'question', 'words': [{'box': [95, 355, 169, 370], 'text': 'Suggestion:'}], 'linking': [[2, 16]], 'id': 2}, {'box': [482, 268, 518, 282], 'text': 'Date:', 'label': 'question', 'words': [{'box': [482, 268, 518, 282], 'text': 'Date:'}], 'linking': [[3, 12]], 'id': 3}, {'box': [511, 309, 570, 323], 'text': 'Licensee', 'label': 'answer', 'words': [{'box': [511, 309, 570, 323], 'text': 'Licensee'}], 'linking': [[13, 4]], 'id': 4}, {'box': [211, 651, 217, 662], 'text': '', 'label': 'question', 'words': [{'box': [211, 651, 217, 662], 'text': ''}], 'linking': [], 'id': 5}, {'box': [461, 605, 483, 619], 'text': 'Yes', 'label': 'question', 'words': [{'box': [461, 605, 483, 619], 'text': 'Yes'}], 'linking': [[19, 6]], 'id': 6}, {'box': [545, 603, 563, 617], 'text': 'No', 'label': 'question', 'words': [{'box': [545, 603, 563, 617], 'text': 'No'}], 'linking': [[19, 7]], 'id': 7}, {'box': [525, 904, 641, 926], 'text': '597005708', 'label': 'other', 'words': [{'box': [525, 904, 641, 926], 'text': '597005708'}], 'linking': [], 'id': 8}, {'text': 'R&D QUALITY IMPROVEMENT SUGGESTION/ SOLUTION FORM', 'box': [256, 201, 423, 230], 'linking': [], 'label': 'header', 'words': [{'text': 'R&D', 'box': [257, 203, 279, 214]}, {'text': 'QUALITY', 'box': [285, 203, 334, 216]}, {'text': 'IMPROVEMENT', 'box': [341, 201, 418, 211]}, {'text': 'SUGGESTION/', 'box': [256, 215, 324, 229]}, {'text': '', 'box': [324, 216, 332, 230]}, {'text': 'SOLUTION', 'box': [331, 214, 387, 228]}, {'text': 'FORM', 'box': [395, 215, 423, 228]}], 'id': 9}, {'text': 'Name / Phone Ext. :', 'box': [89, 272, 204, 289], 'linking': [[10, 11]], 'label': 'question', 'words': [{'text': 'Name', 'box': [89, 274, 118, 289]}, {'text': '/', 'box': [117, 274, 127, 288]}, {'text': 'Phone', 'box': [128, 274, 163, 289]}, {'text': 'Ext.', 'box': [169, 272, 196, 287]}, {'text': ':', 'box': [196, 274, 204, 288]}], 'id': 10}, {'text': 'M. Hamann P. Harper, P. Martinez', 'box': [215, 271, 451, 287], 'linking': [[10, 11]], 'label': 'answer', 'words': [{'text': 'M.', 'box': [215, 272, 230, 287]}, {'text': 'Hamann', 'box': [237, 272, 287, 286]}, {'text': 'P.', 'box': [293, 272, 307, 286]}, {'text': 'Harper,', 'box': [314, 274, 363, 285]}, {'text': 'P.', 'box': [370, 272, 384, 285]}, {'text': 'Martinez', 'box': [390, 271, 451, 282]}], 'id': 11}, {'text': '9/ 3/ 92', 'box': [543, 264, 590, 279], 'linking': [[3, 12]], 'label': 'answer', 'words': [{'text': '9/', 'box': [543, 265, 560, 279]}, {'text': '3/', 'box': [560, 264, 575, 279]}, {'text': '92', 'box': [575, 264, 590, 279]}], 'id': 12}, {'text': 'R&D Group:', 'box': [420, 310, 491, 323], 'linking': [[13, 4]], 'label': 'question', 'words': [{'text': 'R&D', 'box': [420, 310, 442, 323]}, {'text': 'Group:', 'box': [448, 310, 491, 323]}], 'id': 13}, {'text': 'J. S. Wigand', 'box': [236, 313, 327, 327], 'linking': [[15, 14]], 'label': 'answer', 'words': [{'text': 'J.', 'box': [236, 313, 251, 327]}, {'text': 'S.', 'box': [256, 313, 273, 326]}, {'text': 'Wigand', 'box': [278, 313, 327, 327]}], 'id': 14}, {'text': 'Supervisor / Manager', 'box': [91, 316, 218, 331], 'linking': [[15, 14]], 'label': 'question', 'words': [{'text': 'Supervisor', 'box': [91, 316, 161, 330]}, {'text': '/', 'box': [163, 318, 169, 331]}, {'text': 'Manager', 'box': [169, 317, 218, 327]}], 'id': 15}, {'text': 'Discontinue coal retention analyses on licensee submitted product samples (Note : Coal Retention testing is not performed by most licensees. Other B&W physical measurements as ends stability and inspection for soft spots in ciparettes are thought to be sufficient measures to assure cigarette physical integrity. The proposed action will increase laboratory productivity . )', 'box': [190, 346, 594, 447], 'linking': [[2, 16]], 'label': 'answer', 'words': [{'text': 'Discontinue', 'box': [190, 355, 268, 366]}, {'text': 'coal', 'box': [274, 353, 303, 366]}, {'text': 'retention', 'box': [309, 352, 375, 365]}, {'text': 'analyses', 'box': [381, 351, 435, 365]}, {'text': 'on', 'box': [443, 352, 458, 363]}, {'text': 'licensee', 'box': [464, 348, 520, 362]}, {'text': 'submitted', 'box': [527, 346, 594, 361]}, {'text': 'product', 'box': [190, 369, 240, 383]}, {'text': 'samples', 'box': [247, 367, 301, 380]}, {'text': '(Note', 'box': [318, 365, 352, 379]}, {'text': ':', 'box': [352, 367, 359, 380]}, {'text': 'Coal', 'box': [373, 366, 402, 376]}, {'text': 'Retention', 'box': [408, 366, 472, 376]}, {'text': 'testing', 'box': [479, 365, 529, 376]}, {'text': 'is', 'box': [536, 363, 549, 374]}, {'text': 'not', 'box': [554, 363, 578, 374]}, {'text': 'performed', 'box': [190, 383, 256, 394]}, {'text': 'by', 'box': [261, 381, 275, 394]}, {'text': 'most', 'box': [282, 383, 311, 393]}, {'text': 'licensees.', 'box': [318, 380, 386, 391]}, {'text': 'Other', 'box': [401, 378, 437, 389]}, {'text': 'B&W', 'box': [443, 378, 465, 389]}, {'text': 'physical', 'box': [471, 377, 528, 391]}, {'text': 'measurements', 'box': [191, 398, 275, 406]}, {'text': 'as', 'box': [282, 397, 297, 405]}, {'text': 'ends', 'box': [304, 394, 332, 405]}, {'text': 'stability', 'box': [339, 394, 402, 405]}, {'text': 'and', 'box': [409, 392, 430, 402]}, {'text': 'inspection', 'box': [437, 392, 508, 403]}, {'text': 'for', 'box': [515, 391, 535, 402]}, {'text': 'soft', 'box': [542, 391, 571, 401]}, {'text': 'spots', 'box': [193, 411, 228, 422]}, {'text': 'in', 'box': [235, 409, 250, 420]}, {'text': 'ciparettes', 'box': [256, 409, 327, 419]}, {'text': 'are', 'box': [332, 408, 352, 418]}, {'text': 'thought', 'box': [360, 406, 410, 419]}, {'text': 'to', 'box': [415, 406, 430, 416]}, {'text': 'be', 'box': [436, 404, 453, 417]}, {'text': 'sufficient', 'box': [458, 405, 529, 415]}, {'text': 'measures', 'box': [535, 405, 592, 415]}, {'text': 'to', 'box': [193, 425, 208, 433]}, {'text': 'assure', 'box': [214, 423, 255, 431]}, {'text': 'cigarette', 'box': [261, 420, 325, 434]}, {'text': 'physical', 'box': [331, 419, 390, 432]}, {'text': 'integrity.', 'box': [395, 418, 463, 431]}, {'text': 'The', 'box': [478, 416, 500, 429]}, {'text': 'proposed', 'box': [506, 418, 566, 431]}, {'text': 'action', 'box': [193, 436, 236, 447]}, {'text': 'will', 'box': [240, 436, 269, 447]}, {'text': 'increase', 'box': [277, 434, 333, 445]}, {'text': 'laboratory', 'box': [339, 433, 410, 446]}, {'text': 'productivity', 'box': [418, 430, 502, 445]}, {'text': '.', 'box': [503, 433, 507, 444]}, {'text': ')', 'box': [508, 430, 514, 444]}], 'id': 16}, {'text': 'Suggested Solutions (s) :', 'box': [95, 486, 250, 504], 'linking': [[17, 18]], 'label': 'question', 'words': [{'text': 'Suggested', 'box': [95, 489, 159, 504]}, {'text': 'Solutions', 'box': [165, 487, 222, 501]}, {'text': '(s)', 'box': [223, 486, 241, 503]}, {'text': ':', 'box': [243, 489, 250, 503]}], 'id': 17}, {'text': 'Delete coal retention from the list of standard analyses performed on licensee submitted product samples. Special requests for coal retention testing could still be submitted on an exception basis.', 'box': [263, 483, 593, 553], 'linking': [[17, 18]], 'label': 'answer', 'words': [{'text': 'Delete', 'box': [263, 486, 306, 500]}, {'text': 'coal', 'box': [313, 486, 341, 499]}, {'text': 'retention', 'box': [348, 486, 412, 497]}, {'text': 'from', 'box': [416, 485, 447, 498]}, {'text': 'the', 'box': [453, 485, 475, 498]}, {'text': 'list', 'box': [480, 483, 508, 496]}, {'text': 'of', 'box': [515, 483, 532, 494]}, {'text': 'standard', 'box': [536, 483, 593, 494]}, {'text': 'analyses', 'box': [264, 501, 320, 514]}, {'text': 'performed', 'box': [324, 501, 392, 512]}, {'text': 'on', 'box': [397, 501, 412, 511]}, {'text': 'licensee', 'box': [419, 499, 475, 512]}, {'text': 'submitted', 'box': [482, 499, 546, 510]}, {'text': 'product', 'box': [264, 517, 314, 528]}, {'text': 'samples.', 'box': [320, 514, 374, 528]}, {'text': 'Special', 'box': [390, 513, 439, 526]}, {'text': 'requests', 'box': [446, 513, 502, 524]}, {'text': 'for', 'box': [508, 511, 530, 522]}, {'text': 'coal', 'box': [538, 510, 566, 523]}, {'text': 'retention', 'box': [263, 529, 330, 540]}, {'text': 'testing', 'box': [335, 527, 387, 540]}, {'text': 'could', 'box': [390, 527, 428, 538]}, {'text': 'still', 'box': [433, 525, 468, 536]}, {'text': 'be', 'box': [473, 525, 488, 535]}, {'text': 'submitted', 'box': [496, 524, 560, 537]}, {'text': 'on', 'box': [566, 524, 584, 537]}, {'text': 'an', 'box': [264, 543, 281, 553]}, {'text': 'exception', 'box': [286, 539, 350, 553]}, {'text': 'basis.', 'box': [355, 541, 397, 551]}], 'id': 18}, {'text': 'Have you contacted your Manager/ Supervisor?', 'box': [96, 608, 398, 624], 'linking': [[19, 6], [19, 7]], 'label': 'header', 'words': [{'text': 'Have', 'box': [96, 612, 127, 623]}, {'text': 'you', 'box': [131, 613, 156, 624]}, {'text': 'contacted', 'box': [161, 612, 225, 623]}, {'text': 'your', 'box': [229, 610, 260, 623]}, {'text': 'Manager/', 'box': [264, 609, 314, 622]}, {'text': '', 'box': [314, 608, 322, 622]}, {'text': 'Supervisor?', 'box': [323, 608, 398, 621]}], 'id': 19}, {'text': 'Manager Comments:', 'box': [98, 651, 211, 665], 'linking': [[20, 21], [20, 22]], 'label': 'question', 'words': [{'text': 'Manager', 'box': [98, 654, 150, 665]}, {'text': 'Comments:', 'box': [154, 651, 211, 664]}], 'id': 20}, {'text': 'Manager, please contact suggester and forward', 'box': [232, 644, 547, 662], 'linking': [[20, 21]], 'label': 'answer', 'words': [{'text': 'Manager,', 'box': [232, 648, 288, 662]}, {'text': 'please', 'box': [296, 649, 338, 662]}, {'text': 'contact', 'box': [344, 648, 394, 662]}, {'text': 'suggester', 'box': [401, 648, 464, 661]}, {'text': 'and', 'box': [469, 647, 491, 658]}, {'text': 'forward', 'box': [497, 644, 547, 657]}], 'id': 21}, {'text': 'comments to the Quality Council.', 'box': [99, 662, 323, 677], 'linking': [[20, 22]], 'label': 'answer', 'words': [{'text': 'comments', 'box': [99, 666, 155, 677]}, {'text': 'to', 'box': [162, 665, 177, 676]}, {'text': 'the', 'box': [183, 665, 205, 675]}, {'text': 'Quality', 'box': [211, 663, 261, 676]}, {'text': 'Council.', 'box': [267, 662, 323, 676]}], 'id': 22}, {'text': 'qip . wp', 'box': [102, 823, 145, 838], 'linking': [], 'label': 'other', 'words': [{'text': 'qip', 'box': [102, 824, 123, 837]}, {'text': '.', 'box': [124, 824, 130, 838]}, {'text': 'wp', 'box': [130, 823, 145, 837]}], 'id': 23}]}
            #D:/CORD/train/json\receipt_00000.json
            image_path = file_path.replace(".json", ".png").replace("json", "image")

            # ../data/FUNSD/training_data/images\0000971160.png

            file_name = os.path.basename(image_path)
            # 0000971160.png
            image = Image.open(image_path)

            width, length = image.size

            words = []
            bboxes = []
            ner_tags = []

            for item in data["valid_line"]:
                cur_line_bboxes = []
                line_words, label = item["words"], item["category"]
                line_words = [w for w in line_words if w["text"].strip() != ""]
                if len(line_words) == 0:
                    continue
                if label == "other":
                    for w in line_words:
                        words.append(w["text"])
                        ner_tags.append("O")
                        cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), width,length))
                else:
                    words.append(line_words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(line_words[0]["quad"]), width,length))
                    for w in line_words[1:]:
                        words.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), width,length))
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)

            for i in range(len(bboxes)):
                a.add(ner_tags[i])
                fw.write(words[i] + "\t" + ner_tags[i] + "\n")
                fbw.write(
                    words[i]
                    + "\t"
                    + string_box(bboxes[i])
                    + "\n"
                )
                fiw.write(
                    words[i]
                    + "\t"
                    + actual_bbox_string(bboxes[i], width, length)
                    + "\t"
                    + file_name
                    + "\n"
                )

            fw.write("\n")
            fbw.write("\n")
            fiw.write("\n")

def seg_file(file_path,tokenizer,max_len,flag=False):
    # "../data/FUNSD/data/train.txt.tmp"
    # 分词器
    # 510 一个文档中最大文本长度
    subword_len_counter=0
    #../data/FUNSD/data\train.txt ../data/FUNSD/data\train_box.txt ../data/FUNSD/data\train_image.txt
    output_path=file_path[:-4]
    with open(file_path,"r",encoding="utf8"
              ) as f_p,open(
        output_path,"w",encoding="utf8"
    ) as fw_p:
        path_list=[]
        for line in f_p:

            line=line.rstrip()

            if flag and line:
                img = line.split("\t")[-1]
                if img not in path_list:
                    path_list.append(img)

            '''
            R&D	O
            R&D	383 91 493 175
            R&D	292 91 376 175	762 1000	0000971160.png
            '''
            if not line:#不存在的行，写入后换行，然后文档长度重新定义为0
                #将此image_path写入
                fw_p.write(line+"\n")
                subword_len_counter=0
                continue

            token=line.split("\t")[0]
            #R&D
            current_subwords_len=len(tokenizer.tokenize(token))
            #['r', '&', 'd'] 3

            if current_subwords_len==0:
                continue

            if (subword_len_counter+current_subwords_len)>max_len:#如果超过了510长度
                #print(line)#5.	B-ANSWER
                #5.	473 275 486 289	762 1000	0000999294.png
                fw_p.write("\n"+line+"\n")#另起一行，写入txt文件中
                #读到重复文档的时候，都要重新加一次进来
                if flag:
                    img = line.split("\t")[-1]
                    path_list.append(img)

                subword_len_counter=current_subwords_len#并调整到当作下一个文档重新开始
                continue

            subword_len_counter+=current_subwords_len

            fw_p.write(line+"\n")

        if flag:
            with open("test_image_path.txt",'w') as ft:
                for i in range(len(path_list)):
                    ft.write('/test/image/'+path_list[i]+'\n')


def seg(args):
    tokenizer=AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True
    )
    seg_file(
        os.path.join(args.output_dir,args.data_split+".txt.tmp"),
        tokenizer,
        args.max_len,
        False
    )
    seg_file(
        os.path.join(args.output_dir,args.data_split+"_box.txt.tmp"),
        tokenizer,
        args.max_len,
        False
    )
    seg_file(
        os.path.join(args.output_dir,args.data_split+"_image.txt.tmp"),
        tokenizer,
        args.max_len,
        True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="json")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--max_len", type=int, default=510)
    args = parser.parse_args()

    convert(args)

    seg(args)



