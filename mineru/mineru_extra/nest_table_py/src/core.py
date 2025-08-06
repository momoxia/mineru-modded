from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

from .utils.bbox import calculate_overlap_area_2_minbox_area_ratio, find_min_bbox_area
from .utils import *

@dataclass
class UnitResult:
    info: Dict
    match_list: List

@dataclass
class OcrResult:
    info: Dict
    match_list: List

@dataclass
class TableRelation:
    outer: Tuple
    units: UnitResult
    relation: List
    father = []
    child = []

@dataclass
class UnitText:
    unit: Tuple
    text: str
    bbox: Tuple


@dataclass
class MidInfo:
    table_relation: Dict[int, TableRelation]
    text_macthed: Dict[int, List[UnitText]]


class NestTableCore:
    def __init__(self):
        self.cv_load = CvLoad()

    # 线关系搜索
    @staticmethod
    def _relation_search(rectH,
                         rectV):
        
        for h, line_h in rectH.items():
            for v, line_v in rectV.items():
                if line_overlap(line_h["p"], line_v["p"]):
                    line_h["r"].add(v)
                    line_v["r"].add(h)

        return rectH, rectV
    
    # 表格单元格信息获取
    @staticmethod
    def _get_table_unit(rectH,
                        rectV,
                        cluster: List):
        unit_result = []
        for clu in cluster:
            pre_match_list = []
            pre_table_unit = {}
            crowd_temp = {'H': [], 'V': []}
            for node in clu:
                crowd_temp[node[0]].append(node[1])
            
            crowd_temp["H"].sort()
            crowd_temp["V"].sort()

            group_list = permutations(crowd_temp["H"])
            for group in group_list:
                h0 = group[0]
                h1 = group[1]
                v0_set = list(rectH[h0]["r"])
                v1_set = list(rectH[h1]["r"])
                common = list(set(v0_set).intersection(set(v1_set)))
                common = sorted(common)
                v = set()
                if common is not None:
                    for idx in range(len(common) - 1):
                        v.add((common[idx], common[idx + 1]))

                for _ in v:
                    l0 = rectH[h0]["p"]
                    l1 = rectH[h1]["p"]
                    l2 = rectV[_[0]]['p']
                    l3 = rectV[_[1]]['p']
                    bbox = intersections((l0, l1, l2, l3))
                    if bbox is None:
                        continue
                    point = (h0, h1, _[0], _[1])
                    pre_table_unit[bbox] = point
                    pre_match_list.append(bbox)
            pre_match_list.sort(key=lambda item: (item[1], item[0]))
            if pre_table_unit:
                pre_unit = UnitResult(pre_table_unit, pre_match_list)
                unit_result.append(pre_unit)

        return unit_result, len(unit_result)

    # 计算表格单元关系
    @staticmethod
    def _get_table_relation(table_unit: List[UnitResult], 
                            table_num: int) -> None | Dict[int, TableRelation]:

        if table_num <= 1:
            return None
        else:
            relations: Dict[int, TableRelation] = {}
            idx_list = [i for i in range(table_num)]
            idx_group = permutations(idx_list)
            for group in idx_group:
                unit_group = (table_unit[group[0]], table_unit[group[1]])
                tem_surface_0_x = set()
                tem_surface_0_y = set()
                tem_surface_1_x = set()
                tem_surface_1_y = set()  
                ov = 0
                info_0: Dict = unit_group[0].info
                info_1: Dict = unit_group[1].info

                for bbox, point in info_0.items():
                    tem_surface_0_x.add(bbox[0])
                    tem_surface_0_y.add(bbox[1])
                    tem_surface_0_x.add(bbox[2])
                    tem_surface_0_y.add(bbox[3])

                for bbox, point in info_1.items():
                    tem_surface_1_x.add(bbox[0])
                    tem_surface_1_y.add(bbox[1])
                    tem_surface_1_x.add(bbox[2])
                    tem_surface_1_y.add(bbox[3])

                if tem_surface_0_x:
                    x00 = min(tem_surface_0_x)
                    x01 = max(tem_surface_0_x)
                    y00 = min(tem_surface_0_y)
                    y01 = max(tem_surface_0_y)

                if tem_surface_1_x:
                    x10 = min(tem_surface_1_x)
                    x11 = max(tem_surface_1_x)
                    y10 = min(tem_surface_1_y)
                    y11 = max(tem_surface_1_y)
                
                if tem_surface_0_x and tem_surface_1_x:
                    biggest_0 = (x00, y00, x01, y01)
                    biggest_1 = (x10, y10, x11, y11)
                    ov = overlap_relation(biggest_0, biggest_1)
                else:
                    pass
                
                exist_0 = relations.get(group[0], None)
                exist_1 = relations.get(group[1], None)

                if exist_0 is None:
                    unit_relate = TableRelation(outer=biggest_0,
                                                units=unit_group[0],
                                                relation=[])
                    relations[group[0]] = unit_relate

                if exist_1 is None:
                    unit_relate = TableRelation(outer=biggest_1,
                                                units=unit_group[1],
                                                relation=[])
                    relations[group[1]] = unit_relate

                # 1 是 0 的子集
                # 这里其实可以用链表结构，后人想改就改吧，C++版本会加入的
                if ov == 1:
                    sub_table = {group[1]: biggest_1}
                    relations[group[0]].relation.append(sub_table)
                    relations[group[0]].child.append(group[1])
                    relations[group[1]].father.append(group[0])

                # 0 是 1 的子集
                if ov == 2:
                    sub_table = {group[0]: biggest_0}
                    relations[group[1]].relation.append(sub_table)
                    relations[group[1]].child.append(group[0])
                    relations[group[0]].father.append(group[1])

            return relations

    # 获取表格的处理顺序
    @staticmethod
    def _get_process_order(relations: Dict[int, TableRelation])->list:
        if not relations:
            return []
        process_order: List = []
        processed: Set = set()
        table_relations: Dict[int, TableRelation] = relations.copy()

        while table_relations:
            deepest_tables = []
            for idx, relation in table_relations.items():
                if not relation.child or all(child in processed for child in relation.child):
                    deepest_tables.append(idx)
            
            if not deepest_tables:
                deepest_tables = list(table_relations.keys())
            
            process_order.extend(deepest_tables)
            
            for idx in deepest_tables:
                processed.add(idx)
                table_relations.pop(idx, None)
 
        return process_order[::-1]

    @staticmethod
    def _match_text(table_relation: TableRelation,
                    ocr_result: OcrResult,
                    matched_ocr_boxes: set) -> List[UnitText]:
 
        
        
        matched_list: List[UnitText] = []
        
        ocr_info = ocr_result.info
        ocr_match = ocr_result.match_list
        unit_result = table_relation.units.match_list
       
        for ocr_bbox in ocr_match:
            if ocr_bbox in matched_ocr_boxes:
                continue
                
            matched_bbox = {}
            for unit_bbox in unit_result:
                if calculate_overlap_area_2_minbox_area_ratio(unit_bbox, ocr_bbox) > 0.5:
                    matched_bbox[unit_bbox] = unit_bbox
            
            min_key, min_bbox = find_min_bbox_area(matched_bbox)
            
            if min_key is not None:
                text = ocr_info[ocr_bbox]
                matched_list.append(UnitText(unit=min_key, text=text, bbox=ocr_bbox))

                matched_ocr_boxes.add(ocr_bbox)

        return matched_list
    @staticmethod
    def _post_process(table_relation: TableRelation, text_macthed: List[UnitText]):
        units = table_relation.units.info
        x_set: Set = set()
        y_set: Set = set()
        sub_tables: Dict = {}
        unit_info: Dict = {}

        location: Tuple[int, int, int, int] = None
        unit_dtype: str = None
        text = None
        position: List[Tuple[float, float, float, float]] = []

        for unit_bbox, unit_lines in units.items():
            x_set.add(unit_lines[0])
            x_set.add(unit_lines[1])
            y_set.add(unit_lines[2])
            y_set.add(unit_lines[3])
        x_list = sorted(list(x_set))
        y_list = sorted(list(y_set))
        x_map = {x_list[i]: i for i, x in enumerate(x_list)}
        y_map = {y_list[i]: i for i, y in enumerate(y_list)}

        # 为每个子表找到最合适的一个单元格（基于重叠面积比例）
        table_assignments = {}  # 存储每个子表分配到的单元格
        
        for relation in table_relation.relation:
            for table_index, relation_bbox in relation.items():
                best_unit = None
                best_ratio = 0
                candidate_units = []
                
                # 收集所有完全包含子表的单元格
                for unit_bbox, unit_lines in units.items():
                    if (relation_bbox[0] >= unit_bbox[0] and
                        relation_bbox[1] >= unit_bbox[1] and
                        relation_bbox[2] <= unit_bbox[2] and
                        relation_bbox[3] <= unit_bbox[3]):
                        candidate_units.append(unit_bbox)
                
                # 如果没有完全包含的单元格，则查找部分重叠的单元格
                if not candidate_units:
                    for unit_bbox, unit_lines in units.items():
                        # 计算重叠面积与子表面积的比例
                        overlap_width = max(0, min(relation_bbox[2], unit_bbox[2]) - max(relation_bbox[0], unit_bbox[0]))
                        overlap_height = max(0, min(relation_bbox[3], unit_bbox[3]) - max(relation_bbox[1], unit_bbox[1]))
                        overlap_area = overlap_width * overlap_height
                        
                        table_area = (relation_bbox[2] - relation_bbox[0]) * (relation_bbox[3] - relation_bbox[1])
                        
                        if table_area > 0 and overlap_area > 0:
                            ratio = overlap_area / table_area
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_unit = unit_bbox
                
                # 如果有完全包含的单元格，选择面积最小的一个
                elif candidate_units:
                    min_area = float('inf')
                    for unit_bbox in candidate_units:
                        area = (unit_bbox[2] - unit_bbox[0]) * (unit_bbox[3] - unit_bbox[1])
                        if area < min_area:
                            min_area = area
                            best_unit = unit_bbox
                
                # 记录子表的分配结果
                if best_unit is not None:
                    table_assignments[table_index] = (best_unit, relation_bbox)

        # 将分配结果添加到sub_tables中
        for table_index, (unit_bbox, relation_bbox) in table_assignments.items():
            if unit_bbox not in sub_tables:
                sub_tables[unit_bbox] = []
            sub_tables[unit_bbox].append((relation_bbox, table_index))

        for text_unit in text_macthed:        
            unit_dtype = 'str'
            text = text_unit.text
            compose = units[text_unit.unit]
            location = (x_map[compose[0]],
                        x_map[compose[1]],
                        y_map[compose[2]],
                        y_map[compose[3]])
            if location in unit_info:
                unit_info[location].append((unit_dtype, text, text_unit.bbox))
            else:
                unit_info[location] = [(unit_dtype, text, text_unit.bbox)]
        
        need_sort = set()
        for unit_bbox, relations in sub_tables.items():
            unit_dtype = 'table'
            compose = units[unit_bbox]
            location = (x_map[compose[0]],
                        x_map[compose[1]],
                        y_map[compose[2]],
                        y_map[compose[3]])
            need_sort.add(location)
            
            for relation_bbox, table_index in relations:
                if location in unit_info:
                    unit_info[location].append((unit_dtype, table_index, relation_bbox))
                else:
                    unit_info[location] = [(unit_dtype, table_index, relation_bbox)]

        for need in need_sort:
            sort_target = unit_info[need]
            unit_info[need] = sorted(sort_target, key=lambda x: x[2][1])
        
        return unit_info
    def _draw_table_cells(self, table_unit: List[UnitResult], img, save_path: str = "table_cells_check.png"):
        """
        绘制表格单元格并保存图像用于检查
        """
        import cv2
        import numpy as np
        import os
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 复制原图用于绘制
        result_img = img.copy()
        
        # 为每个表格单元格绘制边界框
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, unit_result in enumerate(table_unit):
            color = colors[i % len(colors)]
            for bbox in unit_result.match_list:
                x, y, x2, y2 = bbox
                # 绘制矩形框
                cv2.rectangle(result_img, (int(x), int(y)), (int(x2), int(y2)), color, 2)
                # 添加索引标签
                cv2.putText(result_img, f'T{i}', (int(x), int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存图像
        cv2.imwrite(save_path, result_img)
        print(f"表格单元格检查图像已保存到: {save_path}")
    
    def _draw_ocr_results(self, ocr_result: OcrResult, img, save_path: str = "ocr_results_check.png"):
        """
        绘制OCR识别结果并保存图像用于检查
        """
        import cv2
        import numpy as np
        import os
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 复制原图用于绘制
        result_img = img.copy()
        
        # 绘制OCR识别结果
        for bbox in ocr_result.match_list:
            # 绘制边界框
            x, y, x2, y2 = bbox
            cv2.rectangle(result_img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 获取并绘制文本
            text = ocr_result.info[bbox]
            # 在边界框附近添加文本标签
            cv2.putText(result_img, text, (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存图像
        cv2.imwrite(save_path, result_img)
        print(f"OCR识别结果检查图像已保存到: {save_path}")
        
        
    def __call__(self, img) -> None:
        self.cv_load(img)
        # 读取线段信息
        cv_inf = self.cv_load
        ocr_inf, mat_lis = img_ocr(img)
        ocr_result = OcrResult(ocr_inf, mat_lis)
        # 读取OCR信息
        if ocr_inf:
            H = cv_inf.H
            V = cv_inf.V
            # self._draw_ocr_results(ocr_result, img, "ocr_results_check.png")
        else:
            return None
        # 关系搜索
        H, V = self._relation_search(H, V)

        # 获取表格线簇
        nest_table_cluster = dfs_search(H, V)

        # 获取单元格信息
        unit_result, tabel_num = self._get_table_unit(H, V, nest_table_cluster)
        
        # 检查嵌套关系，如果不是嵌套则返回None退出
        relation = self._get_table_relation(unit_result, tabel_num)

        #if relation is not None:
            #self._draw_table_cells(unit_result, img, "table_cells_check.png")
            
        if relation is None:
            return None
        
        # 用图方法获取处理顺序（最深层 -> 最外层）
        process_order = self._get_process_order(relation)

        matched_table: Set = set()
        matched_result : Dict[int, List[UnitText]] = {}
        for order in process_order:
            process_table: TableRelation = relation[order]
            matched_result[order] = self._match_text(process_table, ocr_result , matched_table)
        info = MidInfo(relation, matched_result)
        table_relation = info.table_relation
        text_matched = info.text_macthed
        table_info = {}
        for idx, table in table_relation.items():
            table_info[idx] = self._post_process(table, text_matched[idx])
        
        if process_order:
            main_table_index = process_order[-1] 
            main_table = table_info[main_table_index]
            sub_table = {}

            for idx in process_order[:-1]:
                #print(f"Index: {idx}")
                #print(f"Table info keys: {list(table_info[idx].keys())}")
                #print(f"Table info: {table_info[idx]}")
                sub_table[idx] = table_info[idx].copy()
                #print(f"Sub table {idx}: {sub_table[idx]}")
                #print("-" * 50)
        else:
            main_table = table_info[0] if 0 in table_info else next(iter(table_info.values()))
            sub_table = {k: v for k, v in table_info.items() if k != 0}  
        print(main_table)
        
        return (main_table, sub_table)


        

        

        
        

        

        
        
        


        

        

        






