from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import pdfplumber
from pdfplumber.pdf import PDF
from loguru import logger
import sys
import numpy as np
# 读取文件中的表格信息(链表)
@dataclass
class TableInfo:
    """表格信息"""
    page_num: int
    page_size: Tuple[int, int]
    table_info: List[List]
    

@dataclass
class TableHash:
    """Public 表格栈"""
    def __init__(self):
        self.tables = {}

    def push(self, page: int, table: TableInfo):
        if page in self.tables:
            self.tables[page].table_info.append(table)
        else:
            self.tables[page] = table

    def pop(self) -> Optional[TableInfo]:
        tables_copy = self.tables.copy()
        self.clear()
        return tables_copy if tables_copy else None

    def is_empty(self) -> bool:
        return len(self.tables) == 0
    
    def size(self) -> int:
        return len(self.tables)
    
    def clear(self):
        self.tables.clear()
    
    
table_hash = TableHash()
# 调用数据哈希表获取表格信息

class Plumber:
    
    """PDF-Plumber 将表格信息压入栈"""
    def __init__(self):
        self.last_path = None
        self.THRESHOLD = 0.1

    def __call__(self, file_path: str):
        with pdfplumber.open(file_path) as pdf_file:
            logger.info(f"-- Pre-Processing PDF {file_path} --")
            self._load_pdf(pdf_file)

        self.last_path = file_path
    
    def _load_pdf(self, pdf_file: PDF) -> None:
        total_pages = len(pdf_file.pages)
        for page_idx, page in enumerate(pdf_file.pages):

            progress = (page_idx + 1) / total_pages * 100
            bar_length = 30
            filled_length = int(bar_length * (page_idx + 1) // total_pages)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            logger.info(f"\rProcessing: |{bar}| {progress:.1f}% ({page_idx + 1}/{total_pages})")
            
            
            size = (page.height, page.width)
            all_table = page.find_tables()
            if not all_table:
                continue
            filtered_bboxes = self._overlap_filter_with_bboxes(all_table, self.THRESHOLD)
            Table = TableInfo(page_idx, size, filtered_bboxes)
            table_hash.push(page=page_idx, table=Table)

    def _overlap_filter_with_bboxes(self, tables, threshold: float) -> List[List]:
        """查询重叠bbox对象并直接返回bbox列表"""

        sorted_tables = sorted(tables, 
                              key=lambda t: (t.bbox[2]-t.bbox[0])*(t.bbox[3]-t.bbox[1]), 
                              reverse=True)
        
        filtered_tables = []
        bbox_list = []
        
        for cur_table in sorted_tables:
            is_sub = False
            cur_bbox = cur_table.bbox
            
            # 空间预筛选：只比较可能重叠的表格
            for kept_table in filtered_tables:
                kept_bbox = kept_table.bbox
                # 快速排除明显不重叠的情况
                if (cur_bbox[2] < kept_bbox[0] or cur_bbox[0] > kept_bbox[2] or 
                    cur_bbox[3] < kept_bbox[1] or cur_bbox[1] > kept_bbox[3]):
                    continue
                
                overlap_area, min_area, valid = self._overlap_calculate_optimized(cur_bbox, kept_bbox)
                if valid and min_area > 0:
                    overlap_ratio = overlap_area / min_area
                    if overlap_ratio > threshold:
                        is_sub = True
                        break  # 提前终止内层循环
            
            if not is_sub:
                filtered_tables.append(cur_table)
                bbox_list.append(list(cur_bbox))  # 直接收集bbox，转换为list确保兼容性
                
        return bbox_list
        
    @staticmethod
    def _overlap_calculate_optimized(rect1: tuple, rect2: tuple) -> Tuple[float, float, bool]:
        """优化的方框重叠查询"""
        x0_1, y0_1, x1_1, y1_1 = rect1 
        x0_2, y0_2, x1_2, y1_2 = rect2

        # 快速边界检查 - 提前排除不重叠情况
        if x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1:
            return 0.0, 0.0, False

        # 计算重叠区域
        x_left = max(x0_1, x0_2)
        y_top = max(y0_1, y0_2)
        x_right = min(x1_1, x1_2)
        y_bottom = min(y1_1, y1_2)

        overlap_area = (x_right - x_left) * (y_bottom - y_top)

        # 计算两个矩形的面积
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

        min_area = min(area1, area2)
        return overlap_area, min_area, min_area > 0
    
    @staticmethod
    def _handel_rate(large: float, small: float, data_list: list):
        """尺寸调整方法"""
        result = []
        for data in data_list:
            ar = np.array(data)
            result.append(((small / large) * ar).tolist())  
        return result