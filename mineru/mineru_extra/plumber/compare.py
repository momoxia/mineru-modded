MODE_MAP = {'max','rect1', 'rect2'}

def overlap_calculate_return(rect1,
                             rect2, 
                             mode='max', 
                             overlap_threshold=0.3):
        """方框重叠查询"""
        if mode not in MODE_MAP:
            raise ValueError('Mode should be one of %s' % MODE_MAP.keys())
        
    
        x0_1, y0_1, x1_1, y1_1 = rect1 
        x0_2, y0_2, x1_2, y1_2 = rect2

        # 快速边界检查 - 提前排除不重叠情况
        if x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1:
            return None

        # 计算重叠区域
        x_left = max(x0_1, x0_2)
        y_top = max(y0_1, y0_2)
        x_right = min(x1_1, x1_2)
        y_bottom = min(y1_1, y1_2)

        overlap_area = (x_right - x_left) * (y_bottom - y_top)

        # 计算两个矩形的面积
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

        has_overlap = False

        if overlap_area > 0:
             overlap_ratio = overlap_area / min(area1, area2)
             if overlap_ratio >= overlap_threshold:
                has_overlap = True

        if has_overlap:
            # 返回最大空间
            if mode == "max":
                 x_0 = min(x0_1, x0_2)
                 y_0 = min(y0_1, y0_2)
                 x_1 = max(x1_1, x1_2)
                 y_1 = max(y1_1, y1_2)

                 return [x_0, y_0, x_1, y_1]
            
            # 返回第一个框
            elif mode == "rect1":
                 return [x0_1, y0_1, x1_1, y1_1]
            
            # 返回第二个框
            elif mode == "rect2":
                 return [x0_2, y0_2, x1_2, y1_2]
             
        
        
        
              