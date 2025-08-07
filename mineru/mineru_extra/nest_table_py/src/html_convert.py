import html


class HtmlConverter:
    def __call__(self, 
                 main_table,
                 sub_table):
        
        html_table = self.generate_html_table(main_table, sub_table)
        # 修改为简单的HTML结构，去除样式和多余标签
        full_html = f"<html><body>{html_table}</body></html>"
    
        return full_html

    def process_subtable(self, item, tables_dict, depth):
        
        """
        处理嵌套子表格逻辑
        Args:
            item: 表格项元组 ('table', table_id, ...)
            tables_dict: 所有可用表格的字典
            depth: 当前嵌套深度
        Returns:
            子表格的HTML字符串
        """

        table_id = item[1]
       
        if tables_dict and table_id in tables_dict:
            sub_table = self.generate_html_table(
                tables_dict[table_id], 
                tables_dict, 
                depth + 1
            )
            # 直接返回子表格HTML，不添加额外的包装元素
            return sub_table
        return f'[TABLE {table_id}]'

    def generate_html_table(self, table_data, tables_dict=None, depth=0):
        """
        将表格数据转换为HTML表格
        Args:
            table_data: 当前表格数据字典
            tables_dict: 所有可用表格的字典 {table_id: table_data}
            depth: 当前嵌套深度（防止无限递归）
        Returns:
            HTML表格字符串
        """
        # 安全限制：最大嵌套深度
        MAX_DEPTH = 5
        if depth > MAX_DEPTH:
            return "[MAX DEPTH EXCEEDED]"
        
        # 检查 table_data 是否为空
        if not table_data:
            return ""
        
        # 确定表格的行列数
        try:
            max_row = max(cell[1] for cell in table_data.keys())
            max_col = max(cell[3] for cell in table_data.keys())
            
        except ValueError:
            # 处理空的 table_data
            return ""
        
        # 创建空表格网格
        grid = [[{'content': '', 'rowspan': 1, 'colspan': 1, 'covered': False} 
                for _ in range(max_col)] 
                for _ in range(max_row)]
        
        # 处理每个单元格
        for coord, contents in table_data.items():
            r_start, r_end, c_start, c_end = coord
            rowspan = r_end - r_start
            colspan = c_end - c_start
            
            # 生成单元格内容（使用分离后的逻辑）
            cell_content = []
            for item in contents:
                if item[0] == 'str':
                    cell_content.append(html.escape(item[1]))
                elif item[0] == 'table':
                    # 调用子表格处理函数
                    cell_content.append(self.process_subtable(item, tables_dict, depth))
            
            # 标记主单元格
            grid[r_start][c_start] = {
                'content': '<br>'.join(cell_content),
                'rowspan': rowspan,
                'colspan': colspan,
                'covered': False
            }
            
            # 标记被合并的单元格
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if r != r_start or c != c_start:
                        grid[r][c] = {'covered': True}
        
        # 生成HTML，去除样式相关的属性，保持简洁格式
        html_output = ['<table>']
        for r in range(max_row):
            html_output.append('<tr>')
            for c in range(max_col):
                cell = grid[r][c]
                if cell.get('covered', False):
                    continue
                    
                rowspan_attr = f' rowspan="{cell["rowspan"]}"' if cell['rowspan'] > 1 else ''
                colspan_attr = f' colspan="{cell["colspan"]}"' if cell['colspan'] > 1 else ''
                
                html_output.append(f'<td{rowspan_attr}{colspan_attr}>{cell["content"]}</td>')
            html_output.append('</tr>')
        html_output.append('</table>')
        
        return ''.join(html_output)