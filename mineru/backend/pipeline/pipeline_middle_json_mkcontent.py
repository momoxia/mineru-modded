import re
from loguru import logger
import unicodedata
import itertools


from mineru.utils.config_reader import get_latex_delimiter_config
from mineru.backend.pipeline.para_split import ListLineTag
from mineru.utils.enum_class import BlockType, ContentType, MakeMode
from mineru.utils.language import detect_lang


def __is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))


def make_blocks_to_markdown(paras_of_layout,
                                      mode,
                                      img_buket_path='',
                                      ):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.INTERLINE_EQUATION:
            if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
                continue
            if para_block['lines'][0]['spans'][0].get('content', ''):
                para_text = merge_para_with_text(para_block)
            else:
                para_text += f"![]({img_buket_path}/{para_block['lines'][0]['spans'][0]['image_path']})"
        elif para_type == BlockType.IMAGE:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                # 检测是否存在图片脚注
                has_image_footnote = any(block['type'] == BlockType.IMAGE_FOOTNOTE for block in para_block['blocks'])
                # 如果存在图片脚注，则将图片脚注拼接到图片正文后面
                if has_image_footnote:
                    for block in para_block['blocks']:  # 1st.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += merge_para_with_text(block) + '  \n'
                    for block in para_block['blocks']:  # 2nd.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                    for block in para_block['blocks']:  # 3rd.拼image_footnote
                        if block['type'] == BlockType.IMAGE_FOOTNOTE:
                            para_text += '  \n' + merge_para_with_text(block)
                else:
                    for block in para_block['blocks']:  # 1st.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                    for block in para_block['blocks']:  # 2nd.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += '  \n' + merge_para_with_text(block)
        elif para_type == BlockType.TABLE:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TABLE_CAPTION:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TABLE_BODY:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.TABLE:
                                    # if processed by table model
                                    if span.get('html', ''):
                                        para_text += f"\n{span['html']}\n"
                                    elif span.get('image_path', ''):
                                        para_text += f"![]({img_buket_path}/{span['image_path']})"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TABLE_FOOTNOTE:
                        para_text += '\n' + merge_para_with_text(block) + '  '

        if para_text.strip() == '':
            continue
        else:
            # page_markdown.append(para_text.strip() + '  ')
            page_markdown.append(para_text.strip())

    return page_markdown


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)

latex_delimiters_config = get_latex_delimiter_config()

default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}

delimiters = latex_delimiters_config if latex_delimiters_config else default_delimiters

display_left_delimiter = delimiters['display']['left']
display_right_delimiter = delimiters['display']['right']
inline_left_delimiter = delimiters['inline']['left']
inline_right_delimiter = delimiters['inline']['right']

def merge_para_with_text(para_block):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.TEXT]:
                span['content'] = full_to_half(span['content'])
                block_text += span['content']
    block_lang = detect_lang(block_text)

    para_text = ''
    for i, line in enumerate(para_block['lines']):

        if i >= 1 and line.get(ListLineTag.IS_LIST_START_LINE, False):
            para_text += '  \n'

        for j, span in enumerate(line['spans']):

            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = escape_special_markdown_char(span['content'])
            elif span_type == ContentType.INLINE_EQUATION:
                if span.get('content', ''):
                    content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                if span.get('content', ''):
                    content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"

            content = content.strip()

            if content:
                langs = ['zh', 'ja', 'ko']
                # logger.info(f'block_lang: {block_lang}, content: {content}')
                if block_lang in langs: # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                    if j == len(line['spans']) - 1 and span_type not in [ContentType.INLINE_EQUATION]:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if j == len(line['spans'])-1 and span_type == ContentType.TEXT and __is_hyphen_at_line_end(content):
                            para_text += content[:-1]
                        else:  # 西方文本语境下 content间需要空格分隔
                            para_text += f'{content} '
                    elif span_type == ContentType.INTERLINE_EQUATION:
                        para_text += content
            else:
                continue

    return para_text


# def make_blocks_to_content_list(para_block, img_buket_path, page_idx):
#     para_type = para_block['type']
#     para_content = {}
#     if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
#         para_content = {
#             'type': ContentType.TEXT,
#             'text': merge_para_with_text(para_block),
#         }
#     elif para_type == BlockType.TITLE:
#         para_content = {
#             'type': ContentType.TEXT,
#             'text': merge_para_with_text(para_block),
#         }
#         title_level = get_title_level(para_block)
#         if title_level != 0:
#             para_content['text_level'] = title_level
#     elif para_type == BlockType.INTERLINE_EQUATION:
#         if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
#             return None
#         para_content = {
#             'type': ContentType.EQUATION,
#             'img_path': f"{img_buket_path}/{para_block['lines'][0]['spans'][0].get('image_path', '')}",
#         }
#         if para_block['lines'][0]['spans'][0].get('content', ''):
#             para_content['text'] = merge_para_with_text(para_block)
#             para_content['text_format'] = 'latex'
#     elif para_type == BlockType.IMAGE:
#         para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
#         for block in para_block['blocks']:
#             if block['type'] == BlockType.IMAGE_BODY:
#                 for line in block['lines']:
#                     for span in line['spans']:
#                         if span['type'] == ContentType.IMAGE:
#                             if span.get('image_path', ''):
#                                 para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
#             if block['type'] == BlockType.IMAGE_CAPTION:
#                 para_content[BlockType.IMAGE_CAPTION].append(merge_para_with_text(block))
#             if block['type'] == BlockType.IMAGE_FOOTNOTE:
#                 para_content[BlockType.IMAGE_FOOTNOTE].append(merge_para_with_text(block))
#     elif para_type == BlockType.TABLE:
#         para_content = {'type': ContentType.TABLE, 'img_path': '', BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
#         for block in para_block['blocks']:
#             if block['type'] == BlockType.TABLE_BODY:
#                 for line in block['lines']:
#                     for span in line['spans']:
#                         if span['type'] == ContentType.TABLE:
#                             if span.get('html', ''):
#                                 para_content[BlockType.TABLE_BODY] = f"{span['html']}"

#                             if span.get('image_path', ''):
#                                 para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

#             if block['type'] == BlockType.TABLE_CAPTION:
#                 para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
#             if block['type'] == BlockType.TABLE_FOOTNOTE:
#                 para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))

#     para_content['page_idx'] = page_idx

#     return para_content

_BLANK_FLAGS = {"", "空表格", "空图片", "空文本", "空行", " "}      # 末尾那个是 nbsp

# def _clean_text(txt: str) -> str:
#     """去掉前后空白与换行"""
#     return txt.replace("\u3000", " ").strip()     # 全角空格 → 普通空格
_WS_CHARS = dict.fromkeys(c for c in map(chr, range(0x110000))
                          if unicodedata.category(c) in ('Zs', 'Zl', 'Zp'))

def _clean_text(txt: str) -> str:
    # ① Unicode 归一化（半→全，兼去掉 compatibility 字符）
    txt = unicodedata.normalize('NFKC', txt)
    # ② 把所有“空白类”字符（Zs/Zl/Zp）统统变成普通空格
    txt = txt.translate(_WS_CHARS).replace('\u3000', ' ').replace('\u00A0', ' ')
    # ③ 合并多空格
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def _has_meaningful_text(block) -> bool:
    """
    False  ⇒ 该块应被视为“无意义”而被过滤掉
    """
    # ① 先做统一的空白/全角空白清洗
    txt = _clean_text(merge_para_with_text(block))     # ← 这里换成 _clean_text
    if not txt:
        return False

    # ② 明确定义的无效占位词
    if txt in _BLANK_FLAGS:            # {"", "空表格", "空图片", ...}
        return False

    # ③ 只有标点 / 空白
    if all(ch.isspace() or unicodedata.category(ch).startswith('P')
           for ch in txt):
        return False

    return True


def _table_has_body(tbl) -> bool:
    """TableBody 是否包含 html/latex 或截图"""
    for sub in tbl["blocks"]:
        if sub["type"] != BlockType.TABLE_BODY:
            continue
        for ln in sub["lines"]:
            for sp in ln["spans"]:
                if sp["type"] != ContentType.TABLE:
                    continue
                if sp.get("html") or sp.get("latex") or sp.get("image_path"):
                    return True
    return False


def para_to_standard_format_v2(para_block, img_buket_path, page_idx, block_index=None):
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX, BlockType.DISCARDED,
        BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
        BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.TITLE:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
        title_level = get_title_level(para_block)
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
            return None
        para_content = {
            'type': ContentType.EQUATION,
            'img_path': f"{img_buket_path}/{para_block['lines'][0]['spans'][0].get('image_path', '')}",
        }
        if para_block['lines'][0]['spans'][0].get('content', ''):
            para_content['text'] = merge_para_with_text(para_block)
            para_content['text_format'] = 'latex'
    elif para_type == BlockType.IMAGE:
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(_clean_text(merge_para_with_text(block)))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(_clean_text(merge_para_with_text(block)))
    elif para_type == BlockType.TABLE:
        para_content = {'type': ContentType.TABLE, 'img_path': '', BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:
                            if span.get('latex', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['latex']}"
                            elif span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))

    para_content['page_idx'] = page_idx
    if block_index:
        para_content['id'] = block_index

    return para_content


def make_page_to_content_list(page_info, img_bucket_path, page_idx):
    page_content = []
    seq_gen = itertools.count(1)   # global reading‑order fallback
    dseq    = 1                    # discarded‑block counter
    
    # Discarded blocks
    for d in page_info.get("discarded_blocks", []):
        if not _has_meaningful_text(d):  # ★ 新增
            continue
        content_id = f"D{dseq}"; dseq += 1
        page_content.append(
            para_to_standard_format_v2(d, img_bucket_path, page_idx, content_id)
        )

    table_sort = {
            BlockType.TABLE_CAPTION : 1,
            BlockType.TABLE_BODY    : 2,
            BlockType.TABLE_FOOTNOTE: 3,
    }

    # Main content
    for blk in page_info["para_blocks"]:
        block_type = blk["type"]
        if block_type in (BlockType.TEXT, BlockType.TITLE, BlockType.INTERLINE_EQUATION,
                 BlockType.LIST, BlockType.INDEX):
            content_id = next(seq_gen)
            page_content.append(
                para_to_standard_format_v2(blk, img_bucket_path, page_idx, content_id)
            )
            continue
        elif block_type == BlockType.IMAGE:
            content_id = next(seq_gen)
            page_content.append(
                para_to_standard_format_v2(blk, img_bucket_path, page_idx, content_id)
            )
            continue

        elif block_type == BlockType.TABLE:
            parts = sorted(blk["blocks"], key=lambda b: table_sort[b["type"]])
            # ① caption
            for sub in parts:
                if sub["type"] == BlockType.TABLE_CAPTION:
                    content_id = next(seq_gen)
                    page_content.append(
                        para_to_standard_format_v2(sub, img_bucket_path, page_idx, content_id)
                    )

            # ② body 复合 —— 仅当真的有内容
            if _table_has_body(blk):
                content_id = next(seq_gen)
                for sub in blk["blocks"]:
                    if sub["type"] == BlockType.TABLE_BODY:
                        page_content.append(
                            para_to_standard_format_v2(blk, img_bucket_path, page_idx, content_id)
                        )

            # ③ footnote
            for sub in parts:
                if sub["type"] == BlockType.TABLE_FOOTNOTE:
                    content_id = next(seq_gen)
                    page_content.append(
                        para_to_standard_format_v2(sub, img_bucket_path, page_idx, content_id)
                    )
            continue



    return page_content


def union_make(pdf_info_dict: list,
               make_mode: str,
               img_buket_path: str = '',
               ):
    output_content = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        page_idx = page_info.get('page_idx')
        if not paras_of_layout:
            continue
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            page_markdown = make_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            for para_block in paras_of_layout:
                # para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx)
                # if para_content:
                #     output_content.append(para_content)
                page_content = make_page_to_content_list(page_info, img_buket_path, page_idx)
                if page_content:
                    output_content.extend(page_content)
                

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.CONTENT_LIST:
        return output_content
    else:
        logger.error(f"Unsupported make mode: {make_mode}")
        return None


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level


def escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content