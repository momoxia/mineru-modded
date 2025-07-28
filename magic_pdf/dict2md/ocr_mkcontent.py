import re
import unicodedata
from loguru import logger
import itertools

from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.language import detect_lang
from magic_pdf.libs.markdown_utils import ocr_escape_special_markdown_char
from magic_pdf.post_proc.para_split_v3 import ListLineTag


def __is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))


def ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict: list,
                                                img_buket_path):
    markdown_with_para_and_pagination = []
    page_no = 0
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        if not paras_of_layout:
            markdown_with_para_and_pagination.append({
                'page_no':
                    page_no,
                'md_content':
                    '',
            })
            page_no += 1
            continue
        page_markdown = ocr_mk_markdown_with_para_core_v2(
            paras_of_layout, 'mm', img_buket_path)
        markdown_with_para_and_pagination.append({
            'page_no':
                page_no,
            'md_content':
                '\n\n'.join(page_markdown)
        })
        page_no += 1
    return markdown_with_para_and_pagination


def ocr_mk_markdown_with_para_core_v2(paras_of_layout,
                                      mode,
                                      img_buket_path='',
                                      ):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.Text, BlockType.List, BlockType.Index, BlockType.Discarded]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Title:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.InterlineEquation:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Image:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:  # 1st.拼image_body
                    if block['type'] == BlockType.ImageBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Image:
                                    if span.get('image_path', ''):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:  # 2nd.拼image_caption
                    if block['type'] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 3rd.拼image_footnote
                    if block['type'] == BlockType.ImageFootnote:
                        para_text += merge_para_with_text(block) + '  \n'
        elif para_type == BlockType.Table:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TableBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Table:
                                    # if processed by table model
                                    if span.get('latex', ''):
                                        para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                                    elif span.get('html', ''):
                                        para_text += f"\n\n{span['html']}\n\n"
                                    elif span.get('image_path', ''):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block) + '  \n'

        if para_text.strip() == '':
            continue
        else:
            page_markdown.append(para_text.strip() + '  ')

    return page_markdown


def detect_language(text):
    en_pattern = r'[a-zA-Z]+'
    en_matches = re.findall(en_pattern, text)
    en_length = sum(len(match) for match in en_matches)
    if len(text) > 0:
        if en_length / len(text) >= 0.5:
            return 'en'
        else:
            return 'unknown'
    else:
        return 'empty'


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


def merge_para_with_text(para_block):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.Text]:
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
            if span_type == ContentType.Text:
                content = ocr_escape_special_markdown_char(span['content'])
            elif span_type == ContentType.InlineEquation:
                content = f"${span['content']}$"
            elif span_type == ContentType.InterlineEquation:
                content = f"\n$$\n{span['content']}\n$$\n"

            content = content.strip()

            if content:
                langs = ['zh', 'ja', 'ko']
                # logger.info(f'block_lang: {block_lang}, content: {content}')
                if block_lang in langs: # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                    if j == len(line['spans']) - 1 and span_type not in [ContentType.InlineEquation]:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    if span_type in [ContentType.Text, ContentType.InlineEquation]:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if j == len(line['spans'])-1 and span_type == ContentType.Text and __is_hyphen_at_line_end(content):
                            para_text += content[:-1]
                        else:  # 西方文本语境下 content间需要空格分隔
                            para_text += f'{content} '
                    elif span_type == ContentType.InterlineEquation:
                        para_text += content
            else:
                continue
    # 连写字符拆分
    # para_text = __replace_ligatures(para_text)
    return para_text

# # Edit by XD
# # ========= ocr_mkcontent.py =========
# def para_to_standard_format_v2(
#     para_block, img_bucket_path, page_idx,
#     drop_reason=None, label=None, block_index=None
# ):
#     t = para_block["type"]
#
#     # ---------- 普通文本系 ----------
#     if t in (
#         BlockType.Text, BlockType.Title, BlockType.List,
#         BlockType.Index, BlockType.Discarded,
#         BlockType.ImageCaption, BlockType.ImageFootnote,
#         BlockType.TableCaption, BlockType.TableFootnote,
#     ):
#         item = {"type": "text", "text": merge_para_with_text(para_block)}
#         if t == BlockType.Title:
#             lvl = get_title_level(para_block)
#             if lvl:
#                 item["text_level"] = lvl
#
#     # ---------- 行间公式 ----------
#     elif t == BlockType.InterlineEquation:
#         item = {
#             "type": "equation",
#             "text": merge_para_with_text(para_block),
#             "text_format": "latex",
#         }
#
#     # ---------- Image 复合记录（父块） ----------
#     elif t == BlockType.Image:
#         body_path = ""
#         captions, footnotes = [], []
#         for sub in para_block["blocks"]:
#             if sub["type"] == BlockType.ImageBody:
#                 for ln in sub["lines"]:
#                     for sp in ln["spans"]:
#                         if sp["type"] == ContentType.Image and sp.get("image_path"):
#                             body_path = join_path(img_bucket_path, sp["image_path"])
#             elif sub["type"] == BlockType.ImageCaption:
#                 captions.append(merge_para_with_text(sub))
#             elif sub["type"] == BlockType.ImageFootnote:
#                 footnotes.append(merge_para_with_text(sub))
#         item = {
#             "type": "image",
#             "img_path": body_path,
#             "img_caption": captions,
#             "img_footnote": footnotes,
#         }
#
#     # ---------- Table 复合记录（父块） ----------
#     elif t == BlockType.Table:
#         body_html, body_img = "", ""
#         captions, footnotes = [], []
#         for sub in para_block["blocks"]:
#             st = sub["type"]
#             if st == BlockType.TableBody:
#                 for ln in sub["lines"]:
#                     for sp in ln["spans"]:
#                         if sp["type"] != ContentType.Table:
#                             continue
#                         if sp.get("latex"):
#                             body_html = f"\n\n$\n{sp['latex']}\n$\n\n"
#                         elif sp.get("html"):
#                             body_html = f"\n\n{sp['html']}\n\n"
#                         if sp.get("image_path"):
#                             body_img = join_path(img_bucket_path, sp["image_path"])
#             elif st == BlockType.TableCaption:
#                 captions.append(merge_para_with_text(sub))
#             elif st == BlockType.TableFootnote:
#                 footnotes.append(merge_para_with_text(sub))
#         item = {
#             "type": "table",
#             "table_body": body_html,
#             "table_caption": captions,
#             "table_footnote": footnotes,
#             "img_path": body_img,
#         }
#
#     # ---------- 兜底 ----------
#     else:
#         item = {"type": "text", "text": merge_para_with_text(para_block)}
#
#     # 公共字段
#     item["page_idx"] = page_idx
#     if drop_reason:
#         item["drop_reason"] = drop_reason
#     if label:
#         item["label"] = label
#     if block_index:
#         item["id"] = block_index
#     return item



# ========= ocr_mkcontent.py =========
# ========= ocr_mkcontent.py =========
# ★ 公用小工具 -----------------------------------------------------------------
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
        if sub["type"] != BlockType.TableBody:
            continue
        for ln in sub["lines"]:
            for sp in ln["spans"]:
                if sp["type"] != ContentType.Table:
                    continue
                if sp.get("html") or sp.get("latex") or sp.get("image_path"):
                    return True
    return False



# ★ para_to_standard_format_v2 (保持主体不变，仅在 caption/footnote 处过滤空串) -----
def para_to_standard_format_v2(
    para_block, img_bucket_path, page_idx,
    drop_reason=None, label=None, block_index=None
):
    t = para_block["type"]

    # ---------- 普通文本系 ------------------------------------------------------
    if t in (
        BlockType.Text, BlockType.Title, BlockType.List,
        BlockType.Index, BlockType.Discarded,
        BlockType.ImageCaption, BlockType.ImageFootnote,
        BlockType.TableCaption, BlockType.TableFootnote,
    ):
        item = {"type": "text", "text": merge_para_with_text(para_block)}
        if t == BlockType.Title:
            lvl = get_title_level(para_block)
            if lvl:
                item["text_level"] = lvl

    # ---------- 行间公式 --------------------------------------------------------
    elif t == BlockType.InterlineEquation:
        item = {
            "type": "equation",
            "text": merge_para_with_text(para_block),
            "text_format": "latex",
        }

    # ---------- Image 复合记录 ---------------------------------------------------
    elif t == BlockType.Image:
        body_path, captions, footnotes = "", [], []
        for sub in para_block["blocks"]:
            st = sub["type"]
            if st == BlockType.ImageBody:
                for ln in sub["lines"]:
                    for sp in ln["spans"]:
                        if sp["type"] == ContentType.Image and sp.get("image_path"):
                            body_path = join_path(img_bucket_path, sp["image_path"])
            elif st == BlockType.ImageCaption:
                captions.append(_clean_text(merge_para_with_text(sub)))
            elif st == BlockType.ImageFootnote:
                footnotes.append(_clean_text(merge_para_with_text(sub)))
        item = {
            "type": "image",
            "img_path": body_path,
            "img_caption": captions,
            "img_footnote": footnotes,
        }

    # ---------- Table 复合记录 ---------------------------------------------------
    elif t == BlockType.Table:
        body_html, body_img, captions, footnotes = "", "", [], []
        for sub in para_block["blocks"]:
            st = sub["type"]
            if st == BlockType.TableBody:
                for ln in sub["lines"]:
                    for sp in ln["spans"]:
                        if sp["type"] != ContentType.Table:
                            continue
                        if sp.get("latex"):
                            body_html = f"\n\n$\n{sp['latex']}\n$\n\n"
                        elif sp.get("html"):
                            body_html = f"\n\n{sp['html']}\n\n"
                        if sp.get("image_path"):
                            body_img = join_path(img_bucket_path, sp["image_path"])
            elif st == BlockType.TableCaption:
                captions.append(_clean_text(merge_para_with_text(sub)))
            elif st == BlockType.TableFootnote:
                footnotes.append(_clean_text(merge_para_with_text(sub)))
        item = {
            "type": "table",
            "table_body": body_html,
            "table_caption": captions,
            "table_footnote": footnotes,
            "img_path": body_img,
        }

    # ---------- 兜底 ------------------------------------------------------------
    else:
        item = {"type": "text", "text": merge_para_with_text(para_block)}

    # 公共字段 -------------------------------------------------------------------
    item["page_idx"] = page_idx
    if drop_reason:
        item["drop_reason"] = drop_reason
    if label:
        item["label"] = label
    if block_index:
        item["id"] = block_index
    return item


def _ensure_labels_for_page(pg):
    """Assign labels to para_blocks based on the same order as draw_layout_bbox."""
    # 1⃣ 先按原有逻辑给 discarded_blocks 打 D# 号
    dseq = itertools.count(1)
    for d in pg.get("discarded_blocks", []):
        if "label" not in d:
            d["label"] = f"D{next(dseq)}"

    # 2⃣ 构造 layout 顺序的 bbox 列表（和 draw_layout_bbox 一致）
    table_type_order = {
        'table_caption': 1,
        'table_body': 2,
        'table_footnote': 3
    }
    layout_seq = []
    for blk in pg.get("para_blocks", []):
        t = blk["type"]
        if t in (BlockType.Text, BlockType.Title,
                 BlockType.InterlineEquation, BlockType.List,
                 BlockType.Index):
            layout_seq.append((blk, blk["bbox"]))
        elif t == BlockType.Image:
            layout_seq.append((blk, blk["bbox"]))
        elif t == BlockType.Table:
            for sub in sorted(blk["blocks"], key=lambda b: table_type_order[b["type"]]):
                layout_seq.append((sub, sub["bbox"]))

    # 3⃣ 为每个 (block, bbox) 依次赋序号
    for idx, (blk, _) in enumerate(layout_seq, start=1):
        blk["label"] = str(idx)

    # 4⃣ 最后，把没被列到 layout_seq 里的块都视为无用，丢弃它们
    pg["para_blocks"] = [blk for blk in pg["para_blocks"]
                         if "label" in blk or blk["type"] == BlockType.Table]


def union_make(pdf_info_list, make_mode, drop_mode, img_bucket_path=""):
    out_items, md_parts = [], []

    for pg in pdf_info_list:
        # --- ensure the page has labels before we start consuming it
        _ensure_labels_for_page(pg)

        # ---- 整页丢弃策略 ----
        if pg.get("need_drop") and drop_mode == DropMode.SINGLE_PAGE:
            continue
        if pg.get("need_drop") and drop_mode == DropMode.WHOLE_PDF:
            raise RuntimeError(pg["drop_reason"])

        page_idx    = pg["page_idx"]
        drop_reason = pg.get("drop_reason")
        seq_gen = itertools.count(1)   # global reading‑order fallback
        dseq    = 1                    # discarded‑block counter

        # -------- ① discarded ----------
        for d in pg.get("discarded_blocks", []):
            if not _has_meaningful_text(d):  # ★ 新增
                continue
            lab = f"D{dseq}"; dseq += 1
            out_items.append(
                para_to_standard_format_v2(d, img_bucket_path, page_idx,
                                           drop_reason, lab, lab)
            )

        # -------- ② 正文 ----------
        table_sort = {
            BlockType.TableCaption : 1,
            BlockType.TableBody    : 2,
            BlockType.TableFootnote: 3,
        }


        for blk in pg["para_blocks"]:
            t = blk["type"]

            # ---- 普通文字 / 标题 / 列表 / 公式 ----
            if t in (BlockType.Text, BlockType.Title, BlockType.InterlineEquation,
                     BlockType.List, BlockType.Index):
                content_id = next(seq_gen)
                lab = str(blk.get("label", None))
                out_items.append(
                    para_to_standard_format_v2(blk, img_bucket_path, page_idx,
                                               drop_reason, lab, content_id)
                )
                continue

            # ---- Image 复合 ----
            if t == BlockType.Image:
                content_id = next(seq_gen)
                lab = str(blk.get("label", None))
                out_items.append(
                    para_to_standard_format_v2(
                        blk, img_bucket_path, page_idx,
                        drop_reason, lab, content_id
                    )
                )

            # ---- Table 复合 ----
            if t == BlockType.Table:
                parts = sorted(blk["blocks"], key=lambda b: table_sort[b["type"]])

                # ① caption
                for sub in parts:
                    if sub["type"] == BlockType.TableCaption:
                        content_id = next(seq_gen)
                        lab = str(sub.get("label", None))
                        out_items.append(
                            para_to_standard_format_v2(sub, img_bucket_path, page_idx,
                                                       drop_reason, lab, content_id)
                        )

                # ② body 复合 —— 仅当真的有内容
                if _table_has_body(blk):
                    content_id = next(seq_gen)
                    for sub in blk["blocks"]:
                        if sub["type"] == BlockType.TableBody:
                            lab = str(sub.get("label", None))
                            out_items.append(
                                para_to_standard_format_v2(blk, img_bucket_path, page_idx,
                                                           drop_reason, lab, content_id)
                            )

                # ③ footnote
                for sub in parts:
                    if sub["type"] == BlockType.TableFootnote:
                        content_id = next(seq_gen)
                        lab = str(sub.get("label", None))
                        out_items.append(
                            para_to_standard_format_v2(sub, img_bucket_path, page_idx,
                                                       drop_reason, lab, content_id)
                        )
                continue



    # -------- ③ 输出 --------
    if make_mode == MakeMode.STANDARD_FORMAT:
        return out_items

    for it in out_items:
        if it["type"] in ("text", "equation"):
            if "text_level" in it:
                prefix = "#" * int(it["text_level"])
                md_parts.append(f"{prefix} {it['text']}")
            else:
                md_parts.append(it["text"])
        elif it["type"] == "image":
            md_parts.append(f"![]({it['img_path']})")
        elif it["type"] == "table":
            md_parts.append(f"![]({it['img_path']})")
    return "\n\n".join(md_parts)


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level