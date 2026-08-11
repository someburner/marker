import re
from copy import deepcopy
from typing import Annotated

from marker.logger import get_logger
from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.registry import get_block_class

logger = get_logger()


class SparseTableProcessor(BaseProcessor):
    """
    Detect compact, sparse text blocks that are likely small tables.

    Datasheets often encode 8-bit register diagrams as two sparse text rows:
    one row of bit positions and one row of field labels.  The layout model can
    misclassify these as Text, which prevents TableProcessor from running.
    """

    use_sparse_table_processor: Annotated[
        bool,
        "Whether to relabel sparse sequential text blocks from Text to Table.",
    ] = False
    sparse_table_min_width_ratio: Annotated[
        float,
        "Minimum page-width ratio for a sparse bitfield table candidate.",
    ] = 0.55
    sparse_table_max_height_ratio: Annotated[
        float,
        "Maximum page-height ratio for a sparse bitfield table candidate.",
    ] = 0.08
    sparse_table_max_row_gap_ratio: Annotated[
        float,
        "Maximum page-height ratio between split sparse table rows.",
    ] = 0.04
    sparse_table_min_sequence_length: Annotated[
        int,
        "Minimum number of sequential columns needed for a sparse table candidate.",
    ] = 3
    sparse_table_max_sequence_length: Annotated[
        int,
        "Maximum number of sequential columns allowed for a sparse table candidate.",
    ] = 32
    sparse_table_max_rows: Annotated[
        int,
        "Maximum number of text rows in a sparse table candidate.",
    ] = 4

    bit_header = ("7", "6", "5", "4", "3", "2", "1", "0")
    code_token_regex = re.compile(r"^[A-Za-z0-9_./:\-\[\]{}()]+$")
    range_label_regex = re.compile(r"^[A-Za-z0-9_./:\-]+(?:\[[0-9]+(?::[0-9]+)?\])$")
    numeric_suffix_regex = re.compile(r"^([A-Za-z_./:\-]+)([0-9]+)$")

    def __call__(self, document: Document):
        if not self.use_sparse_table_processor:
            return

        for page in document.pages:
            blocks = list(page.structure_blocks(document))
            for idx, block in enumerate(blocks):
                if block.removed:
                    continue
                if block.block_type != BlockTypes.Text:
                    continue

                next_block = self.next_text_block(blocks, idx)
                if self.is_split_sparse_bitfield_table(
                    block, next_block, page, document
                ):
                    self.relabel_as_table(page, block, extra_block=next_block)
                    continue

                if not self.is_sparse_bitfield_table(block, page, document):
                    continue

                self.relabel_as_table(page, block)

    @staticmethod
    def next_text_block(blocks, idx):
        for next_block in blocks[idx + 1 :]:
            if next_block.removed:
                continue
            if next_block.block_type == BlockTypes.Text:
                return next_block
            if next_block.block_type not in (BlockTypes.Line, BlockTypes.Span):
                return None
        return None

    def relabel_as_table(self, page, block, extra_block=None):
        table_cls = get_block_class(BlockTypes.Table)
        structure = deepcopy(block.structure) if block.structure else []
        polygon = deepcopy(block.polygon)
        if extra_block is not None:
            if extra_block.structure:
                structure += deepcopy(extra_block.structure)
            polygon = polygon.merge([extra_block.polygon])

        new_block = table_cls(
            polygon=polygon,
            page_id=block.page_id,
            structure=structure,
            text_extraction_method=block.text_extraction_method,
            source="heuristics",
            top_k=block.top_k,
            metadata=block.metadata,
        )
        page.replace_block(block, new_block)
        if extra_block is not None:
            page.remove_structure_items([extra_block.id])
            extra_block.removed = True
        logger.debug(f"Relabelled sparse bitfield table {block.id}")

    def is_sparse_bitfield_table(self, block, page, document: Document) -> bool:
        if block.structure is None:
            return False

        width_ratio = block.polygon.width / max(page.polygon.width, 1)
        height_ratio = block.polygon.height / max(page.polygon.height, 1)
        if width_ratio < self.sparse_table_min_width_ratio:
            return False
        if height_ratio > self.sparse_table_max_height_ratio:
            return False

        lines = [
            " ".join(line.raw_text(document).split())
            for line in block.structure_blocks(document)
            if line.block_type == BlockTypes.Line
        ]
        lines = [line for line in lines if line]
        if not 2 <= len(lines) <= self.sparse_table_max_rows:
            return False

        column_count = self.sparse_sequence_length(lines[0])
        if column_count is None:
            return False

        return all(
            self.looks_like_table_row(line, column_count) for line in lines[1:]
        )

    def is_split_sparse_bitfield_table(
        self, block, next_block, page, document: Document
    ) -> bool:
        if next_block is None:
            return False

        if block.structure is None or next_block.structure is None:
            return False

        block_lines = self.block_lines(block, document)
        next_lines = self.block_lines(next_block, document)
        if len(block_lines) != 1:
            return False

        column_count = self.sparse_sequence_length(block_lines[0])
        if column_count is None:
            return False
        if not next_lines or len(next_lines) > self.sparse_table_max_rows - 1:
            return False
        if not all(
            self.looks_like_table_row(line, column_count) for line in next_lines
        ):
            return False

        polygon = block.polygon.merge([next_block.polygon])
        width_ratio = polygon.width / max(page.polygon.width, 1)
        height_ratio = polygon.height / max(page.polygon.height, 1)
        row_gap_ratio = (
            max(0, next_block.polygon.y_start - block.polygon.y_end)
            / max(page.polygon.height, 1)
        )
        if width_ratio < self.sparse_table_min_width_ratio:
            return False
        if height_ratio > self.sparse_table_max_height_ratio:
            return False
        if row_gap_ratio > self.sparse_table_max_row_gap_ratio:
            return False

        return True

    @staticmethod
    def block_lines(block, document: Document) -> list[str]:
        lines = [
            " ".join(line.raw_text(document).split())
            for line in block.structure_blocks(document)
            if line.block_type == BlockTypes.Line
        ]
        return [line for line in lines if line]

    def is_bit_header_line(self, line: str) -> bool:
        return tuple(line.split()) == self.bit_header

    def sparse_sequence_length(self, line: str) -> int | None:
        tokens = line.split()
        if tuple(tokens) == self.bit_header:
            return len(tokens)

        sequence_length = self.sequential_integer_tokens(tokens)
        if sequence_length is not None:
            return sequence_length

        sequence_length = self.sequential_suffix_tokens(tokens)
        if sequence_length is not None:
            return sequence_length

        return self.sequential_split_suffix_tokens(tokens)

    def sequential_integer_tokens(self, tokens: list[str]) -> int | None:
        if not all(token.isdigit() for token in tokens):
            return None
        return self.valid_sequence_length([int(token) for token in tokens])

    def sequential_suffix_tokens(self, tokens: list[str]) -> int | None:
        matches = [self.numeric_suffix_regex.match(token) for token in tokens]
        if not all(matches):
            return None

        prefixes = [match.group(1) for match in matches if match]
        if len(set(prefixes)) != 1 or not self.is_code_prefix(prefixes[0]):
            return None

        values = [int(match.group(2)) for match in matches if match]
        return self.valid_sequence_length(values)

    def sequential_split_suffix_tokens(self, tokens: list[str]) -> int | None:
        if len(tokens) % 2:
            return None

        prefixes = tokens[0::2]
        suffixes = tokens[1::2]
        if len(set(prefixes)) != 1 or not self.is_code_prefix(prefixes[0]):
            return None
        if not all(suffix.isdigit() for suffix in suffixes):
            return None

        return self.valid_sequence_length([int(suffix) for suffix in suffixes])

    def valid_sequence_length(self, values: list[int]) -> int | None:
        if not (
            self.sparse_table_min_sequence_length
            <= len(values)
            <= self.sparse_table_max_sequence_length
        ):
            return None
        if len(values) < 2:
            return None

        diffs = [next_value - value for value, next_value in zip(values, values[1:])]
        if set(diffs) not in ({1}, {-1}):
            return None
        return len(values)

    @staticmethod
    def is_code_prefix(prefix: str) -> bool:
        return prefix.upper() == prefix and any(char.isalpha() for char in prefix)

    def looks_like_table_row(self, line: str, column_count: int) -> bool:
        tokens = line.split()
        if len(tokens) == 1:
            return bool(self.range_label_regex.match(tokens[0]))
        if len(tokens) == column_count:
            return all(self.code_token_regex.match(token) for token in tokens)
        if len(tokens) == column_count * 2:
            return all(
                self.code_token_regex.match(label) and bit.isdigit()
                for label, bit in zip(tokens[0::2], tokens[1::2], strict=True)
            )
        return False
