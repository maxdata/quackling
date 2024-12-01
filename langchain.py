from enum import Enum

from docling.document_converter import DocumentConverter
from docling_core.types import Document as DLDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel
from typing import Iterator, Iterable, List

from .hierarchical_chunker import HierarchicalChunker, BaseChunker


class DocumentMetadata(BaseModel):
    dl_doc_hash: str


class ChunkDocMetadata(BaseModel):
    dl_doc_id: str
    path: str


class DoclingLoader(BaseLoader):
    class ParseType(str, Enum):
        MARKDOWN = "markdown"
        JSON = "json"

    def __init__(self, file_path: str | list[str], parse_type: ParseType) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._parse_type = parse_type
        self._converter = DocumentConverter()

    def _create_lc_doc_from_dl_doc(self, dl_doc: DLDocument) -> LCDocument:
        if self._parse_type == self.ParseType.MARKDOWN:
            text = dl_doc.export_to_markdown()
        elif self._parse_type == self.ParseType.JSON:
            text = dl_doc.model_dump_json()
        else:
            raise RuntimeError(f"Unexpected parse type encountered: {self._parse_type}")
        lc_doc = LCDocument(
            page_content=text,
            metadata=DocumentMetadata(
                dl_doc_hash=dl_doc.file_info.document_hash,
            ).model_dump(),
        )
        return lc_doc

    def lazy_load_pdf(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert_single(source).output
            lc_doc = self._create_lc_doc_from_dl_doc(dl_doc=dl_doc)
            yield lc_doc


class HierarchicalJSONSplitter:
    def __init__(
        self,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.chunker: BaseChunker = chunker or HierarchicalChunker()

    def split_documents(self, documents: Iterable[LCDocument]) -> List[LCDocument]:
        all_chunk_docs: list[LCDocument] = []
        for doc in documents:
            lc_doc: LCDocument = LCDocument.model_validate(doc)
            dl_doc: DLDocument = DLDocument.model_validate_json(lc_doc.page_content)
            chunk_iter = self.chunker.chunk(dl_doc=dl_doc)
            chunk_docs = [
                LCDocument(
                    page_content=chunk.text,
                    metadata=ChunkDocMetadata(
                        dl_doc_id=dl_doc.file_info.document_hash,
                        path=chunk.path,
                    ).model_dump(),
                )
                for chunk in chunk_iter
            ]
            all_chunk_docs.extend(chunk_docs)

        return all_chunk_docs