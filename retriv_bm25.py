import random
from typing import Dict, Iterable, List, Set, Union
import numba as nb
import numpy as np
import orjson
from numba.typed import List as TypedList
import subprocess
from retriv.base_retriever import BaseRetriever
from retriv.paths import docs_path, sr_state_path
from retriv.sparse_retriever.build_inverted_index import build_inverted_index
from retriv.sparse_retriever.preprocessing import (
    get_stemmer,
    get_stopwords,
    get_tokenizer,
    preprocessing,
    preprocessing_multi,
)
from retriv.sparse_retriever.sparse_retrieval_models.bm25 import bm25
from retriv.sparse_retriever.sparse_retrieval_models.tf_idf import tf_idf


class SparseRetriever(BaseRetriever):
    def __init__(
            self,
            index_name: str = "new-index",
            model: str = "bm25",
            min_df: int = 1,
            tokenizer: Union[str, callable] = "whitespace",
            stemmer: Union[str, callable] = "english",
            stopwords: Union[str, List[str], Set[str]] = "english",
            do_lowercasing: bool = True,
            do_ampersand_normalization: bool = True,
            do_special_chars_normalization: bool = True,
            do_acronyms_normalization: bool = True,
            do_punctuation_removal: bool = True,
            hyperparams: dict = None,
    ):
        """The Sparse Retriever is a traditional searcher based on lexical matching. It supports BM25, the retrieval model used by major search engines libraries, such as Lucene and Elasticsearch. retriv also implements the classic relevance model TF-IDF for educational purposes.

        Args:
            index_name (str, optional): [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index. Defaults to "new-index".

            model (str, optional): defines the retrieval model to use for searching (`bm25` or `tf-idf`). Defaults to "bm25".

            min_df (int, optional): terms that appear in less than `min_df` documents will be ignored. If integer, the parameter indicates the absolute count. If float, it represents a proportion of documents. Defaults to 1.

            tokenizer (Union[str, callable], optional): [tokenizer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable tokenizer or disable tokenization by setting the parameter to `None`. Defaults to "whitespace".

            stemmer (Union[str, callable], optional): [stemmer](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to use during preprocessing. You can pass a custom callable stemmer or disable stemming setting the parameter to `None`. Defaults to "english".

            stopwords (Union[str, List[str], Set[str]], optional): [stopwords](https://github.com/AmenRa/retriv/blob/main/docs/text_preprocessing.md) to remove during preprocessing. You can pass a custom stop-word list or disable stop-words removal by setting the parameter to `None`. Defaults to "english".

            do_lowercasing (bool, optional): whether or not to lowercase texts. Defaults to True.

            do_ampersand_normalization (bool, optional): whether to convert `&` in `and` during pre-processing. Defaults to True.

            do_special_chars_normalization (bool, optional): whether to remove special characters for letters, e.g., `übermensch` → `ubermensch`. Defaults to True.

            do_acronyms_normalization (bool, optional): whether to remove full stop symbols from acronyms without splitting them in multiple words, e.g., `P.C.I.` → `PCI`. Defaults to True.

            do_punctuation_removal (bool, optional): whether to remove punctuation. Defaults to True.

            hyperparams (dict, optional): Retrieval model hyperparams. If `None`, it is automatically set to `{b: 0.75, k1: 1.2}`. Defaults to None.
        """

        assert model.lower() in {"bm25", "tf-idf"}
        assert min_df > 0, "`min_df` must be greater than zero."
        self.init_args = {
            "model": model.lower(),
            "min_df": min_df,
            "index_name": index_name,
            "do_lowercasing": do_lowercasing,
            "do_ampersand_normalization": do_ampersand_normalization,
            "do_special_chars_normalization": do_special_chars_normalization,
            "do_acronyms_normalization": do_acronyms_normalization,
            "do_punctuation_removal": do_punctuation_removal,
            "tokenizer": tokenizer,
            "stemmer": stemmer,
            "stopwords": stopwords,
        }

        self.model = model.lower()
        self.min_df = min_df
        self.index_name = index_name

        self.do_lowercasing = do_lowercasing
        self.do_ampersand_normalization = do_ampersand_normalization
        self.do_special_chars_normalization = do_special_chars_normalization
        self.do_acronyms_normalization = do_acronyms_normalization
        self.do_punctuation_removal = do_punctuation_removal

        self.tokenizer = get_tokenizer(tokenizer)
        self.stemmer = get_stemmer(stemmer)
        self.stopwords = [self.stemmer(sw) for sw in get_stopwords(stopwords)]

        self.id_mapping = None
        self.inverted_index = None
        self.vocabulary = None
        self.doc_count = None
        self.doc_lens = None
        self.avg_doc_len = None
        self.relative_doc_lens = None
        self.doc_index = None

        self.file_list = None

        self.preprocessing_kwargs = {
            "tokenizer": self.tokenizer,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "do_lowercasing": self.do_lowercasing,
            "do_ampersand_normalization": self.do_ampersand_normalization,
            "do_special_chars_normalization": self.do_special_chars_normalization,
            "do_acronyms_normalization": self.do_acronyms_normalization,
            "do_punctuation_removal": self.do_punctuation_removal,
        }

        self.preprocessing_pipe = preprocessing_multi(**self.preprocessing_kwargs)

        self.hyperparams = dict(b=0.75, k1=1.2) if hyperparams is None else hyperparams

        self.maximum_query_length = None

    def set_maximum_query_length(self, qlen):
        self.maximum_query_length = qlen

    def save(self) -> None:
        """Save the state of the retriever to be able to restore it later."""

        state = {
            "init_args": self.init_args,
            "id_mapping": self.id_mapping,
            "doc_count": self.doc_count,
            "inverted_index": self.inverted_index,
            "vocabulary": self.vocabulary,
            "doc_lens": self.doc_lens,
            "relative_doc_lens": self.relative_doc_lens,
            "hyperparams": self.hyperparams,
        }

        np.savez_compressed(sr_state_path(self.index_name), state=state)

    @staticmethod
    def load(index_name: str = "new-index"):
        """Load a retriever and its index.

        Args:
            index_name (str, optional): Name of the index. Defaults to "new-index".

        Returns:
            SparseRetriever: Sparse Retriever.
        """

        state = np.load(sr_state_path(index_name), allow_pickle=True)["state"][()]

        se = SparseRetriever(**state["init_args"])
        # se.initialize_doc_index()
        se.id_mapping = state["id_mapping"]
        se.doc_count = state["doc_count"]
        se.inverted_index = state["inverted_index"]
        se.vocabulary = set(se.inverted_index)
        se.doc_lens = state["doc_lens"]
        se.relative_doc_lens = state["relative_doc_lens"]
        se.hyperparams = state["hyperparams"]

        state = {
            "init_args": se.init_args,
            "id_mapping": se.id_mapping,
            "doc_count": se.doc_count,
            "inverted_index": se.inverted_index,
            "vocabulary": se.vocabulary,
            "doc_lens": se.doc_lens,
            "relative_doc_lens": se.relative_doc_lens,
            "hyperparams": se.hyperparams,
        }

        return se

    def index_aux(self, show_progress: bool = True):
        """Internal usage."""
        # collection = read_jsonl(
        #     docs_path(self.index_name),
        #     generator=True,
        #     callback=lambda x: x["text"],
        # )
        collection = self.multi_file_iterator(
            self.file_list, callback=lambda x: x["contents"]
        )

        # Preprocessing --------------------------------------------------------
        collection = self.preprocessing_pipe(collection, generator=True)

        # Inverted index -------------------------------------------------------
        (
            self.inverted_index,
            self.doc_lens,
            self.relative_doc_lens,
        ) = build_inverted_index(
            collection=collection,
            n_docs=self.doc_count,
            min_df=self.min_df,
            show_progress=show_progress,
        )
        self.avg_doc_len = np.mean(self.doc_lens, dtype=np.float32)
        self.vocabulary = set(self.inverted_index)

    def initialize_id_mapping(self, file_list):
        ids = self.multi_file_iterator(
            file_list,
            callback=lambda x: x["id"],
        )
        self.id_mapping = dict(enumerate(ids))

    @staticmethod
    def multi_file_iterator(file_list, callback=None):
        for file in file_list:
            with open(file, "r") as fn:
                while True:
                    x = fn.readline()
                    if x == "":
                        break
                    if callback is None:
                        yield orjson.loads(x)
                    else:
                        yield callback(orjson.loads(x))

    @staticmethod
    def check_id_continuous(file_list):
        process = subprocess.Popen(['head', '-n', '1', file_list[0]],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if orjson.loads(stdout)['id'] != 0:
            return False
        for fid in range(0, len(file_list) - 1):
            process = subprocess.Popen(['tail', '-n', '1', file_list[fid]],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            tail_id = orjson.loads(stdout)['id']
            process = subprocess.Popen(['head', '-n', '1', file_list[fid + 1]],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            next_head_id = orjson.loads(stdout)['id']
            if next_head_id != tail_id + 1:
                return False
        return True

    def index(
            self,
            file_list: list,
            callback: callable = None,
            show_progress: bool = True,
            fast_id_mapping: dict = None,
    ):
        """Index a given collection of documents.

        Args:
            collection (Iterable): collection of documents to index.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            SparseRetriever: Sparse Retriever.
        """

        # [zy-7/10/2023] do not save data again
        # self.save_collection(collection, callback)
        # self.initialize_doc_index()
        self.file_list = file_list
        if len(self.file_list) > 1:
            print("initialize_id_mapping")
            if self.check_id_continuous(file_list) is True:
                print("id is continuous")
                process = subprocess.Popen(['tail', '-n', '1', file_list[-1]],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                last_id = orjson.loads(stdout)['id']
                self.id_mapping = {idx: idx for idx in range(last_id + 1)}  # total line num is last_id + 1
            else:
                print("id is not continuous")
                self.initialize_id_mapping(file_list)
        else:
            process = subprocess.Popen(['head', '-n', '1', file_list[0]],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            first_line = orjson.loads(stdout)
            process = subprocess.Popen(['tail', '-n', '1', file_list[0]],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            last_line = orjson.loads(stdout)
            assert "internal_idx" in first_line
            assert first_line['internal_idx'] == 0
            last_id = last_line['internal_idx']
            self.id_mapping = {idx: idx for idx in range(last_id + 1)}

        self.doc_count = len(self.id_mapping)
        self.index_aux(show_progress)
        self.save()
        return self

    def index_file(
            self, path: str, callback: callable = None, show_progress: bool = True
    ):
        """Index the collection contained in a given file.

        Args:
            path (str): path of file containing the collection to index.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            SparseRetriever: Sparse Retriever
        """

        collection = self.collection_generator(path=path, callback=callback)
        return self.index(collection=collection, show_progress=show_progress)

    def query_preprocessing(self, query: str) -> List[str]:
        """Internal usage."""
        return preprocessing(query, **self.preprocessing_kwargs)

    def get_term_doc_freqs(self, query_terms: List[str]) -> nb.types.List:
        """Internal usage."""
        return TypedList([self.inverted_index[t]["tfs"] for t in query_terms])

    def get_doc_ids(self, query_terms: List[str]) -> nb.types.List:
        """Internal usage."""
        return TypedList([self.inverted_index[t]["doc_ids"] for t in query_terms])

    def search(self, query: str, return_docs: bool = True, cutoff: int = 100,
               subset_doc_ids: np.ndarray = None) -> List:
        # [zy-7/10/2023] add subset_doc_ids for implement buffer retrieval

        """Standard search functionality.

        Args:
            query (str): what to search for.

            return_docs (bool, optional): wether to return the texts of the documents. Defaults to True.

            cutoff (int, optional): number of results to return. Defaults to 100.

        Returns:
            List: results.
        """
        query_terms = self.query_preprocessing(query)
        query_terms = random.sample(query_terms, min(len(query_terms), self.maximum_query_length))
        if not query_terms:
            return {}
        query_terms = [t for t in query_terms if t in self.vocabulary]
        if not query_terms:
            return {}

        doc_ids = self.get_doc_ids(query_terms)
        term_doc_freqs = self.get_term_doc_freqs(query_terms)

        if self.model == "bm25":
            unique_doc_ids, scores = bm25(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                relative_doc_lens=self.relative_doc_lens,
                doc_count=self.doc_count,
                cutoff=cutoff,
                subset_doc_ids=subset_doc_ids,
                **self.hyperparams,
            )
        elif self.model == "tf-idf":
            unique_doc_ids, scores = tf_idf(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                doc_lens=self.doc_lens,
                cutoff=cutoff,
            )
        else:
            raise NotImplementedError()

        # unique_doc_ids = self.map_internal_ids_to_original_ids(unique_doc_ids)

        if not return_docs:
            return list(zip(unique_doc_ids, scores))

        return self.prepare_results(unique_doc_ids, scores)
