import numpy as np
from retriv_bm25 import SparseRetriever


class DefragmentConfig:
    def __init__(
            self,
            defragmentation_method="dense_chain",
            fragments_buffer_size=2048,
            shuffle_chains=False,
            num_chains=1,
            over_fragmented_length=16,
            multihop=True,
            drop_mid_fragment=False,
            retriever_path=None,
            seed=666,
            **kwargs
    ):
        self.defragmentation_method = defragmentation_method
        self.fragments_buffer_size = fragments_buffer_size
        self.shuffle_chains = shuffle_chains
        self.num_chains = num_chains
        self.over_fragmented_length = over_fragmented_length
        self.multihop = multihop
        self.drop_mid_fragment = drop_mid_fragment
        self.retriever_path = retriever_path

        if len(kwargs) > 0:
            raise ValueError(f"invalid kwargs: {kwargs}")

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def __repr__(self):
        repr = f"DefragmentConfig:\n" \
               f"defragmentation_method: {self.defragmentation_method}\n" \
               f"fragments_buffer_size: {self.fragments_buffer_size}\n" \
               f"shuffle_chains: {self.shuffle_chains}\n" \
               f"num_chains: {self.num_chains}\n" \
               f"multihop: {self.multihop}\n" \
               f"drop_mid_fragment: {self.drop_mid_fragment}\n" \
               f"over_fragmented_length: {self.over_fragmented_length}\n" \
               f"retriever_path: {self.retriever_path}\n"
        return repr

    def load_rng_state(self, rng_state_path):
        self.rng.set_state(np.load(rng_state_path))


def bm25_defragment_retriv(retriever: SparseRetriever, fragments_buffer: list[dict],
                           chunk_size, defragment_config: DefragmentConfig, tokenizer, shift=0):
    batch_size = 2 if defragment_config.shuffle_chains else 1
    batch_cur_fragment = [fragments_buffer.pop(0) for _ in range(batch_size)]
    batch_fragments_delayed_to_combine = [[cur_frag] for cur_frag in batch_cur_fragment]
    cur_num_tokens = sum([len(cur_frag["token"]) for cur_frag in batch_cur_fragment])

    id_to_position = {item["internal_idx"]: cur_pos for cur_pos, item in enumerate(fragments_buffer)}
    rest_fragment_ids = np.array(list(id_to_position.keys()))
    rest_fragment_ids.sort()

    wait_to_recover_ids = []

    retrieval_step = 0
    while cur_num_tokens < chunk_size and len(rest_fragment_ids) > 0:
        batch_cur_retrieved_id = []
        for bid in range(batch_size):
            retrieved_ids_scores = retriever.search(
                query=batch_cur_fragment[bid]["contents"],
                return_docs=False,
                subset_doc_ids=rest_fragment_ids - shift,
                cutoff=batch_size
            )
            if len(retrieved_ids_scores) == 0:
                retrieved_ids_scores = [[rest_fragment_ids[0], 0]]
            for cur_retrieved_id, _ in retrieved_ids_scores:
                # avoid two query retrieve a same fragment
                if cur_retrieved_id not in batch_cur_retrieved_id:
                    batch_cur_retrieved_id.append(cur_retrieved_id)
                    break
        batch_cur_fragment = []
        for bid, rid in enumerate(batch_cur_retrieved_id):
            if not defragment_config.drop_mid_fragment:
                rest_fragment_ids = np.delete(rest_fragment_ids, np.where(rest_fragment_ids == rid))
                cur_retrieved_fragment = fragments_buffer[id_to_position[rid]]
                batch_cur_fragment.append(cur_retrieved_fragment)
                batch_fragments_delayed_to_combine[bid].append(cur_retrieved_fragment)
                cur_num_tokens += len(cur_retrieved_fragment["token"])
                if cur_num_tokens >= chunk_size:
                    break
            else:
                rest_fragment_ids = np.delete(rest_fragment_ids, np.where(rest_fragment_ids == rid))
                cur_retrieved_fragment = fragments_buffer[id_to_position[rid]]
                batch_cur_fragment.append(cur_retrieved_fragment)
                if retrieval_step % 2 == 0:
                    wait_to_recover_ids.append(rid)
                else:  # do not drop, add to delay_to_combine
                    batch_fragments_delayed_to_combine[bid].append(cur_retrieved_fragment)
                    cur_num_tokens += len(cur_retrieved_fragment["token"])
                    if cur_num_tokens >= chunk_size:
                        break
        retrieval_step += 1

    if cur_num_tokens < chunk_size:
        return None, None

    # recover
    if defragment_config.drop_mid_fragment:
        rest_fragment_ids = rest_fragment_ids.tolist()
        rest_fragment_ids.extend(wait_to_recover_ids)
    # remove all retrieved fragments from the buffer
    fragments_buffer = [fragments_buffer[id_to_position[rest_frag_id]] for rest_frag_id in rest_fragment_ids]

    combine_chains_order = []
    for bid in range(batch_size):
        combine_chains_order.extend([bid] * len(batch_fragments_delayed_to_combine[bid]))
    if defragment_config.shuffle_chains:
        defragment_config.rng.shuffle(combine_chains_order)

    iter_chains = [iter(chain) for chain in batch_fragments_delayed_to_combine]
    chunk = []
    last_f = None
    for cur_id in combine_chains_order:
        cur_frag = next(iter_chains[cur_id])
        chunk.extend(cur_frag["token"])
        last_f = cur_frag

    new_chunk = chunk[:chunk_size]
    rest_tokens = chunk[chunk_size:]

    if len(rest_tokens) >= defragment_config.over_fragmented_length:
        fragments_buffer.append({
            "internal_idx": last_f["internal_idx"], "contents": last_f["contents"], "token": rest_tokens,
        })
    return fragments_buffer, new_chunk


def bm25_defragment_retriv_simplified(
        retriever: SparseRetriever, fragments_buffer: list[dict],
        chunk_size, defragment_config: DefragmentConfig, tokenizer
):
    cur_fragment = fragments_buffer.pop(0)
    fragments_delayed_to_combine = [cur_fragment]
    cur_num_tokens = len(cur_fragment["token"])

    id_to_position = {item["internal_idx"]: cur_pos for cur_pos, item in enumerate(fragments_buffer)}
    rest_fragment_ids = np.array(list(id_to_position.keys()))
    rest_fragment_ids.sort()

    while cur_num_tokens < chunk_size and len(rest_fragment_ids) > 0:
        retrieved_ids_scores = retriever.search(
            query=cur_fragment["contents"],
            return_docs=False,
            subset_doc_ids=rest_fragment_ids,
            cutoff=1
        )
        if len(retrieved_ids_scores) == 0:
            cur_retrieved_id = rest_fragment_ids[0]
        else:
            cur_retrieved_id = retrieved_ids_scores[0][0]

        rest_fragment_ids = np.delete(rest_fragment_ids, np.where(rest_fragment_ids == cur_retrieved_id))
        cur_retrieved_fragment = fragments_buffer[id_to_position[cur_retrieved_id]]
        fragments_delayed_to_combine.append(cur_retrieved_fragment)
        cur_num_tokens += len(cur_retrieved_fragment["token"])
        if cur_num_tokens >= chunk_size:
            break

    if cur_num_tokens < chunk_size:
        return None, None

    fragments_buffer = [fragments_buffer[id_to_position[rest_frag_id]] for rest_frag_id in rest_fragment_ids]

    chunk = []
    for cur_fragment in fragments_delayed_to_combine:
        chunk.extend(cur_fragment["token"])

    new_chunk = chunk[:chunk_size]
    rest_tokens = chunk[chunk_size:]

    if len(rest_tokens) >= defragment_config.over_fragmented_length:
        last_f = fragments_delayed_to_combine[-1]
        fragments_buffer.append({
            "internal_idx": last_f["internal_idx"], "contents": last_f["contents"], "token": rest_tokens,
        })

    return fragments_buffer, new_chunk
