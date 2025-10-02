import os
import pandas as pd
import torch
from model.base_model import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import h5py
from tqdm import tqdm
from data.dataset_factory import get_dataset_factory
import faiss


class KGEmbedder(BaseModel):
    """
    A class used to generate entity linking for all specified mentions to a knowledge graph/database.
    """
    ref_db_col_names = ("CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                        "CODE", "STR", "SRL", "SUPPRESS", "CVF")
    ref_db_id_col = "CUI"
    ref_db_lang_col = "LAT"
    ref_db_lang = "ENG"
    ref_db_str_col = "STR"

    def __init__(self, exp_file=None):
        super().__init__(exp_file=exp_file)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ref_db_file = self.params['data']['ref_db_file']
        self.ref_db_embeddings_file = self.params['inf']['ref_db_embeddings_file']
        self.kg_path = self.params['data']['kg_path']
        self.kg_entity_embeddings_file = self.params['inf']['kg_entity_embeddings_file']
        self.kg_relation_embeddings_file = self.params['inf']['kg_relation_embeddings_file']
        self.index = None
        self.data = None
        self.entity_ids = None
        self.entity_names = None
    
    def setup(self):
        super().setup()
        self.setup_data()
        self.setup_model()

    def run(self):
        kg_entity_embeddings, kg_relation_embeddings = self.generate_kg_embeddings()
        if self.params['inf']['link_to_ref_db']:
            # TODO: Implement this
            self.generate_all_linked_entities()

    def generate_kg_embeddings(self):
        # Generate and save KG entity embeddings
        if os.path.exists(self.kg_entity_embeddings_file) and not self.params['inf']['kg_embeddings_rewrite']:
            print(f"Loading kg entity embeddings from {self.kg_entity_embeddings_file}.")
            kg_entity_embeddings = self.load_kg_embeddings(self.kg_entity_embeddings_file)
        else:
            kg_entity_embeddings = self.generate_and_save_kg_entry_embeddings(self.kg_entity_embeddings_file,
                                                                              self.data.get_entries() if self.params['inf']['add_relations_to_entities'] else self.data.get_entities(),
                                                                              desc_string="Generating kg entity embeddings")
        # Generate and save KG relation embeddings
        if os.path.exists(self.kg_relation_embeddings_file) and not self.params['inf']['kg_embeddings_rewrite']:
            print(f"Loading kg relation embeddings from {self.kg_relation_embeddings_file}.")
            kg_relation_embeddings = self.load_kg_embeddings(self.kg_relation_embeddings_file)
        else:
            kg_relation_embeddings = self.generate_and_save_kg_entry_embeddings(self.kg_relation_embeddings_file,
                                                                                self.data.get_relations(),
                                                                                desc_string="Generating kg relation embeddings")
        return kg_entity_embeddings, kg_relation_embeddings

    def generate_and_save_kg_entry_embeddings(self, embeddings_file, kg_entries_gen,
                                              desc_string="Generating kg entry embeddings"):
        kg_entry_embedding_dict = dict()
        with h5py.File(embeddings_file, 'w') as f:
            for raw_entry in tqdm(kg_entries_gen, desc=desc_string):
                raw_entry_name = self.get_kg_entry_name(raw_entry)
                safe_entry_name = self.make_h5_safe_name(raw_entry_name)
                kg_entry_name = self.convert_h5_safe_name_to_name(safe_entry_name)
                # skip if already seen
                if kg_entry_name in kg_entry_embedding_dict:
                    continue
                entry_embedding_arr = self.get_kg_entry_embeddings(raw_entry)  # Shape: [D]
                f.create_dataset(safe_entry_name, data=entry_embedding_arr)
                kg_entry_embedding_dict[kg_entry_name] = entry_embedding_arr
        return kg_entry_embedding_dict

    def load_kg_embeddings(self, embeddings_file):
        """Load kg embeddings from a file"""
        embeddings_dict = dict()
        with h5py.File(embeddings_file, 'r') as f:
            for kg_entry in f.keys():
                embedding = f[kg_entry][:]
                embeddings_dict[self.convert_h5_safe_name_to_name(kg_entry)] = embedding
        return embeddings_dict

    def retrieve_top_kg_entries(self, question_text):
        """Perform approximate nearest neighbors search"""
        assert self.index is not None, "Index not set. Call set_index_for_kg_embeddings() first."
        question_embedding = self.get_question_embedding(question_text)
        sim_lst, matching_index_lst = self.index.search(np.array([question_embedding]), self.params['inf']['top_k_entries'])
        matching_index_lst, sim_lst = matching_index_lst[0].tolist(), sim_lst[0].tolist()
        return [self.get_kg_entry_by_index(index) for index, sim in zip(matching_index_lst, sim_lst) if (self.params['inf']['sim_threshold'] == None) or (sim >= self.params['inf']['sim_threshold'])]

    def set_index_for_kg_embeddings(self):
        use_entity_index = self.params['inf']['use_entity_index']
        use_relation_index = self.params['inf']['use_relation_index']
        assert (use_entity_index and (not use_relation_index)) or (use_relation_index and (not use_entity_index)), "Only one of use_entity_index or use_relation_index must be True."
        kg_entity_embeddings, kg_relation_embeddings = self.generate_kg_embeddings()
        if use_entity_index:
            self._set_index_for_kg_embeddings(kg_entity_embeddings, self.data.get_entities())
        elif use_relation_index:
            self._set_index_for_kg_embeddings(kg_relation_embeddings, self.data.get_relations())
        else:
            raise ValueError("Either use_entity_index or use_relation_index must be True.")

    def _set_index_for_kg_embeddings(self, kg_embeddings, kg_entries_gen):
        """Generate an index for the kg embeddings"""
        kg_entry_embeddings_arr = []
        for raw_kg_entry in kg_entries_gen:
            # colect proper entity name
            raw_kg_entry_name = self.get_kg_entry_name(raw_kg_entry)
            kg_entry_safe_name = self.make_h5_safe_name(raw_kg_entry_name)
            kg_entry_name = self.convert_h5_safe_name_to_name(kg_entry_safe_name)
            # load embeddings
            entry_embedding = kg_embeddings[kg_entry_name]
            kg_entry_embeddings_arr.append(np.squeeze(entry_embedding))
        kg_entry_embeddings_arr = np.stack(kg_entry_embeddings_arr)
        assert len(kg_entry_embeddings_arr.shape) == 2, f"Expected 2D array for kg_entry_embeddings_arr, got {kg_entry_embeddings_arr.shape}"
        if self.params['inf']['index_type'] == 'L2':
            kg_entry_index = faiss.IndexFlatL2(kg_entry_embeddings_arr.shape[1])
        elif self.params['inf']['index_type'] == 'dot':
            kg_entry_index = faiss.IndexFlatIP(kg_entry_embeddings_arr.shape[1])
        else:
            raise ValueError(f"Invalid index type '{self.params['inf']['index_type']}'.")
        kg_entry_index.add(kg_entry_embeddings_arr)
        self.index = kg_entry_index

    def get_kg_entry_by_index(self, entry_index):
        if self.params['inf']['use_entity_index']:
            return self.data.get_entity_by_index(entry_index)
        elif self.params['inf']['use_relation_index']:
            return self.data.get_relation_by_index(entry_index)
        else:
            raise ValueError("Either use_entity_index or use_relation_index must be True.")

    def get_kg_entry_name(self, entry_lst):
        return " ".join(entry_lst)

    def make_h5_safe_name(self, name):
        """Convert a name into a safe format for HDF5 dataset names"""
        safe_name = name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('.', '_')
        return safe_name

    def convert_h5_safe_name_to_name(self, safe_name):
        """Convert a safe name back to its original format"""
        name = safe_name.replace('_', ' ')
        return name

    def get_kg_entry_embeddings(self, kg_entry):
        kg_entry_embeddings = []
        kg_entry_head = kg_entry[0]
        kg_entry_connector = kg_entry[1] if len(kg_entry) > 1 else None
        kg_entry_tail = kg_entry[2] if len(kg_entry) > 2 else None

        # Get embeddings for each component
        head_embedding = self.get_mention_wo_context_embeddings(kg_entry_head)
        kg_entry_embeddings.append(head_embedding)
        if kg_entry_connector is not None:
            connector_embedding = self.get_mention_wo_context_embeddings(kg_entry_connector)
            kg_entry_embeddings.append(connector_embedding)
        if kg_entry_tail is not None:
            tail_embedding = self.get_mention_wo_context_embeddings(kg_entry_tail)
            kg_entry_embeddings.append(tail_embedding)
        kg_entry_embeddings = np.stack(kg_entry_embeddings)  # Shape: [num_components, D]
        return kg_entry_embeddings

    # TODO: Implement/Fix this
    def generate_all_linked_entities(self):
        """Given all the mention paragraphs, generate all the linked entities for all the mentions in all the paragraphs
        and save the results to a file"""
        model = self.load_sentence_model()
        paragraphs = pd.read_csv(self.data_path)
        linked_entities = []

        for _, row in paragraphs.iterrows():
            paragraph = row['text']  # Updated to use the correct column name 'text'
            mentions = self.extract_mentions(paragraph)
            for mention in mentions:
                linked_entity = self.get_linked_entity(model, paragraph, mention)
                linked_entities.append({
                    'pid': row['pid'],
                    'mention': mention,
                    'linked_id': linked_entity
                })
        result_df = pd.DataFrame(linked_entities)
        result_df.to_csv(self.params['inf']['save_file'], index=False)

    # TODO: Implement/Fix this
    def get_linked_entity(self, mention, paragraph):
        """Get the linked entities for all the mentions in the paragraph"""
        mention_embedding = self.get_mention_w_context_embeddings(mention, paragraph)
        top_k_entities = self.get_top_k_linked_entity(mention_embedding, k=self.params['inf']['top_k_linked_entities'])
        linked_entity = self.get_top_linked_entity(mention_embedding, top_k_entities)
        return linked_entity

    def get_question_embedding(self, question_text):
        return self.get_mention_wo_context_embeddings(question_text)

    def get_mention_wo_context_embeddings(self, mention):
        """Get the embeddings for the mention without any context (useful for db embeddings)"""
        if not mention:
            mention = ""
        with torch.no_grad():
            mention_embeddings = self.model.encode(mention)
            if self.params['inf']['normalize_embeddings']:
                mention_embeddings = mention_embeddings/np.linalg.norm(mention_embeddings)
        return mention_embeddings

    def get_mention_w_context_embeddings(self, mention, context):
        """Get the embeddings for the mention with context (useful for embeddings within a sentence/paragraph)"""
        context_embeddings, context_token_ids = self.get_context_embeddings(context)
        mention_embedding = self.extract_mention_embeddings_from_context_embeddings(mention, context_embeddings,
                                                                                    context_token_ids)

        return mention_embedding

    # TODO: Implement/Fix this
    def get_top_k_linked_entity(self, mention_embedding, k=10, threshold=0.95):
        """Get the top-k average db_embeddings for the mention with cosine similarity above threshold"""
        similarities = cosine_similarity(mention_embedding.cpu().reshape(1, -1).numpy(), self.db_embeddings.cpu().numpy())  # Ensure CPU numpy array
        top_k_indices = np.argsort(similarities[0])[-k:]
        top_k_similarities = similarities[0][top_k_indices]
        top_k_entities = top_k_indices[top_k_similarities > threshold]
        return top_k_entities

    # TODO: Implement/Fix this
    def get_top_linked_entity(self, mention_embedding, top_k_linked_entities):
        """Get the top linked entity alias from the top-k linked entities"""
        if len(top_k_linked_entities) == 0:
            return None
        best_match_idx = top_k_linked_entities[0]
        return self.df.iloc[best_match_idx]['CUI']

    # TODO: Implement/Fix this
    def get_db_embeddings_average_per_linked_entity(self):
        """Get the average embeddings for all the aliases of the entity"""
        entity_ids = self.df['CUI'].unique()
        embeddings = []

        for entity_id in entity_ids:
            aliases = self.df[self.df['CUI'] == entity_id]['STR']
            alias_embeddings = [self.get_mention_wo_context_embeddings(alias) for alias in aliases]
            average_embedding = torch.mean(torch.stack(alias_embeddings).to(self.device), dim=0)
            embeddings.append(average_embedding)

        return torch.stack(embeddings)

    def get_context_embeddings(self, context):
        context_tokens = self.tokenizer(context, return_tensors='pt', add_special_tokens=True).to(self.device)
        context_token_ids = context_tokens['input_ids']
        context_attention_mask = context_tokens['attention_mask']
        context_embeddings = self.get_text_embeddings(context_token_ids, context_attention_mask)
        return context_embeddings, context_token_ids

    def extract_mention_embeddings_from_context_embeddings(self, mention, context_embeddings, context_token_ids):
        mention_token_ids = self.tokenizer.encode(mention, return_tensors='pt', add_special_tokens=True).to(self.device)
        mention_start_pos = self._extract_start_pos_of_mention_mention_ids(mention_token_ids[0], context_token_ids[0])

        if self.params['inf']['mention_embedding_comb_method'] == "start":
            mention_embedding = context_embeddings[0, mention_start_pos]
        elif self.params['inf']['mention_embedding_comb_method'] == "mean":
            mention_end_pos = mention_start_pos + len(mention_token_ids)
            mention_embedding = context_embeddings[0, mention_start_pos:mention_end_pos].mean(dim=0)
        else:
            raise ValueError(f"Invalid method '{self.params['inf']['mention_embedding_comb_method']}'.")
        return mention_embedding

    def get_text_embeddings(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.cpu()
        return outputs

    def _extract_start_pos_of_mention_mention_ids(self, mention_token_ids, context_token_ids):
        mention_start_pos = -1
        context_less_mention_len = len(context_token_ids) - len(mention_token_ids)
        for cand_mention_start_pos in range(context_less_mention_len):
            cand_mention_end_pos = cand_mention_start_pos + len(mention_token_ids)
            if torch.equal(context_token_ids[cand_mention_start_pos:cand_mention_end_pos], mention_token_ids):
                mention_start_pos = cand_mention_start_pos
                break
        if mention_start_pos == -1:
            raise ValueError(f"mention not found in the context.")
        return mention_start_pos

    def load_ref_entities(self):
        ref_db_df = pd.read_csv(self.ref_db_file, sep='|', header=None, names=self.ref_db_col_names, dtype=str,
                                index_col=False)
        ref_db_df = ref_db_df.loc[ref_db_df[self.ref_db_lang_col] == self.ref_db_lang_col]
        self.entity_ids = ref_db_df[self.ref_db_id_col]
        self.entity_names = ref_db_df[self.ref_db_str_col]

    def setup_data(self):
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['kg_type'])
        self.data = dataset_factory.create_dataset(kg_path=self.params['data']['kg_path'],
                                                   verbalize=self.params['inf']['verbalize'])

    def setup_model(self):
        self.model = self.load_sentence_model().to(self.device)

    def load_sentence_model(self):
        model = SentenceTransformer(self.params['inf']["model_name"]).to(self.device)
        return model


    @property
    def required_params(self):
        required_params = super().required_params
        required_params["data"] = ["ref_db_file", "kg_type", "kg_path", "tokenizer_name"]
        required_params["inf"] = ["model_name", "save_file", "verbalize", "link_to_ref_db", "top_k_linked_entities",
                                  "ref_db_embeddings_file", "kg_entity_embeddings_file", "kg_relation_embeddings_file",
                                  "mention_embedding_comb_method", "kg_embeddings_rewrite", "ref_db_embeddings_rewrite",
                                  "top_k_entries", "sim_threshold", "use_entity_index", "use_relation_index",
                                  "index_type", "normalize_embeddings", "add_relations_to_entities"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params
