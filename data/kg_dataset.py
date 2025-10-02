from torch.utils.data import Dataset
import pandas as pd
import os


class SlakeKG(Dataset):
    def __init__(self, kg_path, verbalize=False):
        self.kg_path = kg_path
        self.verbalize = verbalize
        self.entity_names = []
        self.entity_attributes = []
        self.entity_vals = []
        self.entity_entries = []
        self.relation_entities1 = []
        self.relation_types = []
        self.relation_entities2 = []
        self.relation_entries = []
        self.entries = []
        self.load_entity_relations()

    def get_entities(self):
        for entity_entry in zip(self.entity_entries):
            yield entity_entry

    def get_relations(self):
        for relation_entry in zip(self.relation_entries):
            yield relation_entry

    def get_entries(self):
        for entry in zip(self.entries):
            yield entry

    def load_entity_relations(self):
        entity_files = [os.path.join(self.kg_path, "en_organ.csv"), os.path.join(self.kg_path, "en_disease.csv")]
        for entity_file in entity_files:
            entity_df = pd.read_csv(entity_file, sep="#", header=0)
            entity_cols = list(entity_df.columns)
            self.entity_names.extend(entity_df[entity_cols[0]].tolist())
            self.entity_attributes.extend(entity_df[entity_cols[1]].tolist())
            self.entity_vals.extend(entity_df[entity_cols[2]].tolist())
        self.entity_entries = self.combine_entity_entries()
        relation_files = [os.path.join(self.kg_path, "en_organ_rel.csv")]
        for relation_file in relation_files:
            relation_df = pd.read_csv(relation_file, sep="#", header=0)
            relation_cols = list(relation_df.columns)
            self.relation_entities1.extend(relation_df[relation_cols[0]].tolist())
            self.relation_types.extend(relation_df[relation_cols[1]].tolist())
            self.relation_entities2.extend(relation_df[relation_cols[2]].tolist())
        self.relation_entries = self.combine_relation_entries()
        self.entries = self.entity_entries + self.relation_entries

    def combine_entity_entries(self):
        entity_entries = []
        for entity, entity_attribute, entity_val in zip(self.entity_names, self.entity_attributes, self.entity_vals):
            if self.verbalize:
                entity_entries.append(self.verbalize_entity(entity, entity_attribute, entity_val),)
            else:
                entity_entries.append((entity, entity_attribute, entity_val))
        return entity_entries

    def combine_relation_entries(self):
        relation_entries = []
        for relation_entity1, relation_type, relation_entity2 in zip(self.relation_entities1, self.relation_types, self.relation_entities2):
            if self.verbalize:
                relation_entries.append(self.verbalize_relation(relation_entity1, relation_type, relation_entity2),)
            else:
                relation_entries.append((relation_entity1, relation_type, relation_entity2))
        return relation_entries

    def get_entity_by_index(self, index):
        return self.entity_entries[index]

    def get_relation_by_index(self, index):
        return self.relation_entries[index]

    def verbalize_entity(self, entity, entity_attribute, entity_val):
        return f"The {entity_attribute} of {entity} is {entity_val}"

    def verbalize_relation(self, relation_entity1, relation_type, relation_entity2):
        return f"{relation_entity1} {relation_type} {relation_entity2}"

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]
