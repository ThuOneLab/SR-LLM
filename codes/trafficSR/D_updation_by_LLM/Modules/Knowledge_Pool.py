from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.docstore.document import Document
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *


class Knowledge():
    def __init__(self, source: str, target: str, key: str, content: str, comment: str = None, reflection: str = None):
        '''
        source: where the knowledge comes from
        key: the key of the knowledge, used to retrieve the knowledge
        content: the content of the knowledge
        comment: evaluation of the knowledge
        '''
        self.source = source
        self.key = key
        self.target = target
        self.content = content
        self.comment = comment
        self.reflection = reflection

    def knowledge2document(self):
        doc = Document(
            page_content=self.key,
            metadata={
                "source": self.source,
                "key": self.key,
                "target": self.target,
                "content": self.content,
                "comment": self.comment,
                "reflection": self.reflection,
            }
        )
        return doc


class Knowledge_Pool():
    def __init__(self, memory_path):
        self.embedding = OpenAIEmbeddings(
            base_url='https://xiaoai.plus/v1',
            api_key=os.environ['OPENAI_API_KEY'],
            # model="text-embedding-3-small"
        )
        self.encode_type = "sce_language"
        self.content = Chroma(
            embedding_function=self.embedding,
            persist_directory=memory_path,
        )
        self.save_path = memory_path
        print("Loaded Knowledge Pool with {} pieces of knowledge".format(
            len(self.content._collection.get(include=['embeddings'])['embeddings'])
        ))
        self.knowledge_target_names = []
        
    def delete_knowledge(self, delete_ids: list):
        delete_ids = [self.content._collection.get(include=['metadatas'])['ids'][di] for di in delete_ids]
        self.content._collection.delete(ids=delete_ids)
        print("Delete {} pieces of knowledge. Now the knowledge pool has {} pieces of knowledge.".format(
            len(delete_ids),
            len(self.content._collection.get(include=['embeddings'])['embeddings'])
        ))

    def add_knowledge(self, knowledge: Knowledge):
        doc = knowledge.knowledge2document()
        pattern = r'\$(.*?)\$'
        self.knowledge_target_names.append(extract_text_between_dollars(pattern, knowledge.target)[0])
        self.content.add_documents([doc])
        print("Add a piece of knowledge. Now the knowledge pool has {} pieces of knowledge.".format(
            len(self.content._collection.get(include=['embeddings'])['embeddings'])
        ))

    def save_target_names(self):
        # 覆盖保存
        with open(self.save_path + '/knowledge_targets_names.json', 'w') as file:
            json.dump(self.knowledge_target_names, file)

    def read_target_names(self):
        if not os.path.exists(self.save_path + '/knowledge_targets_names.json'):
            self.knowledge_target_names = []
            return
        
        with open(self.save_path + '/knowledge_targets_names.json', 'r') as file:
            self.knowledge_target_names = json.load(file)

    def retrieve_knowledge(self, query, fewshot_num=0, source_type=None, print_similarities=True):
        if len(self.content._collection.get(include=['embeddings'])['embeddings']) == 0:
            return []
        if query is None:
            top_k_index = np.random.choice(len(self.content._collection.get(include=['embeddings'])['embeddings']),
                                           fewshot_num)
        else:
            while True:
                try:
                    query_embedding = self.embedding.embed_query(query)
                    break
                except:
                    print("Embedding error, retrying...")
                    import time
                    time.sleep(10)
            Knowledge_embedding = self.content._collection.get(include=['embeddings'])['embeddings']
            similarity = cosine_distance(query_embedding, Knowledge_embedding)
            if print_similarities:
                print("RAG Fewshot Similarities: ", similarity)
            top_k_index = np.argsort(similarity)[::-1][:fewshot_num]

        fewshot_results = []

        pattern = r'\$(.*?)\$'
        for idx in range(len(top_k_index)):
            this_knowledge = self.content._collection.get(include=['metadatas'])['metadatas'][top_k_index[idx]]
            this_fs_result = {}
            this_fs_result['NeedSymbols'] = extract_text_between_dollars(pattern, this_knowledge['key'])
            this_fs_result['TargetSymbol'] = extract_text_between_dollars(pattern, this_knowledge['target'])
            this_fs_result['HumanMessage'] = f"Based on the symbols {this_knowledge['key']}, please combine them to form a new symbol {this_knowledge['target']}." # , which can be used to construct a car-following model

            this_fs_result['AIMessage'] = this_knowledge['content']
            this_fs_result['HumanComment'] = this_knowledge['comment']
            this_fs_result['AIReflection'] = this_knowledge['reflection']
            fewshot_results.append(this_fs_result)

        return fewshot_results