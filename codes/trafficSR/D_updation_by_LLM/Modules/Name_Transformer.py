class Name_Transformer():
    def __init__(self, ):
        pass

    def name_transform(self, fewshot_results, library_info):
        results = [knowledge['TargetSymbol'][0] not in library_info for knowledge in fewshot_results]
        for knowledge_id, knowledge in enumerate(fewshot_results):
            if knowledge['TargetSymbol'][0] not in library_info:
                print(
                    f"{knowledge['TargetSymbol'][0]} has not been built in library")
            else:
                print(
                    f"{knowledge['TargetSymbol'][0]} has been built in library")
        return results
