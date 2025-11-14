class Directly_Builder():
    def __init__(self, ):
        pass

    def not_achieved(self, fewshot_results, library_info):
        '''查看是否RAG出来的知识已经在库里面了'''
        name_results = [knowledge['TargetSymbol'][0] not in library_info for knowledge in fewshot_results]
        for knowledge_id, knowledge in enumerate(fewshot_results):
            if knowledge['TargetSymbol'][0] not in library_info:
                print(
                    f"Name {knowledge['TargetSymbol'][0]} has not been built in library")
            else:
                print(
                    f"Name {knowledge['TargetSymbol'][0]} has been built in library")

        return name_results

    def can_be_built_directly(self, fewshot_results, symbol_info, library_info):
        '''查看RAG出来的组合，能不能被库里其他组件构造出来NeedSymbols'''
        fewshot_num = len(fewshot_results)
        results = [True] * fewshot_num
        library_symbols = library_info.keys()
        for knowledge_id, knowledge in enumerate(fewshot_results):
            if symbol_info['name'] not in knowledge['NeedSymbols']:
                '''如果最好的符号不在NeedSymbols里面，那么就不能直接构造'''
                results[knowledge_id] = False
                print(
                    f"Best symbol {symbol_info['name']} not in NeedSymbols, so {knowledge['TargetSymbol'][0]} cannot be built directly")
                continue
            
            '''如果NeedSymbols里面有一个不在库里面，那么就不能直接构造'''
            for i in range(len(knowledge['NeedSymbols'])):
                if knowledge['NeedSymbols'][i] not in library_symbols:
                    results[knowledge_id] = False
                    print(
                        f"Need symbol {knowledge['NeedSymbols'][i]} not in library, so {knowledge['TargetSymbol'][0]} cannot be built directly")
                    break
            if results[knowledge_id]:
                print(
                    f"Target {knowledge['TargetSymbol'][0]} can be built directly")
        return results

    def can_be_build_directly_examples(self, fewshot_results, build_directly):
        directly_targetSymbols = []
        directly_built_knowledge_ids = []
        for knowledge_id, knowledge in enumerate(fewshot_results):
            if build_directly[knowledge_id]:
                directly_targetSymbols.append(knowledge['TargetSymbol'][0])
                directly_built_knowledge_ids.append(knowledge_id)
        return directly_targetSymbols, directly_built_knowledge_ids
